import torch
from transformers import BertTokenizerFast, AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import os
import re

class ModelDownloader:
    def __init__(self):
        self.pii_model = "Isotonic/distilbert_finetuned_ai4privacy_v2"
        self.chinese_model = "ckiplab/bert-base-chinese-ner"
    
    def create_text_folder(self):
        if 'txt_file' not in os.listdir('./'):
            os.mkdir('./txt_file')
        if 'result' not in os.listdir('./'):
            os.mkdir('./result')

    def download_model(self):
        if self.pii_model not in os.listdir('./'):
            print(f'{self.pii_model} does not exist, downloading...')
            model = AutoModelForTokenClassification.from_pretrained(self.pii_model)
            tokenizer = AutoTokenizer.from_pretrained(self.pii_model)
            local_directory = "./distilbert_finetuned_ai4privacy_v2r"
            model.save_pretrained('./distilbert_finetuned_ai4privacy_v2')
            tokenizer.save_pretrained('./distilbert_finetuned_ai4privacy_v2')
        else:
            print(f'{self.pii_model} already exists')
        if self.chinese_model not in os.listdir('./'):
            print(f'{self.chinese_model} does not exist, downloading...')
            model = AutoModelForTokenClassification.from_pretrained(self.chinese_model)
            tokenizer = AutoTokenizer.from_pretrained(self.chinese_model)
            local_directory = "./bert-base-chinese-ner"
            model.save_pretrained(local_directory)
            tokenizer.save_pretrained(local_directory)
        else:
            print(f'{self.chinese_model} already exists')
    
class EnglishPII:
    def __init__(self):
        
        self.chinese_pattern = r"[\u4e00-\u9fff]+\d+[\u4e00-\u9fff]+|[\u4e00-\u9fff]+"
        self.pretrained_model = AutoModelForTokenClassification.from_pretrained('./distilbert_finetuned_ai4privacy_v2')
        self.tokenizer = AutoTokenizer.from_pretrained('./distilbert_finetuned_ai4privacy_v2')
        self.model = pipeline("token-classification", model=self.pretrained_model, tokenizer=self.tokenizer, device=-1)
        
    def get_english_pii(self, text):
        model = self.model
        chinese_pattern = self.chinese_pattern
        chinese_texts = re.findall(chinese_pattern, text)
        placeholder = "<CHINESE_WORD_PLACEHOLDER>"
        processed_text = re.sub(chinese_pattern, placeholder, text)
        output = model(processed_text, aggregation_strategy="first")
        output.reverse()
        for x in output:
            processed_text = processed_text[:x['start']] + '[' + x['entity_group'] + ']' + processed_text[x['end']:]
        return processed_text, chinese_texts


class ChineseNER:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('./bert-base-chinese-ner')
        self.model = AutoModelForTokenClassification.from_pretrained('./bert-base-chinese-ner')
    
    def get_chinese_ner(self,text):
        chinese_tokenizer = self.tokenizer
        chinese_model = self.model
        inputs = chinese_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = chinese_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        labels = [chinese_model.config.id2label[pred.item()] for pred in predictions[0]]
        tokens = chinese_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        result = [(token, label) for token, label in zip(tokens, labels) if token != '[PAD]']
        return result
    
    def extract_entities(self, tagged_list):
        entities = {}
        current_entity = []
        current_label = None
        for word, label in tagged_list:
            if label.startswith('B-'):
                if current_entity:
                    entities[''.join(current_entity)] = current_label
                current_entity = [word]
                current_label = label[2:]
            elif label.startswith('I-') and current_entity:
                current_entity.append(word)
            elif label.startswith('E-') and current_entity:
                current_entity.append(word)
                entities[''.join(current_entity)] = current_label
                current_entity = []
                current_label = None
            elif label == 'O':
                continue
        if current_entity:
            entities[''.join(current_entity)] = current_label
        return entities
    
    def run(self, text):
        tagged_list = self.get_chinese_ner(text)
        entities = self.extract_entities(tagged_list)
        return entities
    
class PII:
    def __init__(self):
        self.english_pii = EnglishPII()
        self.chinese_ner = ChineseNER()
    
    def run(self, text):
        english_pii = self.english_pii
        chinese_ner = self.chinese_ner
        english_text, chinese_texts_list = english_pii.get_english_pii(text)
        chinese_entity_list = []
        for chinese_word in chinese_texts_list:
            chinese_entities = chinese_ner.run(chinese_word)
            if len(chinese_entities) == 0:
                chinese_entity_list.append(chinese_word)
            else:
                single_chinese_entity = []
                for entity in chinese_entities:
                    ner_name = f'[{chinese_entities[entity]}]'
                    single_chinese_entity.append(ner_name)
                chinese_entity_list.append(''.join(single_chinese_entity))
        for chinese in chinese_entity_list:
            english_text = english_text.replace("<CHINESE_WORD_PLACEHOLDER>", chinese, 1)
        return english_text
    
if __name__ == '__main__':
    ModelDownloader().create_text_folder()
    ModelDownloader().download_model()
    if len(os.listdir('./txt_file/')) == 0:
        print('Please put the txt files in the txt_file folder')
    else:
        for text in os.listdir('./txt_file/'):
            with open(f'./txt_file/{text}', 'r') as f:
                content = f.read()
            pii_result = PII().run(content)
            with open(f'./result/{text}', 'w') as f:
                f.write(pii_result)
            print(f'PII on {text} has been extracted and saved to ./result/{text}')
