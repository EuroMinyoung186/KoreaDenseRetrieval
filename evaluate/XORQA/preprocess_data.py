import jsonlines
import json
from statistics import mean
import argparse
from tqdm import tqdm
import deepl
from typing import List



def read_jsonlines(path: str):
    lines = []
    trans_dictionary = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            if obj['lang'] == 'ko':
                lines.append({'id' : obj['id'], 'question' : obj['question'], 'answers' : obj['answers']})
    
    with open('./kor_dev_dataset.jsonl', encoding='utf-8', mode='w') as file:
        for i in lines:
            file.write(json.dumps(i) + '\n')        

def read_goldparagraphs(path: str):
    lines = []
    with open(path, 'rb') as f:
        squad_dataset = json.load(f)
    squad_dataset = squad_dataset['data']
    
    goldparagraphs_dict = []
    
    for data in squad_dataset:
        title = data['title'].split('title:')[1].split('_')[0].strip()
        paragraphs = data['paragraphs']
        contexts = []
        ids = []
        splits = []
        answers = []
        answer_start = []
        
        for para in paragraphs:
            contexts.append(para['context'])
            ids.append(para['qas'][0]['id'])
            answers.append(para['qas'][0]['answers'][0]['text'])
            answer_start.append(para['qas'][0]['answers'][0]['answer_start'])
            splits.append(para['qas'][0]['split'])
        
        for i in range(len(contexts)):
            goldparagraphs_dict.append({'title' : title, 'context' : contexts[i], 'id' : ids[i], 'split' : splits[i],
                                        'answer' : answers[i], 'answer_start' : answer_start[i]})
        
    return goldparagraphs_dict
        
def make_json():
    json_file = {}
    json_file['data'] = []   
    answers = read_goldparagraphs('./gp_squad_dev_data.json')
    
    cnt = 0
    with jsonlines.open('./kor_dev_dataset.jsonl') as reader:
        for obj in reader:
            title = ""
            paragraphs = []
            for answer in answers:
                if answer['id'] == obj['id']:
                    title = answer['title']
                    cnt += 1
                    paragraphs.append({'qas' : [{'answer' : [{'text': answer['answer'], 'answer_start' : answer['answer_start']}],
                                                 'question' : obj['question'], 'id' : obj['id']}], 'context' : answer['context']})
            json_file['data'].append({'title' : title, 'paragraphs' : paragraphs})
                    
                
    with open('./xor_dev_retrieve_eng_span_en.jsonl', 'w', encoding='utf-8') as file:
        json.dump(json_file, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    read_jsonlines('./xor_dev_retrieve_eng_span_v1_1.jsonl')
    make_json()


