import os

import sys
import math
import json
import time
import torch

from tqdm.auto import tqdm
import pandas as pd

import utils
import re

from src.llmtuner.model import load_model, load_tokenizer
from src.llmtuner.hparams import get_train_args

use_ria = False
use_syn_rules = True

extract_word = True
syn_dict = utils.load_synonym_dict('/data00/home/zhxiao/work/chatyuan/data/syndict.json')
test_file_path = '/data00/home/zhxiao/work/chatyuan/data/test_dataset_google.json'
batch_size = 4
max_length = 512

prompt = '句子：{sentence}\n任务：针对句子中的#{word}#，请你给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：'
prompt_ria = '句子：{sentence}\n#{word}#的解释：{interpretation}\n任务：针对句子中的#{word}#，请你参考其解释，给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：'
word_type_zh = {'common':'普通词', 'idiom':'成语', 'ood':'词典外词'}


def get_candidate_type(word_type):
    if word_type == 'common':
        return '词'
    if word_type in ['idiom', 'ood']:
        return '词或短语'


def assemble_syn_model(word_list, syn_list):
    
    # if len(word_list) == 0:
    #     return 'ERROR'
    
    model_score = []
    syn_score = []
    for i,word in enumerate(word_list):
        model_score.append(1/(i+1))
        if word in syn_list:
            syn_score.append(1)
        else:
            syn_score.append(0)
    score = []
    for i in range(len(model_score)):
        score.append([word_list[i],model_score[i]+syn_score[i]])
    score.sort(key = lambda x : x[1], reverse = True)

    return score[0][0]


def get_word_from_generation(sentence):
    
    pattern = r'回答：[\s]{0,1}([\u4e00-\u9fa5]{1,8})'
    word_find = re.findall(pattern,sentence)
    if len(set(word_find)) == 1:
        return word_find[0]
    
    return None


def check_idiom_span(predict_idiom, golds):

    for gold in golds:
        if gold in predict_idiom:
            return True
        
    return False


def evalute_from_decode_sentences(pred_words, source_data, print_error=False):
    
    source_words = source_data['source']
    type_words = source_data['type']
    gold_words = source_data['gold']

    assert len(pred_words) == len(source_data)
    result = {_type:{'total':0,'acc':0, 'pre':0} for _type in list(word_type_zh.keys())}
    
    for sub, source, gold, type in zip(pred_words, source_words, gold_words, type_words):
        if sub==source or sub in gold:
            result[type]['pre'] += 1
        if sub!=source and sub in gold:
            result[type]['acc'] += 1
        if print_error and sub != source and sub not in gold:
            print('Error:',sub,'    ',source,'    ',type)
        
        result[type]['total'] += 1

    return result


print('loading test dataset ...')
test_data = utils.load_test_data(test_file_path)
test_data_df = pd.DataFrame(test_data)

print(len(test_data_df))
test_data = test_data_df.to_dict('records')


model_args, data_args, training_args, finetuning_args, general_args = get_train_args()
tokenizer = load_tokenizer(model_args)
tokenizer.padding_side = 'left'
model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

device = utils.get_device(use_cuda=True)
model.to(device)

predict_sentences = []
flag = True
for start in tqdm(range(0, len(test_data), batch_size)):
    batch = test_data[start: start + batch_size]
    questions_batch = []
    source_batch = []
    for example in batch:
        word = example['source']
        source_batch.append(word)
        sentence = example['sentence']
        info = example['interpretation']
        word_type = example['type']
        if use_ria and info:
                question = prompt_ria\
                    .replace('{sentence}',sentence)\
                    .replace('{word}',word)\
                    .replace('{candidate_type}',get_candidate_type(word_type))\
                    .replace('{interpretation}',info)
        else:
            question = prompt\
                .replace('{sentence}',sentence)\
                .replace('{word}',word)\
                .replace('{candidate_type}',get_candidate_type(word_type))

        questions_batch.append(question.replace(' ',''))
        if flag:
            print(questions_batch)
            flag = False
            
    input_texts = [question for question in questions_batch]

    inputs = tokenizer(input_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)

    outputs = model.generate(**inputs, do_sample=False, max_length=max_length, num_beams=10, early_stopping=True, num_return_sequences=10)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [word for word in outputs]
    predict_word = [outputs[i*10:(i+1)*10] for i in range(0, batch_size)]
    predict_word = [words for words in predict_word if len(words) > 0]
    
    if extract_word:
        predict_word = [[temp for word in word_list if (temp:=get_word_from_generation(word))] for word_list in predict_word]

    # print(predict_word)
    # print('source predict len : ', len(source_batch), len(predict_word))
    for source, words in zip(source_batch, predict_word):
        if len(words) == 0:
            ranked_word = 'Error'
        elif source in syn_dict.keys() and use_syn_rules:
            ranked_word = assemble_syn_model(words, syn_dict[source])
        else:
            ranked_word = words[0]
        predict_sentences.append(ranked_word)

test_result = evalute_from_decode_sentences(predict_sentences, test_data_df, print_error=False)
total = 0
total_pre = 0
total_acc = 0
print('='*30,'RESULT','='*30)
for word_type in list(word_type_zh.keys()):
    pre = test_result[word_type]['pre'] / test_result[word_type]['total'] if test_result[word_type]['total'] else 0
    acc = test_result[word_type]['acc'] / test_result[word_type]['total'] if test_result[word_type]['total'] else 0
    print(f"type : {word_type} | pre : {round(pre,3)} | acc : {round(acc,3)}")
    total += test_result[word_type]['total']
    total_pre += test_result[word_type]['pre']
    total_acc += test_result[word_type]['acc']
print(f"total : pre : {round(total_pre/total,3)} | acc : {round(total_acc/total,3)}")

output_dir = './output/predict_Qwen_ria.json' if use_ria else './output/predict_Qwen.json'
with open(output_dir,'w',encoding='utf-8') as wf:
    for data, predict in zip(test_data, predict_sentences):
        data.update({'answer':predict, 'model':'Qwen_RIA' if use_ria else 'Qwen'})
        wf.write(json.dumps(data,ensure_ascii=False)+'\n')

