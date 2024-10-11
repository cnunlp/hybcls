import os
import json
import torch

from tqdm.auto import tqdm
import pandas as pd
import utils
import re

from src.llmtuner.model import load_model, load_tokenizer
from src.llmtuner.hparams import get_train_args


use_ria = False
use_syn_rules = False
print_span = True

extract_word = True
top_n = 10
syn_dict = utils.load_synonym_dict('/data00/home/zhxiao/work/chatyuan/data/syndict.json')
test_file_path = '/data00/home/zhxiao/work/chatyuan/data/test_dataset_google.json'
batch_size = 4
max_length = 512
word_type_zh = {'common':'词', 'idiom':'词或短语', 'ood':'词或短语'}

system_prompt = '请你根据要求执行词汇简化任务，以下是任务示例：\n{few_shot_prompt}请你完成：\n{question}'
prompt_template = '句子：{sentence}\n要求：针对句子中的#{source}#，请你给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：{answer}'
prompt_ria_template = '句子：{sentence}\n#{source}#的解释：{interpretation}\n要求：针对句子中的#{source}#，请你参考其解释，给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：{answer}'

few_shot_examples = [
    {'sentence':'她下班时我们为她端上一杯浓浓的咖啡，说句#温馨#的话语。', 'source':'温馨' ,'candidate_type':'词', 'interpretation':'温暖芳香春夜温馨｜温馨的花园｜午后温馨薄暮凉；温暖。' ,'answer':'温暖'},
    {'sentence':'听了奶奶的一番话，我#恍然大悟#，一头扑在奶奶的怀里。', 'source':'恍然大悟' ,'candidate_type':'词或短语', 'interpretation':'恍然：猛然清醒的样子；悟：心里明白。形容一下子明白过来。' ,'answer':'突然明白'},
    # {'sentence':'他经常#吐槽#我们的老板，让大家都觉得很好笑。', 'source':'吐槽' ,'candidate_type':'词或短语', 'interpretation':'一般是指从对方的语言或行为中找到一个漏洞或关键词作为切入点，发出带有调侃意味的感慨或疑问。' ,'answer':'抱怨'},
    ]

few_shot_prompt = ''
for example in few_shot_examples:
    prompt_example = prompt_ria_template.replace('{interpretation}',example['interpretation']) if use_ria else prompt_template
    prompt_example = prompt_example\
        .replace('{sentence}',example['sentence'])\
        .replace('{source}',example['source'])\
        .replace('{candidate_type}',example['candidate_type'])\
        .replace('{answer}',f"#{example['answer']}#")
    few_shot_prompt += f'{prompt_example}\n\n'

def get_candidate_type(word_type):
    if word_type == 'common':
        return '词'
    if word_type in ['idiom', 'ood']:
        return '词或短语'
    
def assemble_syn_model(word_list, syn_list):

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

    pattern = r'回答： ?#{1,2}([\u4e00-\u9fa5]{1,8})#{1,2}\n?$'
    word_find = re.findall(pattern,sentence.split('\n')[-1])
    if len(set(word_find)) == 1:
        return word_find[0]
    
    return None


def check_span(predict, golds):

    for gold in golds:
        if gold in predict:
            return True
        
    return False


def evalute_from_decode_sentences(pred_words, source_data, print_error=False, print_span=True):
    
    source_words = source_data['source']
    type_words = source_data['type']
    gold_words = source_data['gold']

    assert len(pred_words) == len(source_data)
    result = {_type:{'total':0,'acc':0, 'pre':0} for _type in list(word_type_zh.keys())}
    if print_span: 
        result_span = {_type:{'total':0,'acc':0, 'pre':0} for _type in list(word_type_zh.keys())}
    
    for sub, source, gold, type in zip(pred_words, source_words, gold_words, type_words):
        if sub in gold:
            result[type]['acc'] += 1
            result[type]['pre'] += 1
        elif sub==source:
            result[type]['pre'] += 1
        elif print_error:
            print('Error:',[sub],'    ',source,'    ',type)

        result[type]['total'] += 1

        if print_span:
            if sub == source:
                result_span[type]['pre'] += 1
            elif check_span(sub, gold):
                result_span[type]['acc'] += 1
                result_span[type]['pre'] += 1
        
            result_span[type]['total'] += 1

    return result, result_span if print_span else None

def print_result(test_result):
    total = 0
    total_pre = 0
    total_acc = 0
    for type in list(word_type_zh.keys()):
        pre = test_result[type]['pre'] / test_result[type]['total'] if test_result[type]['total'] else 0
        acc = test_result[type]['acc'] / test_result[type]['total'] if test_result[type]['total'] else 0
        total += test_result[type]['total']
        total_pre += test_result[type]['pre']
        total_acc += test_result[type]['acc']
        print(f"type : {type} | pre : {round(pre,3)} | acc : {round(acc,3)}")
    print(f"total : pre : {round(total_pre/total,3)} | acc : {round(total_acc/total,3)}")


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
        source = example['source']
        source_batch.append(source)
        sentence = example['sentence']
        interpretation = example['interpretation']
        word_type = example['type']
        question = prompt_ria_template.replace('{interpretation}',interpretation) if use_ria else prompt_template
        question = question\
            .replace('{sentence}',sentence)\
            .replace('{source}',source)\
            .replace('{candidate_type}',get_candidate_type(word_type))\
            .replace('{answer}','')
        question = system_prompt\
            .replace('{few_shot_prompt}',few_shot_prompt)\
            .replace('{question}',question)

        questions_batch.append(question.replace(' ',''))
        if flag:
            print(questions_batch)
            flag = False

    inputs = tokenizer(questions_batch, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)

    if top_n > 1:
        outputs = model.generate(**inputs, do_sample=False, max_length=max_length, num_beams=top_n, early_stopping=True, num_return_sequences=top_n)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predict_word = [outputs[i*top_n:(i+1)*top_n] for i in range(0, batch_size)]
        predict_word = [words for words in predict_word if len(words) > 0]
        # print(predict_word)   
        if extract_word:
            predict_word = [[get_word_from_generation(word) for word in word_list if get_word_from_generation(word)] for word_list in predict_word]

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
    else:
        outputs = model.generate(**inputs, do_sample=False, max_length=max_length)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for source, word in zip(source_batch, outputs):
            if extract_word:
                word = get_word_from_generation(word)
            if not word:
                word = 'Error'
            predict_sentences.append(word)

test_result, test_result_span = evalute_from_decode_sentences(predict_sentences, test_data_df, print_span=print_span)

print("*** accurate result ***")
print_result(test_result)
if print_span:
    print("*** span result ***")
    print_result(test_result_span)

with open('temp_result.json','w',encoding='utf-8') as wf:
    for predict, data in zip(predict_sentences, test_data):
        data['predict'] = predict
        wf.write(json.dumps(data,ensure_ascii=False)+'\n')


