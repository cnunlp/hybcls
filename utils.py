import random
import json
import hashlib
import string
from tqdm.auto import tqdm
import openai
import os

openai.api_base = ''
openai.api_key = ''

temperature = 0
topp = 0.9

base_dir = os.path.dirname(os.path.abspath(__file__))
syn_path = os.path.join(base_dir,'src/HIT_cilin.txt')
word_dict_path = os.path.join(base_dir,'src/word_dict.json')
idiom_dict_path = os.path.join(base_dir,'src/idiom_dict.json')
test_dataset_path = os.path.join(base_dir,'src/annotation_data.csv')
distilled_data_path = os.path.join(base_dir,'output/distillation/gpt-4o/result.json')

word_types = ['common', 'idiom', 'ood']

def load_synonym_dict(file_path=syn_path, with_flag=False):
    syn_words = []
    with open(file_path,'r',encoding='gbk') as rf:
        for line in rf:
            words = line.strip().split(' ')
            if words[0][-1] == '=':
                if with_flag:
                    syn_words.append([0,words[1:]])
                else:
                    syn_words.append(words[1:])

    return syn_words


def load_json_list(file_path=word_dict_path):
    with open(file_path,'r',encoding='utf-8') as rf:        
        result = json.load(rf)

    return result


def load_json_line(file_path=idiom_dict_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as rf:
        for line in rf:
            result.append(eval(line))

    return result


def generate_from_chatgpt(question, model_name):    
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature = temperature,
            top_p = topp,
        )
    except:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature = temperature,
            top_p = topp,
        )
        
    answer = response['choices'][0]['message'].get('content')
    if not answer:
        answer = 'ERROR'

    return answer


def err_call_back(err):
    print(f'出错啦：{str(err)}')


def evalute(sources, predicts, golds, types, print_result: bool = False, print_error: bool = False):
    assert len(sources) == len(predicts)
    result = {_type:{'total':0, 'acc':0, 'pre':0} for _type in word_types + ['all']}
    
    for source, predict, gold, type in zip(sources, predicts, golds, types):
        if predict in gold:
            result[type]['acc'] += 1            
            result[type]['pre'] += 1
            result['all']['acc'] += 1
            result['all']['pre'] += 1
        elif predict == source:
            result[type]['pre'] += 1
            result['all']['pre'] += 1
        elif print_error:
            print(f'Type: {type} | Source: {source} | Predict: {predict} | Gold: {gold}')

        result[type]['total'] += 1
        result['all']['total'] += 1    

    for type in result.keys():
        result[type]['pre'] = round(result[type]['pre'] / result[type]['total'],3) if result[type]['total'] else 0
        result[type]['acc'] = round(result[type]['acc'] / result[type]['total'],3) if result[type]['total'] else 0
        if print_result:
            print(f"type : {type} | pre : {result[type]['pre']} | acc : {result[type]['acc']} | total : {result[type]['total']}")

    return result


def evalute_span(sources, predicts, golds, types, print_result: bool = False, print_error: bool = False):
    assert len(sources) == len(predicts)
    result = {_type:{'total':0, 'acc':0, 'pre':0} for _type in word_types + ['all']}
    
    for source, predict, gold, type in zip(sources, predicts, golds, types):
        acc_flag = 0
        if predict == source:
            result[type]['pre'] += 1
            result['all']['pre'] += 1
        else:
            for _gold in gold:
                if _gold in predict:                   
                    result[type]['pre'] += 1
                    result[type]['acc'] += 1
                    result['all']['acc'] += 1
                    result['all']['pre'] += 1
                    acc_flag = 1
                    break

        if acc_flag == 0 and print_error:
            print(f'Type: {type} | Source: {source} | Predict: {predict} | Gold: {gold}')

        result[type]['total'] += 1
        result['all']['total'] += 1    

    for type in result.keys():
        result[type]['pre'] = round(result[type]['pre'] / result[type]['total'],3) if result[type]['total'] else 0
        result[type]['acc'] = round(result[type]['acc'] / result[type]['total'],3) if result[type]['total'] else 0
        if print_result:
            print(f"type : {type} | pre : {result[type]['pre']} | acc : {result[type]['acc']} | total : {result[type]['total']}")

    return result


def generate_random_md5():
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    md5 = hashlib.md5()
    md5.update(random_str.encode('utf-8'))
    md5_value = md5.hexdigest()
    
    return md5_value