# 创建训练集与测试集

import re
import json
from tqdm.auto import tqdm
import pandas as pd
from zhon.hanzi import characters, punctuation
from ltp import StnSplit
from utils import load_synonym_dict, load_json_line, load_json_list, distilled_data_path, test_dataset_path


MAX_INTER_LEN = 100 # 释义长度限制

prompt = '句子：{sentence}\n任务：针对句子中的#{word}#，请你给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：'
prompt_ria = '句子：{sentence}\n#{word}#的解释：{interpretation}\n任务：针对句子中的#{word}#，请你参考其解释，给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：'

crawler = 'google'
ood_dictionary_path = f'./output/crawler/{crawler}_result.json'

distilled_data = load_json_line(distilled_data_path)
test_dataset = pd.read_csv(test_dataset_path, names=['sentence','source','pos','index','candidates'], sep='\t')
complex_word_test = set(test_dataset['source'])
word_dictionary = pd.DataFrame(load_json_list())
word_set = set(word_dictionary['word'])
idiom_dictionary = pd.DataFrame(load_json_line())
idiom_set = set(idiom_dictionary['name'])
ood_dictionary = pd.DataFrame(load_json_line(ood_dictionary_path))
syn_set = set([word for word_list in load_synonym_dict() for word in word_list])

common_words = ( word_set | syn_set ) - idiom_set 
idioms = idiom_set
ood_words = complex_word_test - word_set - idiom_set - syn_set

word_type_zh = {'common':'普通词', 'idiom':'成语', 'ood':'词典外词'}

def check_word_type(word):
    if word in common_words:
        return 'common'
    if word in idioms:
        return 'idiom'
    if word in ood_words:
        return 'ood'    
    print(f'Unknown word type: {word}')
    return None
    

def preprocessing_interpretation(interpretation, word_type):
    if word_type == 'ood':
        zh_text = ''.join(re.findall(f"[0-9a-zA-Z{punctuation}{characters}]",interpretation))
    else:
        zh_text = ''.join(re.findall(f"[{punctuation}{characters}]",interpretation))
    sentences = StnSplit().split(zh_text)
    result = ''
    for sentence in sentences:
        if len(result+sentence) < MAX_INTER_LEN:
            result += sentence
    if result and (len(re.findall(f"[{characters}]",result)) / len(result)) < 0.5:
        result = ''

    return result


def search_word_interpretation(word, word_type):
    if word_type == 'common':
        target_interpretation = word_dictionary.loc[word_dictionary['word']==word]
        if not target_interpretation.empty and not pd.isna(target_interpretation.iloc[0]['explanation']):
            target_interpretation = target_interpretation.iloc[0]['explanation'].replace('～',word)
        else:
            target_interpretation = ''
    elif word_type == 'idiom':
        target_interpretation = idiom_dictionary.loc[idiom_dictionary['name']==word].iloc[0]['content']
    elif word_type == 'ood':
        interpretations = ood_dictionary.loc[ood_dictionary['word']==word].iloc[0]['interpretation']
        target_interpretation = ''
        for interpretation in interpretations:
            interpretation = preprocessing_interpretation(interpretation, word_type)
            if not target_interpretation and interpretation:
                target_interpretation = interpretation
            if word in interpretation:
                target_interpretation = interpretation
                break
        return target_interpretation
    else:
        print(f'Unknown word type:{word_type}')
        return None

    return preprocessing_interpretation(target_interpretation, word_type) if target_interpretation else ''
    

def get_candidate_type(word_type):
    if word_type == 'common':
        return '词'
    if word_type in ['idiom', 'ood']:
        return '词或短语'


def check_word_pairs(word_pairs):
    result_word_pairs = []
    for word_pair in word_pairs:
        complex_word = word_pair[0]
        simple_word = word_pair[1]
        if complex_word in idiom_set or (complex_word in word_set and simple_word in word_set):
            result_word_pairs.append(word_pair)

    return result_word_pairs


def extract_word_pairs_from_answer(pivot_word, answer):
    paragraphs = [paragraph for paragraph in re.split(r'（[1-3]）',answer) if paragraph]
    assert len(paragraphs) == 3 and f"#{pivot_word}#" in paragraphs[0]
    pivot_complexity = re.findall(f"#{pivot_word}#的词汇难度等级为：(初级|中级|高级)。",paragraphs[0])[0]
    pivot_sentence = re.findall(r"句子：(.+)\n",paragraphs[1])[0]
    try:
        assert f"#{pivot_word}#" in pivot_sentence
    except:
        assert pivot_word in pivot_sentence
        pivot_sentence = pivot_sentence.replace(pivot_word, f"#{pivot_word}#")
    multi_level = {'初级':[], '中级':[], "高级":[]}
    multi_level_sentences = re.split(r'[初中高]级：', paragraphs[2])
    multi_level_sentences = [sentence for sentence in multi_level_sentences if len(sentence) > 5]
    assert len(multi_level_sentences) == 3
    for i, sentence in enumerate(multi_level_sentences):
        substitution = re.findall(r'#([\u4e00-\u9fa5]+)#', sentence)
        if len(substitution) == 1 and substitution[0] not in complex_word_test:
            multi_level[list(multi_level.keys())[i]].append(substitution[0])            
    if pivot_word not in complex_word_test:
        multi_level[pivot_complexity].append(pivot_word)
    
    word_pairs = []
    for advanced_word in multi_level['高级']:
        for medium_word in multi_level['中级']:
            word_pairs.append((advanced_word, medium_word))
        for basic_word in multi_level['初级']:            
            word_pairs.append((advanced_word, basic_word))
    for medium_word in multi_level['中级']:
        for basic_word in multi_level['初级']:            
            word_pairs.append((medium_word, basic_word))

    return check_word_pairs(word_pairs), pivot_sentence

train_data = []
train_data_ria = []
with open('./output/train_dataset.json', 'w', encoding='utf-8') as wf, open('./output/train_dataset_ria.json', 'w', encoding='utf-8') as wf_ria:
    for data in tqdm(distilled_data, desc='Processing training dataset'):
        complex2simple_word_pairs, pivot_sentence = extract_word_pairs_from_answer(data['word'], data['answer'])
        for word_pair in complex2simple_word_pairs:
            answer = word_pair[1]
            source = word_pair[0]
            word_type = check_word_type(word_pair[0])
            interpretation = search_word_interpretation(word_pair[0], word_type)
            question = prompt\
                .replace('{sentence}', pivot_sentence.replace(data['word'],source))\
                .replace('{word}', source)\
                .replace('{candidate_type}', get_candidate_type(word_type))
            if not interpretation:
                question_ria = question
            else:
                question_ria = prompt_ria\
                .replace('{sentence}', pivot_sentence.replace(data['word'],source))\
                .replace('{word}', source)\
                .replace('{candidate_type}', get_candidate_type(word_type))\
                .replace('{interpretation}', interpretation)
            write_data = {'question':question, 'answer':answer, 'source':source, 'type':word_type, 'interpretation':interpretation}
            write_data_ria = {'question':question_ria, 'answer':answer, 'source':source, 'type':word_type, 'interpretation':interpretation}
            train_data.append(write_data)
            train_data_ria.append(write_data_ria)
            wf.write(json.dumps(write_data, ensure_ascii=False)+'\n')
            wf_ria.write(json.dumps(write_data_ria, ensure_ascii=False)+'\n')

with open(f'./output/test_dataset_{crawler}.json', 'w', encoding='utf-8') as wf:
    for data in tqdm(test_dataset.to_dict('records'), desc='Processing test dataset'):
        source = data['source']
        word_type = check_word_type(source)
        interpretation = search_word_interpretation(source, word_type)
        wf.write(json.dumps({'sentence':data['sentence'].replace(' ','').replace(source,f"#{source}#"), 'source':source, 'gold':data['candidates'].split(' '), 'type':word_type, 'interpretation':interpretation}, ensure_ascii=False)+'\n')






