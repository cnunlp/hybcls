import random
import json
from tqdm.auto import tqdm
from utils import load_synonym_dict, load_json_list, load_json_line

MAX_PIVOT_NUM = 10000
random.seed(2024)

syn_words = load_synonym_dict(with_flag=True)
word_list = [item['word'] for item in load_json_list()]
idiom_set = set([item['name'] for item in load_json_line()])
random.shuffle(word_list)

with open('./output/result_sample_pivot.json','w',encoding='utf-8') as wf:
    counter = 0
    for word in tqdm(word_list):
        if len(word) not in range(2,5):
            continue
        for i in range(len(syn_words)):
            if syn_words[i][0] == 0 and word in syn_words[i][1]:
                word_type = 'idiom' if word in idiom_set else 'common'
                wf.write(json.dumps({'word':word, 'type':word_type},ensure_ascii=False)+'\n')
                syn_words[i][0] = 1
                counter += 1
                break
        if counter == MAX_PIVOT_NUM:
            break
    
