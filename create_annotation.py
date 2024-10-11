# 在获取所有模型的结果后，创建人工标注文档
import sqlite3

from collections import Counter
from utils import generate_random_md5
import pandas as pd


models_md5_dict = {'BERT-LS':[], 'ChatYuan':[], 'ChatYuan_RIA':[], 'ChatGLM':[], 'ChatGLM_RIA':[], 'Qwen':[], 'Qwen_RIA':[], 'GPT4':[]}
models_predict_path = {
    'BERT-LS':'./output/fewshot/zeroshot_result_bertls.json',
    'ChatYuan':'./output/supervision/predict_chatyuan.json',
    'ChatYuan_RIA':'./output/supervision/predict_chatyuan_ria.json',
    'ChatGLM':'./output/supervision/predict_chatglm.json',
    'ChatGLM_RIA':'./output/supervision/predict_chatglm_ria.json',
    'Qwen':'./output/supervision/predict_Qwen.json',
    'Qwen_RIA':'./output/supervision/predict_Qwen_ria.json',
    'GPT4':'./output/fewshot/gpt-4/result.json',
    }

test_data = []
with open('./output/test_dataset_google.json', 'r', encoding='utf-8') as rf:
    for index, line in enumerate(rf):
        data = eval(line)
        data.update({'index':index})
        test_data.append(data)
test_data = pd.DataFrame(test_data).sample(frac=1, random_state=2024, ignore_index=True)

sampled_test_words_index = []
word_counter = Counter()
for example in test_data.to_dict('records'):
    type_word = example['type']
    if word_counter[type_word] < 20:
        sampled_test_words_index.append(example['index'])
        word_counter[type_word] += 1

assert len(sampled_test_words_index) == 60

predict_data = pd.DataFrame()
for model, data_path in models_predict_path.items():
    temp = []
    with open(data_path, 'r', encoding='utf-8') as rf:
        for index, line in enumerate(rf):
            MD5 = generate_random_md5()
            models_md5_dict[model].append(MD5)
            data = eval(line)
            data.update({'MD5':MD5, 'index':index})
            temp.append(data)
    predict_data = pd.concat([predict_data,pd.DataFrame(temp)[['MD5','index','sentence','source','answer']]], ignore_index=True)
    predict_data['answer'] = [answer.replace('#','') for answer in predict_data['answer']]


annotation_data = pd.DataFrame()
for index in sampled_test_words_index:
    predict_data_select = predict_data[predict_data['index'] == index]
    assert predict_data_select.shape[0] == len(models_md5_dict)
    predict_data_select = predict_data_select.sample(frac=1, random_state=2024, ignore_index=True)
    predict_data_select.loc[list(range(1,predict_data_select.shape[0])),['sentence','source']] = ''
    predict_data_select['Complexity'] = ''
    predict_data_select['Fluency'] = ''
    predict_data_select['Semantics'] = ''
    annotation_data = pd.concat([annotation_data,predict_data_select], ignore_index=True)

annotation_data.to_excel('annotation.xlsx',index=False)

conn = sqlite3.connect('./output/model_MD5.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS record (MD5 TEXT, model TEXT, PRIMARY KEY('MD5'))")
for model, MD5s in models_md5_dict.items():
    for MD5 in MD5s:
        cursor.execute("INSERT OR IGNORE INTO record (MD5, model) VALUES (?, ?)",(MD5, model))
conn.commit()
conn.close()




