# 读取人工标注结果
import sqlite3

import pandas as pd
import numpy as np

annotation_result = pd.read_excel('annotation.xlsx')
conn = sqlite3.connect('./output/model_MD5.db')
cursor = conn.cursor()

test_data = []
with open('./output/test_dataset_google.json', 'r', encoding='utf-8') as rf:
    for index, line in enumerate(rf):
        data = eval(line)
        data.update({'index':index})
        test_data.append(data)
test_data = pd.DataFrame(test_data)

def MD5_to_model(MD5):
    cursor.execute(f"SELECT model FROM record where MD5 = '{MD5}'")
    return cursor.fetchall()[0][0]

model_score = {'BERT-LS':{}, 'ChatYuan':{}, 'ChatYuan_RIA':{}, 'ChatGLM':{}, 'ChatGLM_RIA':{}, 'Qwen':{}, 'Qwen_RIA':{}, 'GPT4':{}}
model_score = {key:{'Complexity':[],'Fluency':[],'Semantics':[]} for key in model_score}

for example in annotation_result.to_dict('records'):
    model = MD5_to_model(example['MD5'])
    model_score[model]['Complexity'].append(example['Complexity'])
    model_score[model]['Fluency'].append(example['Fluency'])
    model_score[model]['Semantics'].append(example['Semantics'])

for result in model_score.values():
    for value in result.values():
        assert len(value) == 60

model_score_avg = {model:{word_type:round(np.mean(scores),2) for word_type, scores in scores.items()} for model, scores in model_score.items()}
for model, scores in model_score_avg.items():
    model_score_avg[model].update({'all':round(np.sum(list(scores.values())),2)})

for model, score in model_score_avg.items():
    print(model, score)

conn.commit()
conn.close()