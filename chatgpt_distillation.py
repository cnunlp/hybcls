import json
import os
import time
import multiprocessing
from utils import load_json_line, generate_from_chatgpt, err_call_back

NUM_FOR_EACH_LEVEL = 1  # 每个难度等级生成的替换词数量
DEBUG = False   # 使用少量数据测试
MODEL_NAME = 'gpt-4o'
NUM_WORKERS = 5

output_dir = f"./output/distillation/{MODEL_NAME}"
pivot_words_dir = './output/result_sample_pivot.json'

prompt_template = '\
词汇难度等级划分标准：\n\
初级：高频使用，具有简单明了的含义，易于理解，通常是儿童和初学者最先接触的词汇。\n\
中级：中频使用，为普通人常用的词汇，适用于一般的交流和写作。\n\
高级：低频使用，为更加复杂和精确的词汇，适用于专业领域、文学作品或高级写作。\n\
\n\
Q:请根据上述词汇难度等级划分标准，完成以下任务：\n\
（1）给出目标词#洪亮#的词汇难度等级。\n\
（2）任意生成一个包含目标词的句子\n\
（3）针对目标词，为每个词汇难度等级给出1个不重复的替换词，满足替换后的句子语义一致且流畅通顺。\n\
A:\n\
（1）#洪亮#的词汇难度等级为：中级。\n\
（2）句子：这句呼喊更似#洪亮#的钟声，把无数醉生梦死的人唤醒。\n\
（3）初级：这句呼喊更似#很响#的钟声，把无数醉生梦死的人唤醒。\n\
中级：这句呼喊更似#轰鸣#的钟声，把无数醉生梦死的人唤醒。\n\
高级：这句呼喊更似#震耳欲聋#的钟声，把无数醉生梦死的人唤醒。\n\
\n\
Q:请根据上述词汇难度等级划分标准，完成以下任务：\n\
（1）给出目标成语#灿若星辰#的词汇难度等级。\n\
（2）任意生成一个包含目标成语的句子\n\
（3）针对目标成语，为每个词汇难度等级给出1个不重复的替换词或短语，满足替换后的句子语义一致且流畅通顺。\n\
A:\n\
（1）#灿若星辰#的词汇难度等级为：高级。\n\
（2）句子：我该像初生的朝阳一样笑得#灿若星辰#，如沉暮一样寂静。\n\
（3）初级：我该像初生的朝阳一样笑得#非常开心#，如沉暮一样寂静。\n\
中级：我该像初生的朝阳一样笑得#无比灿烂#，如沉暮一样寂静。\n\
高级：我该像初生的朝阳一样笑得#耀眼动人#，如沉暮一样寂静。\n\
\n\
Q:请根据上述词汇难度等级划分标准，完成以下任务：\n\
（1）给出目标{word_type}#{pivot_word}#的词汇难度等级。\n\
（2）任意生成一个包含目标{word_type}的句子\n\
（3）针对目标{word_type}，为每个词汇难度等级给出{num_for_each_level}个不重复的替换{candidate_type}，满足替换后的句子语义一致且流畅通顺。\n\
A:\n\
'


def pivot_to_prompt(pivot_words_info):
    prompt = []
    for word_info in pivot_words_info:
        word = word_info['word']
        word_type = word_info['type']
        if word_type == 'common':
            question = prompt_template\
                .replace('{word_type}','词')\
                .replace('{pivot_word}',word)\
                .replace('{num_for_each_level}',str(NUM_FOR_EACH_LEVEL))\
                .replace('{candidate_type}','词')
            prompt.append({'word':word, 'type':word_type, 'question':question})
        elif word_type == 'idiom':
            question = prompt_template\
                .replace('{word_type}','成语')\
                .replace('{pivot_word}',word)\
                .replace('{num_for_each_level}',str(NUM_FOR_EACH_LEVEL))\
                .replace('{candidate_type}','词或短语')
            prompt.append({'word':word, 'type':word_type, 'question':question})
        else:
            print('Word type error')

    return prompt


def aggregate_results():
    with open(os.path.join(output_dir,f"result.json"),'w',encoding='utf-8') as wf:
        for i in range(NUM_WORKERS):
            with open(os.path.join(output_dir,f"worker_{i}.json"),'r',encoding='utf-8') as rf:
                for line in rf:
                    wf.write(line)


def func(id, data_part):
    print(f"*** Process {id} started ***")
    total_len = len(data_part)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir,f"worker_{id}.json"),'w',encoding='utf-8') as wf:
        start_time = time.time()
        for i, data in enumerate(data_part):
            average_time = (time.time()-start_time)/(i+1)
            remain_time = average_time*(total_len-i-1)/60
            print(f"Process {id}: {i}/{total_len} | Average: {round(average_time,1)} s | Remain: {round(remain_time,1)} min")
            wf.write(json.dumps({'word':data['word'], 'type':data['type'], 'question':data['question'], 'answer':generate_from_chatgpt(data['question'], MODEL_NAME)}, ensure_ascii=False)+'\n')

    print(f"*** Process {id} finished ***")


if __name__ == "__main__": 
    data = pivot_to_prompt(load_json_line(pivot_words_dir))
    if DEBUG:
        data = data[:10]
    else:
        data = data[:5000]
    num_data_part = len(data)//NUM_WORKERS    

    pool = multiprocessing.Pool(processes = NUM_WORKERS)
    for i in range(NUM_WORKERS):
        data_part = data[num_data_part*i:num_data_part*(i+1)] if i != NUM_WORKERS-1 else data[num_data_part*i:]
        pool.apply_async(func, (i, data_part), error_callback=err_call_back)

    pool.close()
    pool.join()

    aggregate_results()

    print("Mission accomplished.")