import json
import os
import time
import multiprocessing
import re
import pandas as pd
from copy import deepcopy
from utils import load_json_line, generate_from_chatgpt, err_call_back, evalute, evalute_span
from zhon.hanzi import characters


DEBUG = False   # 使用少量数据测试
CRAWL = False   # 是否执行爬取
EVAL = True     # 是否执行结果测试
SPAN = True     # 是否额外使用span级别评估（fuzzy）
RIA = True      # 是否使用RIA

MODEL_NAME = 'gpt-4'
NUM_WORKERS = 2

output_dir = f"./output/fewshot/{MODEL_NAME}"
test_dataset_path = './output/test_dataset_google.json'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

candidate_type_dict = {'common':'词', 'idiom':'词或短语', 'ood':'词或短语'}

system_prompt = '请你根据要求执行词汇简化任务，以下是任务示例：\n{few_shot_prompt}请你完成：\n{question}'
prompt_template = '句子：{sentence}\n要求：针对句子中的#{source}#，请你给出一个能在句子中将其流畅替换、含义相同且更简单的{candidate_type}。\n回答：{answer}'
prompt_ria_template = '句子：{sentence}\n#{source}#的解释：“{interpretation}”\n要求：针对句子中的#{source}#，请你参考其解释，给出一个能在句子中将其流畅替换且含义相同的{candidate_type}。\n回答：{answer}'

few_shot_examples = [
    {'sentence':'她下班时我们为她端上一杯浓浓的咖啡，说句#温馨#的话语。', 'source':'温馨' ,'candidate_type':'词', 'interpretation':'温暖芳香春夜温馨｜温馨的花园｜午后温馨薄暮凉；温暖。' ,'answer':'温暖'},
    {'sentence':'听了奶奶的一番话，我#恍然大悟#，一头扑在奶奶的怀里。', 'source':'恍然大悟' ,'candidate_type':'词或短语', 'interpretation':'恍然：猛然清醒的样子；悟：心里明白。形容一下子明白过来。' ,'answer':'突然明白'},
    # {'sentence':'他经常#吐槽#我们的老板，让大家都觉得很好笑。', 'source':'吐槽' ,'candidate_type':'词或短语', 'interpretation':'一般是指从对方的语言或行为中找到一个漏洞或关键词作为切入点，发出带有调侃意味的感慨或疑问。' ,'answer':'抱怨'},
    ]

few_shot_prompt = ''
for example in few_shot_examples:
    prompt_example = prompt_ria_template.replace('{interpretation}',example['interpretation']) if RIA else prompt_template
    prompt_example = prompt_example\
        .replace('{sentence}',example['sentence'])\
        .replace('{source}',example['source'])\
        .replace('{candidate_type}',example['candidate_type'])\
        .replace('{answer}',f"#{example['answer']}#")
    few_shot_prompt += f'{prompt_example}\n\n'
        
    
def word_to_prompt(words_data):
    prompt = []
    for word_info in words_data:
        sentence = word_info['sentence']
        source = word_info['source']
        word_type = word_info['type']
        if word_type!='ood':
            continue
        interpretation = word_info['interpretation']
        question = prompt_ria_template.replace('{interpretation}',interpretation) if RIA else prompt_template
        question = question\
            .replace('{sentence}',sentence)\
            .replace('{source}',source)\
            .replace('{candidate_type}',candidate_type_dict[word_type])\
            .replace('{answer}','')
        question = system_prompt\
            .replace('{few_shot_prompt}',few_shot_prompt)\
            .replace('{question}',question)                
        word_info_copy = deepcopy(word_info)
        word_info_copy.update({'question':question})
        prompt.append(word_info_copy)

    return prompt


def aggregate_results():
    with open(os.path.join(output_dir,"result_ria.json" if RIA else "result.json"),'w',encoding='utf-8') as wf:
        for i in range(NUM_WORKERS):
            with open(os.path.join(output_dir,f"worker_{i}_ria.json" if RIA else f"worker_{i}.json"),'r',encoding='utf-8') as rf:
                for line in rf:
                    wf.write(line)


def func(id, data_part):
    print(f"*** Process {id} started ***")
    total_len = len(data_part)
    file_name = f"worker_{id}_ria.json" if RIA else f"worker_{id}.json"
    with open(os.path.join(output_dir, file_name),'w',encoding='utf-8') as wf:
        start_time = time.time()
        for i, data in enumerate(data_part):
            if i == 0:
                print('='*50)
                print(f"RIA: {str(RIA)} | question example: {data['question']}")
                print('='*50)
            average_time = (time.time()-start_time)/(i+1)
            remain_time = average_time*(total_len-i-1)/60
            print(f"Process {id}: {i}/{total_len} | Average: {round(average_time,1)} s | Remain: {round(remain_time,1)} min")
            data.update({'answer':generate_from_chatgpt(data['question'], MODEL_NAME)})
            wf.write(json.dumps(data, ensure_ascii=False)+'\n')

    print(f"*** Process {id} finished ***")


if __name__ == "__main__": 
    if CRAWL:
        test_data = load_json_line(test_dataset_path)
        data = word_to_prompt(test_data)
        if DEBUG:
            data = data[:10]

        num_data_part = len(data)//NUM_WORKERS    

        pool = multiprocessing.Pool(processes = NUM_WORKERS)
        for i in range(NUM_WORKERS):
            data_part = data[num_data_part*i:num_data_part*(i+1)] if i != NUM_WORKERS-1 else data[num_data_part*i:]
            pool.apply_async(func, (i, data_part), error_callback=err_call_back)

        pool.close()
        pool.join()

        aggregate_results()

        print("Crawl accomplished.")

    if EVAL:
        result_dir = os.path.join(output_dir,"result_ria.json" if RIA else "result.json")
        result_data = pd.DataFrame(load_json_line(result_dir))

        answer_extracted = []
        for source, answer in zip(result_data['source'], result_data['answer']):
            pattern = f'#([{characters}]+)#'
            re_result = re.findall(pattern,answer)
            re_result = [word for word in re_result if word != source]
            answer_extracted.append(re_result[0] if len(set(re_result)) == 1 else 'Answer not found')

        print('*'*20,' accurate result ','*'*20)
        evalute(result_data['source'], answer_extracted, result_data['gold'], result_data['type'], print_result=True, print_error=True)
        if SPAN:
            print('*'*20,' span result ','*'*20)
            evalute_span(result_data['source'], answer_extracted, result_data['gold'], result_data['type'], print_result=True, print_error=True)