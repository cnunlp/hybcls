from tqdm.auto import tqdm
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import time
import json
import os
import re
import random
import pandas as pd
from zhon.hanzi import characters
from utils import load_json_line, load_json_list, load_synonym_dict, test_dataset_path


CONTINUE_CRAWL = True # 有时会爬取失败，自动检测未派取成功的词，可多次重复运行本脚本。
TOP_K = 10 # 每个词爬取前多少个解释
MAX_PAGE = 3 # 最多翻多少页

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                'Accept-Encoding': 'gzip, deflate',
                'Cookie': ''} # 百度账号 cookie，格式为：BIDUPSID=******

crawler_output_dir = '../output/crawler/zhidao'

test_dataset = pd.read_csv(test_dataset_path, names=['sentence','source','pos','index','candidates'], sep='\t')
complex_word_test = set(test_dataset['source'])
word_set = set([item['word'] for item in load_json_list()])
idiom_set = set([item['name'] for item in load_json_line()])
syn_set = set([word for word_list in load_synonym_dict() for word in word_list])

ood_words = complex_word_test - word_set - idiom_set - syn_set

if CONTINUE_CRAWL:
    words_crawled = set()
    words_path = os.listdir(crawler_output_dir)
    for word_path in words_path:
        words_crawled.add(re.findall(f'([0-9a-zA-Z{characters}]+)\.json',word_path)[0])
    words_to_crawl = ood_words - words_crawled
else:
    words_to_crawl = ood_words


async def crawler():
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for word in tqdm(words_to_crawl):
            topk_flag = False
            result = []
            try:
                for pn in range(0,10*MAX_PAGE,10):
                    url = f'https://zhidao.baidu.com/search?word={word}的网络意思&pn={pn}&nojc=1'
                    async with session.get(url,headers = headers) as resp:
                        r = await resp.text()
                        soup = BeautifulSoup(r, "html.parser")
                        items = soup.find(id='wgt-list')
                        titles = items.find_all(class_='dl')
                        
                        for title in titles:
                            content = title.find('dd').text
                            if word in content and len(re.findall(r'答：',content)):
                                result.append(re.findall(r'答：(.*)',content)[0])
                            if len(result) >= TOP_K:
                                topk_flag = True
                                break
                    if topk_flag:
                        time.sleep(random.randint(5,10))
                        break
                    time.sleep(random.randint(5,10))
                time.sleep(random.randint(5,10))

                if len(result):
                    print(f'Word {word}: found {len(result)} results.')
                else:
                    print(f'Warning: result for word {word} is emtpy, please check {url} for verification.')
                with open(os.path.join(crawler_output_dir,f"{word}.json"),'w',encoding='utf-8') as wf:
                    wf.write(json.dumps(result,ensure_ascii=False)+'\n')

            except Exception as e:
                print(f"Exception: {e}")
                print(f'Notice: The word "{word}" is invalid or the request frequency limit has been exceeded. Please check {url} .')
                time.sleep(random.randint(5,10))

asyncio.run(crawler())


with open(os.path.join(os.path.dirname(crawler_output_dir),'zhidao_result.json'), 'w',encoding='utf-8') as wf:
    for word in ood_words:
        word_file = f"{word}.json"
        if word_file in os.listdir(crawler_output_dir):
            with open(os.path.join(crawler_output_dir, word_file),'r',encoding='utf-8') as rf:
                wf.write(json.dumps({'word':word, 'interpretation':json.load(rf)},ensure_ascii=False)+'\n')
        else:
            wf.write(json.dumps({'word':word, 'interpretation':['']},ensure_ascii=False)+'\n')



