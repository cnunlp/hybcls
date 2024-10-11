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
from zhon.hanzi import punctuation, characters
from utils import load_json_line, load_json_list, load_synonym_dict, test_dataset_path


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                'Accept-Encoding': 'gzip, deflate',
                'Cookie': ''} # 百度账号 cookie，格式为：BIDUPSID=******
crawler_output_dir = '../output/crawler/'

test_dataset = pd.read_csv(test_dataset_path, names=['sentence','source','pos','index','candidates'], sep='\t')
complex_word_test = set(test_dataset['source'])
word_set = set([item['word'] for item in load_json_list()])
idiom_set = set([item['name'] for item in load_json_line()])
syn_set = set([word for word_list in load_synonym_dict() for word in word_list])

ood_words = complex_word_test - word_set - idiom_set - syn_set
words_to_crawl = ood_words

result = []
async def crawler():
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for word in tqdm(words_to_crawl):         
            print(word)   
            try:
                url = f'https://www.baidu.com/s?wd={word}的网络意思&nojc=1&rtn=baidu'
                print(url)
                async with session.get(url,headers = headers) as resp:
                    r = await resp.text()
                    soup = BeautifulSoup(r, "html.parser")
                    items = soup.find(id='content_left')
                    titles = items.find_all(id='1')
                    result.append({'word':word, 'interpretation':''.join(re.findall(f"[0-9a-zA-Z{punctuation}{characters}]",titles[0].text))})
                    time.sleep(random.randint(5,10))
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(random.randint(5,10))

asyncio.run(crawler())

with open(os.path.join(crawler_output_dir,'baidu_result.json'), 'w',encoding='utf-8') as wf:
    for data in result:
        wf.write(json.dumps(data, ensure_ascii=False)+'\n')