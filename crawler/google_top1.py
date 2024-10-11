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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_json_line, load_json_list, load_synonym_dict, test_dataset_path


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                'Accept-Encoding': 'gzip, deflate',
            }

crawler_output_dir = '../output/crawler'

test_dataset = pd.read_csv(test_dataset_path, names=['sentence','source','pos','index','candidates'], sep='\t')
complex_word_test = set(test_dataset['source'])
word_set = set([item['word'] for item in load_json_list()])
idiom_set = set([item['name'] for item in load_json_line()])
syn_set = set([word for word_list in load_synonym_dict() for word in word_list])

ood_words = complex_word_test - word_set - idiom_set - syn_set
words_to_crawl = ood_words
print(len(words_to_crawl))

result = []
async def crawler():
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for word in tqdm(words_to_crawl):
            try:
                url = f'https://www.google.com/search?q=\"{word}\"+\"意思\"&lr=lang_zh-CN'
                print(url)
                async with session.get(url,headers=headers, proxy='http://127.0.0.1:7890') as resp:
                    r = await resp.text()
                    soup = BeautifulSoup(r, "html.parser")
                    items = soup.find(id='search')
                    title = items.find(class_='hgKElc')
                    if title:
                        interpretation = title.text
                    else:
                        title = items.find(class_='VwiC3b')
                        interpretation = title.text
                    result.append({'word':word, 'interpretation':[re.sub(r'^\d{4}年\d{1,2}月\d{1,2}日 — ','',interpretation)]})
                    time.sleep(random.uniform(1,2))
            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(random.uniform(1,2))

asyncio.run(crawler())

with open(os.path.join(crawler_output_dir,'google_result.json'),'w',encoding='utf-8') as wf:
    for data in result:
        wf.write(json.dumps(data, ensure_ascii=False)+'\n')