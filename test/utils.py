import torch
import random
import re
import os
import pandas as pd
from tqdm.auto import tqdm

from transformers import(
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)


def get_device(use_cuda: bool=True):

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")


def preprocess_chatyuan(text):

    text = list(filter(lambda x: len(x) > 0, text.splitlines()))
    text = '\n'.join(text)
    text = text.replace("\n", "\\n").replace("\t", "\\t")

    return text


def postprocess_chatyuan(text):

    text = text.replace("\\n", "\n").replace("\\t", "\t")

    return text


def load_data_train(train_file_path):

    data_train = []
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data_train.append(eval(line))

    return data_train


def load_test_data(test_file_path):

    result = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            result.append(eval(line))

    return result


def get_scheduler(name):

    if name == 'constant_warmup':
        return get_constant_schedule_with_warmup
    
    elif name == 'linear_warmup':
        return get_linear_schedule_with_warmup
    
    elif name == 'cosine_warmup':
        return get_cosine_schedule_with_warmup
    
    else:
        print('scheduler of unsupport type')
        exit()


def save_model(args, model, saved_steps):

    save_model_path = os.path.join(args['save_model_path'], f'checkpoint_{saved_steps}.pt')

    if not os.path.exists(args['save_model_path']):
        os.makedirs(args['save_model_path'])

    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    checkpoint = {
        'model': model_state,
        'args': args,
        'saved_steps': saved_steps
        }
    torch.save(checkpoint, save_model_path)
    print('Model Saved.')


def load_synonym_dict(syn_path):
    syndict = {}
    with open(syn_path,'r',encoding='utf-8') as f:
        for line in f:
            data = eval(line)
            syndict[data['word']] = data['synonyms']
            
    return syndict
