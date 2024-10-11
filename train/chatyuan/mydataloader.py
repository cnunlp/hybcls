import os
import sys
import json
import torch

from multiprocessing import Pool
from torch.utils.data import DataLoader

from utils import preprocess_chatyuan


class MyDataloader():
    def __init__(self, data, tokenizer, batch_size=1, shuffle=False, max_seq_length=512, max_seq_length_dec=512, device='cpu'):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec
        self.device = device
        self.collate_fn = self.collate_fn_train
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        

    def collate_fn_train(self, examples):

        questions = [preprocess_chatyuan(data['question']) for data in examples]
        answers = [preprocess_chatyuan(data['answer']) for data in examples]

        inputs = self.tokenizer(text=questions, truncation=True, padding=True, max_length=self.max_seq_length, return_tensors='pt').to(self.device)
        outputs = self.tokenizer(text=answers, truncation=True, padding=True, max_length=self.max_seq_length_dec+1, return_tensors='pt').to(self.device)

        labels = outputs['input_ids'][:,:self.max_seq_length_dec]
        decoder_input_ids = labels.new_zeros(labels.shape)
        decoder_input_ids[:,1:] = labels[:,:-1].clone()
        decoder_input_ids[:,0] = 0

        batch_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }

        return batch_inputs

