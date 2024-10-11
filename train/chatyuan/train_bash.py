import os
import sys
import math
import json
import time
import torch
from torch.nn.utils import clip_grad_norm_

import transformers
from transformers import AdamW
from transformers import(
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
    set_seed
)
from datasets import Dataset

from pprint import pprint
from copy import deepcopy
from mydataloader import MyDataloader
from arguments import ModelArgs
import utils

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info('loading arguments ...')
args = ModelArgs().to_dict()
pprint(args)

seed = sys.argv[1]
set_seed(int(seed))
print('seed ### ' + seed)
logger.info('loading tokenzier & model ...')
tokenizer = T5Tokenizer.from_pretrained(args['pretrained_model_path'])
config = T5Config.from_pretrained(args['pretrained_model_path'])
model = T5ForConditionalGeneration.from_pretrained(args['pretrained_model_path'])

device = utils.get_device(args['use_cuda'])
model.to(device)

logger.info('loading train & eval data ...')
train_data = utils.load_data_train(args['train_file_path'])
if args['debug']:
    train_data = train_data[:500]

logger.info('initializing dataloaders ...')
train_dataloader = MyDataloader(train_data, tokenizer, batch_size=args['batch_size'], shuffle=True, max_seq_length=args['max_seq_length'], max_seq_length_dec=args['max_seq_length_dec'], device=device).dataloader

logger.info('********** Data examples **********')
first_batch = next(iter(deepcopy(train_dataloader)))
logger.info('Inputs :')
logger.info(tokenizer.decode(first_batch['input_ids'][0,:]))
logger.info('Decoder_inputs :')
logger.info(tokenizer.decode(first_batch['decoder_input_ids'][0,:]))
logger.info('Labels :')
logger.info(tokenizer.decode(first_batch['labels'][0,:]))

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'])
num_epochs = args['num_epochs']
num_training_steps = len(train_dataloader) * num_epochs
if args['custom_training_steps']:
    num_training_steps = args['custom_training_steps']
scheduler = utils.get_scheduler(args['scheduler'])(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=num_training_steps)

########################################################################
num_train_examples = len(train_data)
gradient_accumulation_steps = args['gradient_accumulation_steps']
log_steps = args['log_steps']
batch_size = args['batch_size']
total_batchsize = batch_size * gradient_accumulation_steps

args['save_model_path'] = args['save_model_path'] + f'seed{seed}/'

logger.info('********** Training **********')
logger.info(f'training examples : {num_train_examples}')
logger.info(f'total batch size : {total_batchsize} | gradient accumulation : {gradient_accumulation_steps}')
logger.info(f'max opimization steps :{num_training_steps}')

global_steps = 1
train_loss = 0

for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.train()
        loss = model(**batch)['loss']

        if args['gradient_accumulation_steps'] > 1:
            loss = loss / args['gradient_accumulation_steps']
            
        loss.backward()
        train_loss += loss.item()

        if global_steps % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(),max_norm=args['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
               
        if global_steps % log_steps == 0:
            current_lr = scheduler.get_last_lr()[0]
            current_loss = train_loss/log_steps
            train_loss = 0
            logger.info(f'global_step : {global_steps} | loss : {round(current_loss,6)} | lr : {current_lr}')

        global_steps += 1

    utils.save_model(args, model, global_steps)
    logger.info(f'last global_step : {global_steps} | loss : {round(current_loss,6)} | lr : {current_lr}')
    logger.info(f'epoch ended with {global_steps} steps.')

