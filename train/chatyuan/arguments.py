# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass
from datetime import datetime

@dataclass
class ModelArgs:
    debug: bool = False

    # 数据路径
    train_file_path: str = '../output/train_dataset.json'
    max_seq_length: int = 512
    max_seq_length_dec: int = 512

    # 模型路径
    pretrained_model_path: str = '*****/ChatYuan-large-v2' 
    save_model_name: str = 'gpt4o'
    save_model_path: str = './output/saved_models/{}/'.format(save_model_name + datetime.now().strftime('_%Y_%m_%d'))

    # 训练参数
    num_epochs: int = 1
    log_steps: int = 100
    custom_training_steps: int = 0

    lr: float = 5e-5
    scheduler: str = 'cosine_warmup'
    warmup_steps: int = 0
    max_grad_norm: float = 1.0

    batch_size: int = 16
    gradient_accumulation_steps: int = 1

    use_cuda: bool = True
    
    def to_dict(self):
        return asdict(self)
