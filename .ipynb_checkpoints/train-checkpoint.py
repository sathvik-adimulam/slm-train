from dataclasses import dataclass 
import os 

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 

from model import get_model 


@dataclass
class DecoderConfig:
    seq_len: int = 4096
    vocab_size: int =50304
    n_blocks: int = 36
    n_head: int = 16
    n_embd: int = 768
    base: int = 10000


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


    
