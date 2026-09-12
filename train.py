from dataclasses import dataclass 
import os 

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

from decoder import get_model 

class DataLoader:
    def __init__(self, path, B, T):
        self.B = B
        self.T = T

        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        print(f"loaded {self.tokens.size} tokens")
        print(f"1 epoch {self.tokens.size // (B * T)} batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.from_numpy(self.tokens[self.current_position : self.current_position + B*T+1])
        x = buf[:-1].view(B, T) #inputs
        y = buf[1:].view(B, T) #targets

        #advance position in tokens
        self.current_position += B * T

        #reset when you reach end of dataset
        if self.current_position + (B * T + 1) > self.tokens.size:
            self.current_position = 0

        return x, y
        
@dataclass
class DecoderConfig:
    seq_len: int = 4096
    vocab_size: int =50304
    n_blocks: int = 36
    n_head: int = 16
    n_embd: int = 768
    base: int = 10000


config = DecoderConfig()
model = get_model(config)
dataloader = DataLoader("final_data/train.bin")

#Training loop 
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

for i in range(50):
    x, y = dataloader.next_batch()
    
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}, Loss: {loss.item()}")



import sys; sys.exit(0)
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


    
