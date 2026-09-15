import inspect
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from decoder import get_model

warnings.filterwarnings("ignore")


class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, base_dir, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.shards = [p for p in (base_dir / split).iterdir()]
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _load_tokens(self, file):
        arr = np.memmap(file, dtype=np.uint16, mode="r")
        return torch.from_numpy(arr).long()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets

        # advance position in tokens
        self.current_position += B * T * self.num_processes

        # reset when you reach end of dataset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = B * T * self.process_rank
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])

        return x, y


@dataclass
class DecoderConfig:
    seq_len: int = 1024
    vocab_size: int = 50304
    n_blocks: int = 36
    n_head: int = 16
    n_embd: int = 768
    base: int = 10000


import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA needed for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Gradient accumulation
total_batch_size = 524288  # 2**19
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "make sure total batch size is divisible bu B*T"
)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

if "RUNPOD_POD_ID" in os.environ:
    base_dir = Path("/workspace/shards")
    checkpoint_dir = Path("/workspace/checkpoints")
else:
    base_dir = Path("shards")
    checkpoint_dir = Path("checkpoints")

base_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Load train and val data
train_loader = DataLoader(B, T, ddp_rank, ddp_world_size, base_dir, "train")
val_loader = DataLoader(B, T, ddp_rank, ddp_world_size, base_dir, "val")

# Load model on right device
device = "cuda" if torch.cuda.is_available() else "cpu"
config = DecoderConfig()
model = get_model(config).to(device)
model = torch.compile(model, dynamic=True)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


def get_total_tokens(base_dir, split):
    # Get train path
    train_path = base_dir / split
    # Find all .bin files in the directory
    bin_files = sorted(list(train_path.glob("*.bin")))

    if not bin_files:
        print(f"No .bin files found in {train_path.absolute()}")
        return 0

    # Calculate number of tokens
    total_tokens = 0
    for file in bin_files:
        arr = np.memmap(file, dtype=np.uint16, mode="r")
        num_tokens = arr.size
        total_tokens += num_tokens

    if master_process:
        print(f"total tokens in {split}: {total_tokens:,}")
    return total_tokens


# Calculate optimal learning rate schedule and num steps
total_train_tokens = get_total_tokens(base_dir, "train")
total_val_tokens = get_total_tokens(base_dir, "val")
max_lr = 3e-4
min_lr = max_lr * 0.1
max_steps = total_train_tokens // total_batch_size + 1
warmup_steps = int(0.02 * max_steps)


def get_lr(it):
    # 1. linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2. return min_lr after cosine decay
    if it > max_steps:
        return min_lr

    # 3. cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, weight_decay, learning_rate, device):
    # create optim groups
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(
            f"num decayed parameters tensors: {len(decay_params)}, with {num_decay_params:,} params"
        )
        print(
            f"num non-decayed parameters tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params"
        )

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0},
    ]
    # Create fusex AdamW optimizer
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )
    return optimizer


# Load optimizer
optimizer = configure_optimizer(
    model=model, weight_decay=0.1, learning_rate=3e-4, device=device
)


def save_checkpoint(model, optimizer, step, loss_accum, val_loss_accum, dir):
    raw_model = model.module if ddp else model
    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": loss_accum,
        "val_loss": val_loss_accum,
    }

    tmp_path = dir / f"{step}_checkpoint.pt.tmp"
    path = dir / f"{step}_checkpoint.pt"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


for step in range(max_steps):
    # Train loop
    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Autocast to bf16 and compute logits and class
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss /= grad_accum_steps  # Mean of losses
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()  # Backprop
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000
    if master_process:
        print(
            f"Step {step + 1}/{max_steps}, Loss: {loss_accum.item():.6f}, LR: {lr:.4e}, Norm: {norm:.4f}, Time: {dt:.2f}ms"
        )

    # Val loop
    last_step = step == max_steps - 1
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = max(1, total_val_tokens // total_batch_size + 1)
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss /= val_loss_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation_loss: {val_loss_accum.item():.4f}")

                # Save checkpoint
                if step % 5000 == 0 or last_step:
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        loss_accum,
                        val_loss_accum,
                        checkpoint_dir,
                    )
        model.train()

if ddp:
    dist.barrier()
    destroy_process_group()
