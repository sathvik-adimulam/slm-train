from dataclasses import dataclass

import torch

from decoder import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DecoderConfig:
    seq_len: int = 1024
    vocab_size: int = 50304
    n_blocks: int = 36
    n_head: int = 16
    n_embd: int = 768
    base: int = 10000


model = get_model(DecoderConfig()).to(device)
checkpoint = torch.load("checkpoints/23419_checkpoint.pt", map_location=device)
model_state_dict = {
    (k[10:] if k.startswith("_orig_mod.") else k): v
    for k, v in checkpoint["model_state_dict"].items()
}
model.load_state_dict(model_state_dict)
