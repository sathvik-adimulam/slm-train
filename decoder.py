import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, dim, base, seq_len):
        super().__init__()
        # Init variables
        self.dim = dim
        self.base = base
        self.seq_len = seq_len

        # Init RoPE variables
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.seq_len)

    def build_rope_cache(self, seq_len):
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=self.theta.dtype)

        # Outer product of theta and seq idx
        idx_theta = torch.outer(seq_idx, self.theta).float()

        # cache cos and sin of idx theta
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x, *, input_pos=None):
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Init variables
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V projections
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)

        # QK Norm
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # RoPE
        self.rope = RoPE(self.head_dim, config.base, config.seq_len)

        # Output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.proj.SCALE_INIT = True

    def forward(self, x):
        # Q, K, V projections
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # QK Norm
        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE
        q, k = self.rope(q), self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.proj(attn)
        return out


class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)  # value
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=False)  # down-proj

        self.w2.SCALE_INIT = True

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.rn1 = nn.RMSNorm(config.n_embd)
        self.mha = CausalSelfAttention(config)
        self.rn2 = nn.RMSNorm(config.n_embd)

        hidden_dim = int(config.n_embd * 8 / 3)
        self.swiglu = SwiGLU(config.n_embd, hidden_dim)

    def forward(self, x):
        x = x + self.mha(self.rn1(x))
        x = x + self.swiglu(self.rn2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_blocks)]
        )
        self.rms_norm = nn.RMSNorm(config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.rms_norm(x)
        logits = self.linear(x)
        return logits


def get_model(config):
    return Decoder(config)
