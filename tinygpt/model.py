from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    max_seq_len: int = 512
    hidden_dim: int = 0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
        assert (self.d_model // self.n_head) % 2 == 0
        if self.hidden_dim == 0:
            h = int(8 * self.d_model / 3)
            self.hidden_dim = 64 * ((h+63) // 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> int:
        return asdict(self)

PRESETS = {
    "debug": dict(n_layer=2, n_head=4, d_model=128, max_seq_len=128),
    "small": dict(n_layer=8, n_head=6, d_model=384, max_seq_len=512),
    "base": dict(n_layer=8, n_head=8, d_model=512, max_seq_len=512),
}

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight 

def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.type_as(x)

class KVCache:
    def __init__(self, cfg: GPTConfig, batch_size: int, max_len: int | None = None,
                 device="cpu", dtype=torch.float32):
        self.max_len = max_len or cfg.max_seq_len
        shape = (cfg.n_layer, batch_size, cfg.n_head, self.max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        T = k_new.shape[2]
        assert self.pos + T <= self.max_len, "KV cache overflow"
        self.k[layer, :, :, self.pos:self.pos + T] = k_new
        self.v[layer, :, :, self.pos:self.pos + T] = v_new
        return self.k[layer, :, :, :self.pos + T], self.v[layer, :, :, :self.pos + T]

    def advance(self, T: int):
        self.pos += T

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        pos = cache.pos if cache is not None else 0
        q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
        k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
            k, v = k.to(q.dtype), v.to(q.dtype)

        S = k.shape[2]  # total keys = pos + T
        if T == 1:  # single-token decode: attends to everything in the cache
            y = F.scaled_dot_product_attention(q, k, v)
        elif S == T:  # full sequence / prefill from empty cache: plain causal mask
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # multi-token chunk after existing context
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.hidden_dim, bias=False)  # fused gate+up
        self.down = nn.Linear(cfg.hidden_dim, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale residual-branch output projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None, cache: KVCache | None = None,
                last_only: bool = True, ignore_index: int = -100):
        """idx: [B, T] token ids.
        - with targets: returns (logits [B,T,V], loss)
        - without: returns (logits, None); logits only for the last position unless last_only=False
        """
        B, T = idx.shape
        pos = cache.pos if cache is not None else 0
        assert pos + T <= self.cfg.max_seq_len, f"sequence too long ({pos + T} > {self.cfg.max_seq_len})"

        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin, cache)
        if cache is not None:
            cache.advance(T)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=ignore_index
            )
            return logits, loss
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x), None