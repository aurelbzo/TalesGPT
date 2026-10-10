"""Hand-written weight-only int8 quantization with per-output-channel scales.

w_q = round(w / s) clipped to [-127, 127], with s_c = max|w_c| / 127 for each output channel c.
At runtime:  y = (x @ w_q^T) * s   (the scale commutes with the matmul because it is per output channel).

What this buys: ~4x smaller linear weights (memory + disk).
What it does NOT automatically buy: speed. We still convert int8 -> float before the matmul, so on CPU
this is often *slower* than fp32. Real speedups need int8 matmul kernels (see torch dynamic quantization,
which bench_quant.py compares against).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor_per_channel(w: torch.Tensor):
    """w: [out, in] float -> (int8 [out, in], float32 scale [out])."""
    w = w.detach().float()
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features))

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "QuantLinear":
        assert lin.bias is None, "bias-free linears only"
        m = cls(lin.in_features, lin.out_features)
        q, s = quantize_tensor_per_channel(lin.weight)
        m.qweight.copy_(q)
        m.scale.copy_(s)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.qweight.to(x.dtype))
        return y * self.scale.to(x.dtype)


def _swap_linears(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear.from_linear(child))
            n += 1
        else:
            n += _swap_linears(child)
    return n


def quantize_model(model: nn.Module) -> nn.Module:
    """In-place: replace every Linear inside the transformer blocks with QuantLinear.
    The token embedding / tied output head stays in fp32 (it is shared, and embeddings are sensitive)."""
    _swap_linears(model.blocks)
    return model


def model_size_mb(model: nn.Module) -> float:
    """In-memory size of parameters + buffers (shared/tied storage counted once)."""
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = t.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total / 1e6


def count_params(model: nn.Module) -> int:
    """Logical parameter count, including int8 weights held as buffers by QuantLinear."""
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for name, b in model.named_buffers() if name.endswith("qweight"))
    return n
