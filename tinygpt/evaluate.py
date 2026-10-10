"""Validation loss / perplexity on a tokenized .bin file."""
from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def eval_ppl(model, val_bin: str, ctx: int, n_windows: int = 100, batch_size: int = 16, device="cpu"):
    """Mean next-token cross-entropy over evenly spaced, non-overlapping windows of `ctx` tokens.
    Returns (loss_in_nats, perplexity). Deterministic, so numbers are comparable across variants."""
    model.eval()
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    n_avail = (len(data) - 1) // ctx
    n = min(n_windows, n_avail)
    starts = np.linspace(0, n_avail - 1, n).astype(int) * ctx
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        chunk = starts[i:i + batch_size]
        x = torch.stack([torch.from_numpy(data[s:s + ctx].astype(np.int64)) for s in chunk]).to(device)
        y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + ctx].astype(np.int64)) for s in chunk]).to(device)
        _, loss = model(x, y)
        total += loss.item() * x.numel()
        count += x.numel()
    mean = total / count
    return mean, math.exp(mean)
