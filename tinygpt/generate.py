"""Autoregressive generation, with and without a KV cache."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT, KVCache


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: [B, V] -> next token ids [B, 1]. temperature<=0 means greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # drop tokens once cumulative probability (excluding themselves) exceeds top_p
        drop = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    if generator is not None and generator.device != probs.device:
        return torch.multinomial(probs.cpu(), 1, generator=generator).to(probs.device)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def generate_stream(model: GPT, prompt_ids: torch.Tensor, max_new_tokens: int,
                    temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
                    eos_id: int | None = None, use_cache: bool = True,
                    generator: torch.Generator | None = None):
    """Yield one tensor [B, 1] of new token ids per step.

    use_cache=True : prefill the prompt once, then feed ONE token per step (O(n) work per token).
    use_cache=False: re-run the whole sequence every step (O(n^2) work per token) - the naive baseline.
    Stops early on `eos_id` (only checked when B == 1).
    """
    model.eval()
    cfg = model.cfg
    B, T = prompt_ids.shape
    max_new = max(0, min(max_new_tokens, cfg.max_seq_len - T))
    idx = prompt_ids

    if use_cache:
        p = next(model.parameters())
        dtype = p.dtype if p.is_floating_point() else torch.float32
        cache = KVCache(cfg, B, device=prompt_ids.device, dtype=dtype)
        logits, _ = model(idx, cache=cache)
    for i in range(max_new):
        if not use_cache:
            logits, _ = model(idx)
        nxt = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
        yield nxt
        if eos_id is not None and B == 1 and nxt.item() == eos_id:
            return
        if use_cache:
            if i < max_new - 1:
                logits, _ = model(nxt, cache=cache)
        else:
            idx = torch.cat([idx, nxt], dim=1)


@torch.no_grad()
def generate(model: GPT, prompt_ids: torch.Tensor, max_new_tokens: int, **kw) -> torch.Tensor:
    """Returns [B, T + new] token ids."""
    out = [prompt_ids]
    for nxt in generate_stream(model, prompt_ids, max_new_tokens, **kw):
        out.append(nxt)
    return torch.cat(out, dim=1)


@torch.no_grad()
def complete(model: GPT, tok, prompt: str, max_new_tokens: int = 200, device="cpu", **kw) -> str:
    """Convenience: text in, text out (B = 1). Stops at <|endoftext|>."""
    eos_id = tok.token_to_id("<|endoftext|>")
    ids = tok.encode(prompt).ids or [eos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    new = []
    for nxt in generate_stream(model, x, max_new_tokens, eos_id=eos_id, **kw):
        t = nxt.item()
        if t == eos_id:
            break
        new.append(t)
    return prompt + tok.decode(new)