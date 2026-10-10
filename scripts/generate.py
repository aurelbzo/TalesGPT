#!/usr/bin/env python
"""Generate text from a checkpoint or exported artifact."""
import argparse
import sys
import time

import torch

from tinygpt.generate import generate_stream
from tinygpt.utils import load_model, load_tokenizer, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    device = pick_device(a.device)
    model, cfg = load_model(a.ckpt, device)
    tok = load_tokenizer(a.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    if a.seed is not None:
        torch.manual_seed(a.seed)
    ids = tok.encode(a.prompt).ids or [eos]
    x = torch.tensor([ids], device=device)
    out, t0 = [], time.perf_counter()
    sys.stdout.write(a.prompt)
    for nxt in generate_stream(model, x, a.max_new_tokens, a.temperature, a.top_k, a.top_p,
                               eos_id=eos, use_cache=not a.no_cache):
        t = nxt.item()
        if t == eos:
            break
        out.append(t)
        prev = tok.decode(out[:-1])  # incremental decode: print only the new text
        sys.stdout.write(tok.decode(out)[len(prev):])
        sys.stdout.flush()
    dt = time.perf_counter() - t0
    print(f"\n\n[{len(out)} tokens in {dt:.2f}s = {len(out)/dt:.1f} tok/s, "
          f"cache={'off' if a.no_cache else 'on'}, device={device}]")


if __name__ == "__main__":
    main()
