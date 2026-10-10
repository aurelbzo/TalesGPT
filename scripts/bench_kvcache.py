#!/usr/bin/env python
"""Benchmark generation with and without the KV cache (and across batch sizes / devices).

  python scripts/bench_kvcache.py --ckpt runs/base_sft/best.pt --devices cpu,mps --new_tokens 128

Per config we report: time-to-first-token (prefill) and steady-state decode tokens/s.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tinygpt.generate import generate_stream
from tinygpt.model import KVCache
from tinygpt.utils import load_model, sync


def run_once(model, device, B, prompt_len, new_tokens, use_cache):
    x = torch.randint(0, model.cfg.vocab_size, (B, prompt_len), device=device)
    sync(device)
    t0 = time.perf_counter()
    ttft, n = None, 0
    # temperature=1 + top_k=0 keeps sampling cheap; no eos_id -> fixed number of tokens
    for _ in generate_stream(model, x, new_tokens, temperature=1.0, top_k=0, use_cache=use_cache):
        n += 1
        if ttft is None:
            sync(device)
            ttft = time.perf_counter() - t0
    sync(device)
    total = time.perf_counter() - t0
    decode_s = max(total - ttft, 1e-9)
    return ttft, (n - 1) * B / decode_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--devices", default="cpu")
    ap.add_argument("--prompt_lens", default="16,128")
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--batch_sizes", default="1,4,16", help="batch sizes for the cached path")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--out", default="results/bench_kvcache")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    rows = []
    for dev_name in a.devices.split(","):
        device = torch.device(dev_name)
        model, cfg = load_model(a.ckpt, device)
        for pl in map(int, a.prompt_lens.split(",")):
            new = min(a.new_tokens, cfg.max_seq_len - pl)
            configs = [(False, 1)] + [(True, b) for b in map(int, a.batch_sizes.split(","))]
            for use_cache, B in configs:
                run_once(model, device, B, pl, min(8, new), use_cache)  # warmup
                res = [run_once(model, device, B, pl, new, use_cache) for _ in range(a.repeats)]
                ttft = sorted(r[0] for r in res)[len(res) // 2] * 1e3
                tps = sorted(r[1] for r in res)[len(res) // 2]
                kv_mb = KVCache(cfg, B, device="cpu").nbytes / 1e6 if use_cache else 0.0
                row = dict(device=dev_name, prompt_len=pl, new_tokens=new, kv_cache=use_cache, batch=B,
                           ttft_ms=round(ttft, 1), decode_tok_s=round(tps, 1), kv_cache_mb=round(kv_mb, 1))
                rows.append(row)
                print(row, flush=True)

    base = {(r["device"], r["prompt_len"]): r["decode_tok_s"] for r in rows if not r["kv_cache"]}
    md = ["| device | prompt | new | KV cache | batch | TTFT (ms) | decode tok/s (total over batch) | speedup vs naive | KV mem (MB) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = r["decode_tok_s"] / base[(r["device"], r["prompt_len"])] if r["kv_cache"] and r["batch"] == 1 else None
        md.append(f"| {r['device']} | {r['prompt_len']} | {r['new_tokens']} | {'on' if r['kv_cache'] else 'off'} | "
                  f"{r['batch']} | {r['ttft_ms']} | {r['decode_tok_s']} | {f'{sp:.1f}x' if sp else '-'} | {r['kv_cache_mb']} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    Path(a.out + ".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()