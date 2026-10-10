#!/usr/bin/env python
"""fp32 vs hand-written int8 vs torch dynamic int8: size, perplexity, speed (CPU).

  python scripts/bench_quant.py --ckpt runs/base_sft/best.pt --val data/tinystories/val.bin

Note: perplexity is measured on plain TinyStories validation text (pretraining distribution).
"""
import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from tinygpt.evaluate import eval_ppl
from tinygpt.generate import generate_stream
from tinygpt.quant import model_size_mb, quantize_model
from tinygpt.utils import load_model


def file_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return mb


def decode_speed(model, new_tokens, prompt_len, repeats):
    x = torch.randint(0, model.cfg.vocab_size, (1, prompt_len))
    list(generate_stream(model, x, 8, temperature=1.0, top_k=0))  # warmup
    best_tps, best_prefill = 0.0, 1e9
    for _ in range(repeats):
        t0, ttft, n = time.perf_counter(), None, 0
        for _ in generate_stream(model, x, new_tokens, temperature=1.0, top_k=0):
            n += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
        total = time.perf_counter() - t0
        best_tps = max(best_tps, (n - 1) / (total - ttft))
        best_prefill = min(best_prefill, ttft)
    return best_tps, best_prefill * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default="data/tinystories/val.bin")
    ap.add_argument("--n_windows", type=int, default=100)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--prompt_len", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results/bench_quant")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    fp32, cfg = load_model(a.ckpt, "cpu")
    variants = {"fp32": fp32}
    variants["int8 (hand-written, weight-only)"] = quantize_model(copy.deepcopy(fp32))
    try:
        if torch.backends.mps.is_available() or os.uname().machine == "arm64":
            torch.backends.quantized.engine = "qnnpack"  # x86 'fbgemm' engine is not available on Apple Silicon
        variants["int8 (torch dynamic quant)"] = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8)
    except Exception as e:  # engine not available on some platforms/versions
        print("torch dynamic quantization unavailable:", e)

    rows = []
    for name, m in variants.items():
        m.eval()
        try:
            loss, ppl = eval_ppl(m, a.val, cfg.max_seq_len, a.n_windows, device="cpu")
            tps, prefill_ms = decode_speed(m, a.new_tokens, a.prompt_len, a.repeats)
        except Exception as e:
            print(f"skipping {name}: {e}")
            continue
        # torch's packed dynamic-quant weights are not visible as parameters/buffers, so in-memory size is n/a
        mem = "n/a" if "dynamic" in name else round(model_size_mb(m), 1)
        row = dict(variant=name, mem_mb=mem, file_mb=round(file_mb(m), 1),
                   val_loss=round(loss, 4), ppl=round(ppl, 3), decode_tok_s=round(tps, 1),
                   prefill_ms=round(prefill_ms, 1))
        rows.append(row)
        print(row, flush=True)

    md = ["| variant | memory (MB) | file (MB) | val loss | perplexity | decode tok/s (B=1) | prefill (ms) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['mem_mb']} | {r['file_mb']} | {r['val_loss']} | {r['ppl']} | "
                  f"{r['decode_tok_s']} | {r['prefill_ms']} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    Path(a.out + ".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
