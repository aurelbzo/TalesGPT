#!/usr/bin/env python
"""Supervised fine-tuning: teach the pretrained model the constrained-story prompt format.
The loss is computed ONLY on the response tokens (prompt tokens are masked with -100).

  python scripts/sft.py --ckpt runs/base/best.pt --out_dir runs/base_sft
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from tinygpt.model import GPT, GPTConfig
from tinygpt.utils import autocast_ctx, load_tokenizer, pick_device, pick_dtype


def load_examples(path, tok, eot, max_len):
    ex, skipped = [], 0
    for line in open(path):
        r = json.loads(line)
        p = tok.encode(r["prompt"]).ids
        s = tok.encode(r["response"]).ids + [eot]
        if len(p) + len(s) > max_len:
            skipped += 1
            continue
        ex.append((p + s, [-100] * len(p) + s))
    print(f"{path}: {len(ex)} examples ({skipped} skipped: longer than {max_len} tokens)")
    return ex


def make_batch(examples, pad_id, device):
    L = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), L - 1), pad_id, dtype=torch.long)
    y = torch.full((len(examples), L - 1), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        n = len(ids)
        x[i, :n - 1] = torch.tensor(ids[:-1])
        y[i, :n - 1] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


@torch.no_grad()
def val_loss(model, examples, a, pad_id, device, amp_dtype):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), a.batch_size):
        x, y = make_batch(examples[i:i + a.batch_size], pad_id, device)
        with autocast_ctx(device, amp_dtype):
            _, loss = model(x, y)
        k = (y != -100).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--out_dir", default="runs/base_sft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    amp_dtype, dtype_name = pick_dtype(device, a.dtype)
    tok = load_tokenizer(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    model.to(device).train()

    train_ex = load_examples(a.train, tok, eot, cfg.max_seq_len)
    val_ex = load_examples(a.val, tok, eot, cfg.max_seq_len)
    total = math.ceil(len(train_ex) / a.batch_size) * a.epochs
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    kw = {"fused": True} if device.type == "cuda" else {}
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), **kw)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_name == "fp16" and device.type == "cuda"))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"SFT: {len(train_ex)} train / {len(val_ex)} val examples, {total} steps, device={device} dtype={dtype_name}")
    v0 = val_loss(model, val_ex, a, eot, device, amp_dtype)
    print(f"val loss before SFT: {v0:.4f}")

    step, t0 = 0, time.time()
    for _ in range(a.epochs):
        random.shuffle(train_ex)
        for i in range(0, len(train_ex), a.batch_size):
            if step < a.warmup:
                lr = a.lr * (step + 1) / a.warmup
            else:
                r = (step - a.warmup) / max(1, total - a.warmup)
                lr = 0.1 * a.lr + 0.5 * 0.9 * a.lr * (1 + math.cos(math.pi * r))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = make_batch(train_ex[i:i + a.batch_size], eot, device)
            with autocast_ctx(device, amp_dtype):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} lr {lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
            if step % a.eval_every == 0 or step == total:
                print(f"  >> val loss {val_loss(model, val_ex, a, eot, device, amp_dtype):.4f}", flush=True)
    vf = val_loss(model, val_ex, a, eot, device, amp_dtype)
    print(f"val loss before {v0:.4f} -> after {vf:.4f}")
    torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), step=step, val_loss=vf), out / "best.pt")
    print(f"saved {out/'best.pt'}")


if __name__ == "__main__":
    main()
