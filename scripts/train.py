#!/usr/bin/env python
"""Pretrain TinyGPT from scratch on tokenized TinyStories.

Examples:
  python scripts/train.py --preset debug --max_steps 50 --device cpu       # smoke test (seconds)
  python scripts/train.py --preset small --bench_steps 30                  # measure tokens/s, print ETA, exit
  python scripts/train.py --preset base --batch_size 64 --grad_accum 1 --compile --out_dir runs/base   # GPU run
  python scripts/train.py --preset base --out_dir runs/base --resume       # continue after an interruption
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tinygpt.generate import complete
from tinygpt.model import GPT, GPTConfig, PRESETS
from tinygpt.utils import autocast_ctx, load_tokenizer, pick_device, pick_dtype, sync


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--data_dir", default="data/tinystories")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out_dir", default="runs/base")
    ap.add_argument("--batch_size", type=int, default=32, help="micro-batch (sequences per forward)")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=15_000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true", help="torch.compile (CUDA only; ~1 min warmup)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_minutes", type=float, default=0, help="stop (and checkpoint) after N minutes")
    ap.add_argument("--bench_steps", type=int, default=0, help="run N steps, print throughput + ETA, exit")
    return ap.parse_args()


def get_batch(data, batch_size, ctx, device, gen=None):
    ix = torch.randint(len(data) - ctx - 1, (batch_size,), generator=gen).tolist()
    x = torch.from_numpy(np.stack([data[i:i + ctx] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64))
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(step, a):
    min_lr = a.lr * a.min_lr_ratio
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if step >= a.max_steps:
        return min_lr
    ratio = (step - a.warmup) / max(1, a.max_steps - a.warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (a.lr - min_lr)


@torch.no_grad()
def estimate_val(model, data, a, ctx, device, amp_dtype):
    model.eval()
    gen = torch.Generator().manual_seed(0)  # same batches every time -> comparable val curves
    losses = []
    for _ in range(a.eval_iters):
        x, y = get_batch(data, a.batch_size, ctx, device, gen)
        with autocast_ctx(device, amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)
    amp_dtype, dtype_name = pick_dtype(device, a.dtype)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.data_dir) / "meta.json").read_text())
    cfg = GPTConfig(vocab_size=meta["vocab_size"], **PRESETS[a.preset])
    ctx = cfg.max_seq_len

    train_data = np.memmap(Path(a.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(Path(a.data_dir) / "val.bin", dtype=np.uint16, mode="r")
    tok = load_tokenizer(a.tokenizer) if Path(a.tokenizer).exists() else None

    raw = GPT(cfg).to(device)
    n_params = raw.num_params()
    tokens_per_step = a.batch_size * a.grad_accum * ctx
    print(f"device={device} dtype={dtype_name} params={n_params/1e6:.2f}M ctx={ctx} "
          f"tokens/step={tokens_per_step:,} total_tokens={tokens_per_step*a.max_steps/1e6:.0f}M "
          f"(train set: {meta['train_tokens']/1e6:.2f}M tokens)")

    decay = [p for p in raw.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw.parameters() if p.dim() < 2]
    opt_kwargs = dict(lr=a.lr, betas=(0.9, a.beta2))
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], **opt_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_name == "fp16" and device.type == "cuda"))

    start_step, best_val = 0, float("inf")
    ckpt_path, best_path, log_path = out / "ckpt.pt", out / "best.pt", out / "log.jsonl"
    if a.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from step {start_step} (best val {best_val:.4f})")
    elif log_path.exists() and not a.bench_steps:
        log_path.unlink()
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))

    model = torch.compile(raw) if (a.compile and device.type == "cuda") else raw
    model.train()

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t_start = time.time()
    t_last, steps_since = time.time(), 0
    loss_acc = torch.zeros((), device=device)
    for step in range(start_step, a.max_steps):
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            x, y = get_batch(train_data, a.batch_size, ctx, device)
            with autocast_ctx(device, amp_dtype):
                _, loss = model(x, y)
            scaler.scale(loss / a.grad_accum).backward()
            loss_acc += loss.detach() / a.grad_accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.grad_clip)
        scaler.step(opt)
        scaler.update()
        steps_since += 1

        done = step + 1
        if done % a.log_interval == 0 or done == a.bench_steps:
            sync(device)
            dt = time.time() - t_last
            tok_s = tokens_per_step * steps_since / dt
            eta_min = (a.max_steps - done) * tokens_per_step / tok_s / 60
            tl = loss_acc.item() / steps_since
            print(f"step {done:6d}/{a.max_steps} loss {tl:.4f} lr {lr:.2e} gnorm {float(gnorm):.2f} "
                  f"{tok_s/1e3:7.1f}k tok/s  ETA {eta_min:6.1f} min", flush=True)
            log(dict(step=done, train_loss=tl, lr=lr, tok_s=tok_s, elapsed_min=(time.time() - t_start) / 60))
            t_last, steps_since, loss_acc = time.time(), 0, torch.zeros((), device=device)
        if a.bench_steps and done >= a.bench_steps:
            print("bench done (first steps include compile/warmup; use the later numbers for ETA).")
            return

        over_time = a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes
        if done % a.eval_interval == 0 or done == a.max_steps or over_time:
            vl = estimate_val(raw, val_data, a, ctx, device, amp_dtype)
            print(f"  >> eval step {done}: val_loss {vl:.4f}  val_ppl {math.exp(vl):.2f}", flush=True)
            log(dict(step=done, val_loss=vl))
            atomic_save(dict(model=raw.state_dict(), optimizer=opt.state_dict(), step=done,
                             best_val=min(best_val, vl), config=cfg.to_dict()), ckpt_path)
            if vl < best_val:
                best_val = vl
                atomic_save(dict(model=raw.state_dict(), config=cfg.to_dict(), step=done, val_loss=vl), best_path)
            if tok is not None:
                raw.eval()
                print("  >> sample:", complete(raw, tok, "Once upon a time", 80, device=device,
                                               temperature=0.8, top_k=40).replace("\n", " "), flush=True)
                raw.train()
            t_last = time.time()
        if over_time:
            print(f"stopping: reached --max_minutes={a.max_minutes}")
            return
    print(f"done. best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f}). checkpoints in {out}/")


if __name__ == "__main__":
    main()