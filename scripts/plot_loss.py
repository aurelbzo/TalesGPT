#!/usr/bin/env python
"""Plot train/val loss and throughput from one or more runs/*/log.jsonl."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/base"])
    ap.add_argument("--out", default="results/loss_curve.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in a.runs:
        recs = [json.loads(l) for l in open(Path(r) / "log.jsonl")]
        tr = [(x["step"], x["train_loss"]) for x in recs if "train_loss" in x]
        va = [(x["step"], x["val_loss"]) for x in recs if "val_loss" in x]
        ts = [(x["step"], x["tok_s"] / 1e3) for x in recs if "tok_s" in x]
        name = Path(r).name
        if tr:
            ax[0].plot(*zip(*tr), label=f"{name} train", alpha=0.6)
        if va:
            ax[0].plot(*zip(*va), "o-", label=f"{name} val")
        if ts:
            ax[1].plot(*zip(*ts), label=name)
    ax[0].set(xlabel="step", ylabel="cross-entropy (nats)", title="Loss", yscale="log")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set(xlabel="step", ylabel="k tokens / s", title="Training throughput")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
