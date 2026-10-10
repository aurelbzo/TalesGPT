#!/usr/bin/env python
"""Build the constrained-story SFT dataset (see tinygpt/sft_data.py for the format).
Train examples come from the TinyStories train split; held-out eval prompts from the validation split."""
import argparse
import json
import random
from pathlib import Path

from tinygpt.data_io import iter_stories
from tinygpt.sft_data import build_prompt, extract_constraints


def build(source, split, n, rng, out_path):
    k = 0
    with open(out_path, "w") as f:
        for story in iter_stories(source, split, max_n=int(n * 1.2) + 10):
            c = extract_constraints(story, rng)
            if c is None:
                continue
            f.write(json.dumps(dict(prompt=build_prompt(c), response=story, constraints=c)) + "\n")
            k += 1
            if k >= n:
                break
    print(f"{split}: wrote {k} examples -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf")
    ap.add_argument("--out_dir", default="data/sft")
    ap.add_argument("--n_train", type=int, default=60_000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    build(a.source, "train", a.n_train, rng, out / "train.jsonl")
    build(a.source, "validation", a.n_val, rng, out / "val.jsonl")


if __name__ == "__main__":
    main()
