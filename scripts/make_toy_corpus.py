#!/usr/bin/env python
"""Generate a tiny synthetic 'stories' corpus so the WHOLE pipeline can be smoke-tested offline in minutes.
(This is NOT TinyStories: the stories are template-based. It only proves the plumbing works.)"""
import argparse
import random
from pathlib import Path

NAMES = ["Lily", "Tom", "Mia", "Ben", "Sara", "Max", "Anna", "Leo"]
ANIMALS = ["dog", "cat", "bird", "rabbit", "duck", "frog"]
THINGS = ["ball", "kite", "boat", "cake", "book", "hat", "drum", "garden"]
PLACES = ["park", "forest", "beach", "farm", "school", "river"]
FEELINGS = ["happy", "sad", "excited", "brave", "tired", "proud"]


def story(rng):
    n, a, t, p, f = (rng.choice(x) for x in (NAMES, ANIMALS, THINGS, PLACES, FEELINGS))
    s = [f"One day, {n} went to the {p} with a little {a}.",
         f"{n} found a big {t} and felt very {f}."]
    if rng.random() < 0.6:
        s.append(f'"Look at this {t}!" said {n}. "It is so nice," said the {a}.')
    s.append(f"They played together until the sun went down. Then {n} went home and smiled.")
    return " ".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/toy")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for name, n in (("train", a.n_train), ("validation", a.n_val)):
        (out / f"{name}.txt").write_text("".join(story(rng) + "\n<|endoftext|>\n" for _ in range(n)))
    print(f"wrote toy corpus to {out}")


if __name__ == "__main__":
    main()
