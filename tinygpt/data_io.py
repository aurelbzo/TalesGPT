from __future__ import annotations

from pathlib import Path
from typing import Iterator

EOT = "<|endoftext|>"
HF_NAME = "roneneldan/TinyStories"

def iter_stories(source: str, split: str = "train", max_n: int | None = None):
    n = 0
    if source == "hf":
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, split=split)     # HF_NAME = "roneneldan/TinyStories"
        for row in ds:
            text = row["text"].strip()
            if text:
                yield text
                n += 1
                if max_n and n >= max_n:
                    return
        return

    path = Path(source) / f"{split}.txt"
    buf: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                story = "".join(buf).strip()
                buf = []                              # 1. reset after every story
                if story:                             # 2. skip empty stories
                    yield story
                    n += 1
                    if max_n and n >= max_n:
                        return
            else:
                buf.append(line)
    tail = "".join(buf).strip()                       # 3. text after the last EOT, if any
    if tail and not (max_n and n >= max_n):
        yield tail
        