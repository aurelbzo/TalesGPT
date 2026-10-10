import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tinygpt.data_io import EOT, iter_stories

def tokenize_split(tok, source, split, out_path, max_stories, chunk=10000):
    eot = tok.token_to_id(EOT)
    n_tokens, n_stories, batch = 0, 0, []

    def flush(f):
        nonlocal n_tokens
        for enc in tok.encode_batch(batch):
            arr = np.array(enc.ids + [eot], dtype=np.uint16)
            arr.tofile(f)
            n_tokens += len(arr)

    with open(out_path, "wb") as f:
        for story in iter_stories(source, split, max_stories):
            batch.append(story)
            n_stories += 1
            if len(batch) >= chunk:
                flush(f)
                batch = []
                print(f" {split}: {n_stories:,} stories, {n_tokens:,} tokens", flush=True)
        if batch:
            flush(f)
        
    print(f"{split}: {n_stories:,} stories, {n_tokens:,} tokens -> {out_path}")
    return n_stories, n_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out_dir",default="data/tinystories")
    ap.add_argument("--max_train_stories", type=int, default=None, help="subset for quick experiments")
    ap.add_argument("--max_val_stories", type=int, default=None)
    a = ap.parse_args() 

    tok = Tokenizer.from_file(a.tokenizer)
    assert tok.get_vocab_size() < 65536
    out = Path(a.out_dir)
    ns_tr, nt_tr = tokenize_split(tok, a.source, "train", out / "train.bin", a.max_train_stories)
    ns_va, nt_va = tokenize_split(tok, a.source, "validation", out / "val.bin", a.max_val_stories)
    meta = dict(vocab_size=tok.get_vocab_size(), eot_id=tok.token_to_id(EOT),
                train_stories=ns_tr, train_tokens=nt_tr, val_stories=ns_va, val_tokens=nt_va)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()