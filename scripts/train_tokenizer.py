import argparse
import sys
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tinygpt.data_io import EOT, iter_stories

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf", help="'hf' or a directory with train.txt / validation.txt")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_stories", type=int, default=200_000, help="stories used to learn merges")
    ap.add_argument("--out", default="data/tokenizer.json")
    a = ap.parse_args()

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    stories = list(iter_stories(a.source, "train", a.max_stories))
    print(f"loaded {len(stories)} stories")
    tok.train_from_iterator(stories, trainer=trainer, length=len(stories))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(a.out)
    print(f"saved {a.out}: vocab={tok.get_vocab_size()} eot_id={tok.token_to_id(EOT)}")
    s = "Once upon a time, a little girl named Lily had a red ball."
    assert all(s.strip() for s in iter_stories("data/smoke/toy", "train"))
    ids = tok.encode(s).ids
    print(f"sanity: {len(ids)} tokens -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    sys.exit(main())