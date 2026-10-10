#!/usr/bin/env python
"""Turn a training checkpoint into deployable artifacts (fp32 + hand-rolled int8) next to the tokenizer.

  python scripts/export.py --ckpt runs/base_sft/best.pt --out_dir artifacts
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from tinygpt.quant import model_size_mb, quantize_model
from tinygpt.utils import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out_dir", default="artifacts")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(a.ckpt, "cpu")
    info = dict(config=cfg.to_dict(), params_m=model.num_params() / 1e6, source_ckpt=a.ckpt)

    torch.save(dict(state_dict=model.state_dict(), config=cfg.to_dict(), format="fp32"), out / "model_fp32.pt")
    info["fp32_mem_mb"] = model_size_mb(model)
    quantize_model(model)
    torch.save(dict(state_dict=model.state_dict(), config=cfg.to_dict(), format="int8"), out / "model_int8.pt")
    info["int8_mem_mb"] = model_size_mb(model)
    shutil.copy(a.tokenizer, out / "tokenizer.json")
    for f in ("model_fp32.pt", "model_int8.pt"):
        info[f + "_file_mb"] = (out / f).stat().st_size / 1e6
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# """Measure constraint-following (required words + dialogue yes/no) on held-out prompts.

#   python scripts/eval_constraints.py --ckpt runs/base/best.pt runs/base_sft/best.pt --n 200

# Checkpoint 1 (pretrained only) is the baseline; the SFT checkpoint should be far better.
# Success = all required words present AND dialogue constraint respected.
# """
# import argparse
# import json
# from pathlib import Path

# import torch

# from tinygpt.generate import generate_stream
# from tinygpt.sft_data import check
# from tinygpt.utils import load_model, load_tokenizer, pick_device


# def evaluate(model, tok, device, rows, a):
#     eos = tok.token_to_id("<|endoftext|>")
#     agg = dict(success=0, all_words=0, dialogue_ok=0, word_hits=0, n_words=0, n_tokens=0)
#     examples = []
#     gen = torch.Generator().manual_seed(a.seed)
#     for r in rows:
#         x = torch.tensor([tok.encode(r["prompt"]).ids], device=device)
#         new = []
#         for nxt in generate_stream(model, x, a.max_new_tokens, a.temperature, a.top_k, eos_id=eos, generator=gen):
#             t = nxt.item()
#             if t == eos:
#                 break
#             new.append(t)
#         story = tok.decode(new)
#         res = check(story, r["constraints"])
#         for k in ("success", "all_words", "dialogue_ok"):
#             agg[k] += int(res[k])
#         agg["word_hits"] += res["word_hits"]
#         agg["n_words"] += res["n_words"]
#         agg["n_tokens"] += len(new)
#         if len(examples) < 3:
#             examples.append(dict(prompt=r["prompt"], story=story[:600], result=res))
#     n = len(rows)
#     return dict(n=n, success_rate=agg["success"] / n, all_words_rate=agg["all_words"] / n,
#                 word_hit_rate=agg["word_hits"] / agg["n_words"], dialogue_ok_rate=agg["dialogue_ok"] / n,
#                 avg_tokens=agg["n_tokens"] / n), examples


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", nargs="+", required=True)
#     ap.add_argument("--tokenizer", default="data/tokenizer.json")
#     ap.add_argument("--data", default="data/smoke/sft/val.jsonl")
#     ap.add_argument("--n", type=int, default=200)
#     ap.add_argument("--max_new_tokens", type=int, default=400)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_k", type=int, default=40)
#     ap.add_argument("--device", default="auto")
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--out_dir", default="results/constraints")
#     a = ap.parse_args()

#     device = pick_device(a.device)
#     tok = load_tokenizer(a.tokenizer)
#     rows = [json.loads(l) for l in open(a.data)][:a.n]
#     results, all_examples = {}, {}
#     for ck in a.ckpt:
#         model, _ = load_model(ck, device)
#         res, ex = evaluate(model, tok, device, rows, a)
#         results[ck], all_examples[ck] = res, ex
#         print(ck, json.dumps(res, indent=2))

#     md = ["| checkpoint | success (all words + dialogue) | all words | avg words hit | dialogue ok | avg tokens |",
#           "|---|---|---|---|---|---|"]
#     for ck, r in results.items():
#         md.append(f"| `{ck}` | {r['success_rate']:.1%} | {r['all_words_rate']:.1%} | "
#                   f"{r['word_hit_rate']:.1%} | {r['dialogue_ok_rate']:.1%} | {r['avg_tokens']:.0f} |")
#     Path(a.out_dir).parent.mkdir(parents=True, exist_ok=True)
#     Path(a.out_dir + ".md").write_text("\n".join(md) + "\n")
#     Path(a.out_dir + ".json").write_text(json.dumps(dict(results=results, examples=all_examples), indent=2))
#     print("\n".join(md))


# if __name__ == "__main__":
#     main()
