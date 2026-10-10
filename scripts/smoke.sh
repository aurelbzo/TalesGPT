#!/usr/bin/env bash
# End-to-end smoke test on a tiny synthetic corpus (no downloads, no GPU, a few minutes on a laptop CPU).
# tokenizer -> data -> pretrain -> SFT -> constraint eval -> export -> KV-cache bench -> quant bench.
# Writes only to data/smoke, runs/smoke*, artifacts_smoke, results/smoke_* (never touches your real run).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
D=data/smoke
python scripts/make_toy_corpus.py --out_dir $D/toy
python scripts/train_tokenizer.py --source $D/toy --vocab_size 512 --out $D/tokenizer.json
python scripts/prepare_data.py --source $D/toy --tokenizer $D/tokenizer.json --out_dir $D/tinystories
python scripts/train.py --preset debug --device cpu --batch_size 16 --grad_accum 1 --max_steps 300 --warmup 20 \
    --lr 2e-3 --eval_interval 100 --eval_iters 10 --log_interval 50 \
    --data_dir $D/tinystories --tokenizer $D/tokenizer.json --out_dir runs/smoke
python scripts/plot_loss.py --runs runs/smoke --out results/smoke_loss_curve.png
python scripts/make_sft_data.py --source $D/toy --n_train 3000 --n_val 100 --out_dir $D/sft
python scripts/sft.py --ckpt runs/smoke/best.pt --out_dir runs/smoke_sft --device cpu --batch_size 16 --lr 1e-3 \
    --train $D/sft/train.jsonl --val $D/sft/val.jsonl --tokenizer $D/tokenizer.json
python scripts/eval_constraints.py --ckpt runs/smoke/best.pt runs/smoke_sft/best.pt --n 60 --device cpu \
    --data $D/sft/val.jsonl --tokenizer $D/tokenizer.json --out results/smoke_constraints
python scripts/export.py --ckpt runs/smoke_sft/best.pt --tokenizer $D/tokenizer.json --out_dir artifacts_smoke
python scripts/bench_kvcache.py --ckpt runs/smoke_sft/best.pt --prompt_lens 16,64 --new_tokens 48 \
    --batch_sizes 1,8 --repeats 2 --out results/smoke_bench_kvcache
python scripts/bench_quant.py --ckpt runs/smoke_sft/best.pt --val $D/tinystories/val.bin --new_tokens 48 \
    --repeats 2 --n_windows 40 --out results/smoke_bench_quant
echo
echo "SMOKE TEST PASSED. Try the server:"
echo "  MODEL_PATH=artifacts_smoke/model_int8.pt TOKENIZER_PATH=artifacts_smoke/tokenizer.json make serve"
