# Usage: make <target>.  Override variables like:  make train PRESET=small
PRESET   ?= base
RUN      ?= runs/$(PRESET)
SFT_RUN  ?= $(RUN)_sft
SOURCE   ?= hf
IMAGE    ?= tinygpt-serve:latest
export PYTHONPATH := .

.PHONY: help setup test smoke tokenizer data bench-train train plot sft-data sft eval-constraints export \
        bench-kv bench-quant serve docker-build docker-run loadtest-local loadtest-docker \
        kind-up k8s-deploy loadtest-k8s k8s-down

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e . --no-deps

test:
	python -m pytest -q

smoke:                  # whole pipeline on a toy corpus, a few minutes, no downloads
	bash scripts/smoke.sh

tokenizer:
	python scripts/train_tokenizer.py --source $(SOURCE)

data:
	python scripts/prepare_data.py --source $(SOURCE)

bench-train:            # measure tokens/s and print the ETA for a full run, then exit
	python scripts/train.py --preset $(PRESET) --out_dir $(RUN) --bench_steps 40 $(TRAIN_ARGS)

train:
	python scripts/train.py --preset $(PRESET) --out_dir $(RUN) $(TRAIN_ARGS)

plot:
	python scripts/plot_loss.py --runs $(RUN)

sft-data:
	python scripts/make_sft_data.py --source $(SOURCE)

sft:
	python scripts/sft.py --ckpt $(RUN)/best.pt --out_dir $(SFT_RUN) $(SFT_ARGS)

eval-constraints:
	python scripts/eval_constraints.py --ckpt $(RUN)/best.pt $(SFT_RUN)/best.pt --n 200

export:
	python scripts/export.py --ckpt $(SFT_RUN)/best.pt --out_dir artifacts

bench-kv:
	python scripts/bench_kvcache.py --ckpt $(SFT_RUN)/best.pt --devices cpu

bench-quant:
	python scripts/bench_quant.py --ckpt $(SFT_RUN)/best.pt

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000

loadtest-local:
	python scripts/load_test.py --url http://localhost:8000 --concurrency 1,2,4,8 --duration 30 --label local-int8

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --cpus=2 -e TORCH_THREADS=2 $(IMAGE)

loadtest-docker:
	python scripts/load_test.py --url http://localhost:8000 --concurrency 1,2,4 --duration 30 --label docker-int8

kind-up:
	kind create cluster --name tinygpt --config k8s/kind-config.yaml
	kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	kubectl patch -n kube-system deployment metrics-server --type=json \
	  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

k8s-deploy:
	kind load docker-image $(IMAGE) --name tinygpt
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
	kubectl rollout status deployment/tinygpt --timeout=180s

loadtest-k8s:
	python scripts/load_test.py --url http://localhost:30080 --concurrency 8 --duration 180 --label k8s-hpa

k8s-down:
	kind delete cluster --name tinygpt
