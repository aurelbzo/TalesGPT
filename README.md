# TinyGPT: a language model from scratch, from training to serving

A 30M-parameter GPT-style language model **written from scratch in PyTorch**, pretrained on ~490M tokens of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), instruction-tuned on a task whose success can be
checked automatically, quantized by hand, and served through a streaming API in Docker (and Kubernetes).

The goal was not to build a useful assistant, but to implement and **measure** every stage of an LLM's life:
tokenizer, pretraining, fine-tuning, evaluation, inference optimisation and deployment.

![Training and validation loss](results/loss_curve.png)

## Results at a glance

| | |
|---|---|
| **Model** | Decoder-only transformer, **29.9M parameters**: 8 layers, 8 heads, d_model 512, context 512, RoPE, SwiGLU, RMSNorm, tied embeddings |
| **Tokenizer** | Byte-level BPE, vocabulary 8,192, trained on 200k stories |
| **Pretraining** | 15,000 steps × 32,768 tokens = **491.5M tokens (≈ 1.05 epochs)** on one free Kaggle T4 GPU, ≈ 3.9 h |
| **Validation** | loss **1.365**, perplexity **3.92** |
| **Instruction tuning** | Dialogue-constraint adherence **47% → 90.5%**; required-word coverage **10.7% → 40.0%**; full success **0% → 6%** |
| **Quantization** | Hand-written int8 weights: **120 MB → 43 MB** in memory (2.8×) |
| **KV cache** | **TODO×** faster decoding than recomputing the full sequence |
| **Serving** | Streaming FastAPI server in Docker: **≈ 37 tokens/s per replica** (2 CPUs); latency grows linearly with concurrent users, which motivates horizontal scaling |

---

## 1. Model

A Llama-style decoder-only transformer (`tinygpt/model.py`), implemented with plain PyTorch operations:

- **RMSNorm** (pre-norm) instead of LayerNorm.
- **Rotary position embeddings (RoPE)** on queries and keys, so attention scores depend on relative positions.
- **SwiGLU** feed-forward layer (hidden size 1,408 ≈ 8/3 × d_model).
- **No biases**, and **tied input/output embeddings** (the output layer reuses the embedding matrix, saving 4.2M parameters).
- A **pre-allocated KV cache** for generation, with three attention paths: prefill, single-token decode, and a chunk after existing context.
- GPT-2-style initialisation, with residual projections scaled by 1/√(2·n_layer).

| Hyper-parameter | Value |
|---|---|
| Layers / heads / head dim | 8 / 8 / 64 |
| d_model / SwiGLU hidden | 512 / 1,408 |
| Context length | 512 tokens |
| Vocabulary | 8,192 |
| Parameters | 29,893,120 |

Correctness is covered by unit tests (`tests/`), including:
- **causality:** changing a future token must not change earlier outputs;
- **RoPE:** scores depend only on relative position;
- **KV cache:** prefill plus token-by-token decoding reproduces the logits of a full forward pass, and greedy outputs match with and without the cache;
- quantization error bounds and API behaviour.

## 2. Data

- **Corpus:** TinyStories: **2,119,719** training stories and **21,990** validation stories, written in simple English.
- **Tokenizer:** byte-level BPE with vocabulary 8,192 and an `<|endoftext|>` special token, trained on the first 200k stories.
- **Preprocessing** (`scripts/prepare_data.py`): every story is tokenized and followed by `<|endoftext|>`, and all stories are concatenated into one flat `uint16` array.
  - The training split is **466.8M tokens** (about 1 GB on disk).
  - Training reads random 512-token windows from it through a memory map, so no padding is needed and the RAM footprint is tiny.

## 3. Pretraining

| Setting | Value |
|---|---|
| Hardware | 1× NVIDIA T4 (Kaggle, free tier), fp16 mixed precision with loss scaling |
| Batch | 16 sequences × 4 gradient-accumulation steps × 512 tokens = 32,768 tokens/step |
| Steps / tokens | 15,000 / 491.5M (≈ 1.05 epochs) |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on matrices only, gradient clipping at 1.0 |
| Learning rate | 300-step linear warm-up to 6e-4, cosine decay to 6e-5 |
| Throughput / time | ≈ 46k tokens/s, ≈ 3.9 h |

**Validation loss during training:**

| Step | 500 | 1,000 | 2,000 | 5,000 | 10,000 | 15,000 |
|---|---|---|---|---|---|---|
| Val loss | 2.239 | 1.926 | 1.740 | 1.555 | 1.422 | **1.365** |
| Val perplexity | 9.39 | 6.86 | 5.70 | 4.74 | 4.15 | **3.92** |

- The loss starts at ≈ 9.0 (= ln 8,192, uniform guessing) and decreases steadily.
- It is still slowly improving at the end, so more data or a larger model would help.
- A perplexity of 3.92 means the model is, on average, about as uncertain as choosing between ~4 plausible next tokens.

**Sample after pretraining** (prompt "Once upon a time", temperature 0.8):

> Once upon a time there was a boy named Jack. He loved cars and trucks. One day he saw a big, shiny car in the park.
> He went over to it and he noticed it had no fuel. Jack asked, "What is fuel for?" The car replied, "My fuel is what
> makes it go. It is a special kind of fuel that gives it to people who need it."

The text is fluent and grammatical, with dialogue and simple plots. Long-range logic is weaker: the last sentence drifts.

## 4. Instruction tuning on a verifiable task

The pretrained model only continues text. To teach it to **follow instructions**, I designed a task whose success can be
checked by code:

```
Write a short story.
Words: ball, happy, garden
Dialogue: yes
Story:
<the model writes the story>
```

**Data** (`scripts/make_sft_data.py`):
- The constraints are extracted from real TinyStories stories: three words of 4-9 letters that appear in the story (common words excluded), plus whether it contains dialogue.
- So a valid answer always exists: the original story.
- Training examples come from the train split, and evaluation prompts from the validation split, so the model is never tested on stories it trained on.

**Training** (`scripts/sft.py`):
- 57,703 examples (2,297 longer than 512 tokens were skipped), 1 epoch, 1,804 steps, batch 32.
- Learning rate 2e-4 with cosine decay, ≈ 11 minutes on the T4.
- The loss is computed **only on the story tokens**: prompt and padding positions are masked with `-100`, so the model learns to answer, not to reproduce the template.
- Validation loss on responses went from 1.183 to 1.115.

**Evaluation** (`scripts/eval_constraints.py`):
- 200 held-out prompts, temperature 0.7, top-k 40.
- Each generated story is checked automatically: required words must appear as **whole words**, and quotation marks must be present if and only if dialogue was requested.

| Checkpoint | Success (all words + dialogue) | All 3 words | Words hit | Dialogue OK | Avg. tokens |
|---|---|---|---|---|---|
| Pretrained only | 0.0% | 0.0% | 10.7% | 47.0% | 125 |
| **After SFT** | **6.0%** | **6.0%** | **40.0%** | **90.5%** | 189 |

**Interpretation:**
- **Dialogue control was learned almost perfectly** (47% is chance level; 90.5% after SFT). It is a single global decision about the story.
- **Word coverage improved about 4×,** but including all three words in one story remains hard (6%). It requires *planning* to place several specific, often unusual words in the narrative, which is difficult for a 30M-parameter model after one short epoch.
- The validation loss barely moved (1.18 → 1.12) while instruction-following changed a lot. The loss measures how well the model predicts story text, not whether it obeys the prompt, which is why a task-specific evaluation is necessary.

**Example after SFT** (prompt: *Words: TODO · Dialogue: TODO*):

> TODO: paste one good and one failed generation here

## 5. Inference optimisation

### KV cache

Without a cache, generating each new token recomputes attention over the whole sequence (O(n²) work in total). With the
cache, past keys and values are stored and each step processes a single token (O(n)).

TODO: paste `results/bench_kvcache.md`

Memory cost of the cache per sequence: 2 × 8 layers × 512 positions × 512 dims × 4 bytes ≈ **16.8 MB** (fp32, full context).

### Weight-only int8 quantization

Each linear layer's weights are quantized per output channel: `w_q = round(w / s)` with `s = max|w| / 127`, stored as int8
together with one float scale per channel. At run time, `y = (x · w_qᵀ) · s`. The embedding / output matrix stays in fp32.

| Variant | Memory |
|---|---|
| fp32 | 119.7 MB |
| int8 (hand-written) | **42.8 MB** (2.8× smaller) |

TODO: paste `results/bench_quant.md` (perplexity and speed of fp32 vs int8)

The weights are converted back to float before each matrix multiplication. That saves memory but not compute: a real
speed-up needs int8 matrix-multiplication kernels.

## 6. Serving

`serve/app.py` is a FastAPI server:

| Endpoint | Purpose |
|---|---|
| `POST /generate` | Full completion plus timing stats (latency, time to first token, tokens/s, queue time) |
| `POST /generate/stream` | Server-Sent Events: one event per generated token |
| `GET /health` · `GET /ready` | Liveness (process up) and readiness (model loaded and warmed up) |
| `GET /metrics` | Prometheus-format counters: requests, tokens, in-flight requests, queue depth, average latency |

Design choices:
- The model is loaded and warmed up once at start-up.
- Generation runs in a worker thread, so the server stays responsive.
- A semaphore runs one generation at a time per process, and extra requests queue. Generation is CPU-bound, so parallel generations would only slow each other down.
- Each request gets its own random generator, so the `seed` parameter gives reproducible outputs.

### Docker

The image is based on `python:3.11-slim` with CPU-only PyTorch. It runs as a non-root user, has a health check, and contains only the int8 model.
Image size: **TODO MB**.

**Load test** (`scripts/load_test.py`): closed-loop users sending 128-token requests for 30 s each, against the container limited to 2 CPUs.

| Concurrent users | Requests | Errors | Tokens/s (total) | p50 latency | p95 latency | Time to first token (p50) |
|---|---|---|---|---|---|---|
| 1 | 9 | 0 | 37.2 | 3.40 s | 3.68 s | 32 ms |
| 2 | 12 | 0 | 37.4 | 6.25 s | 6.81 s | 3,405 ms |
| 4 | 13 | 0 | 37.7 | 13.18 s | 13.56 s | 9,848 ms |

- **Throughput is flat** (≈ 37 tokens/s): one replica generates one request at a time.
- **Latency grows linearly** with the number of users (≈ 3.4 s × users).
- **Time to first token is almost entirely queueing:** 32 ms alone, then one full request of waiting per user ahead.
- One replica therefore has a fixed capacity. Serving more users needs **more replicas** (horizontal scaling) or **batching** several requests into one forward pass.

TODO (optional): add `results/loadtest_local-int8.md` (same test without Docker) for comparison.

### Kubernetes

TODO (optional): describe the kind cluster, the Deployment (readiness, liveness and startup probes, CPU requests and limits), the Service and the
Horizontal Pod Autoscaler, plus what was observed under load (time to scale up, number of replicas, latency before and after).

## 7. Limitations

- A 30M-parameter model trained on children's stories is **not a general assistant**: narrow vocabulary, weak long-range consistency.
- The constraint check is purely lexical (whole-word match, quotation marks). It does not judge story quality, and inflected forms ("played" for "play") count as misses.
- Full constraint success is low (6%) after a single short SFT epoch. Longer fine-tuning was not explored.
- The hand-written int8 reduces memory but not compute; see the benchmark above.
- The server handles one request at a time per replica and does no batching. Load tests ran on a laptop, with the client on the same machine, so absolute latencies are indicative.

## 8. What I would do next

- **Better SFT:** more epochs and a learning-rate sweep, plus tolerance for inflected forms in the checker.
- **Reinforcement learning:** the constraint checker is a ready-made reward function, so it could drive a GRPO-style policy-optimisation step on top of SFT.
- **Continuous batching** in the server, to merge many users' requests into one forward pass.
- **Custom GPU kernels** (Triton) for fused RMSNorm or SwiGLU.
- **Retrieval-augmented generation** on top of the same serving stack.

---

## Reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m pytest -q                       # unit and API tests
bash scripts/smoke.sh                     # whole pipeline on a tiny synthetic corpus (CPU, a few minutes)

# on a GPU machine
make tokenizer data
make train PRESET=base TRAIN_ARGS="--batch_size 16 --grad_accum 4"
make sft-data sft eval-constraints PRESET=base
make export PRESET=base

# serving (any machine)
MODEL_PATH=artifacts/model_int8.pt TOKENIZER_PATH=artifacts/tokenizer.json uvicorn serve.app:app --port 8000
docker build -t tinygpt-serve . && docker run -p 8000:8000 tinygpt-serve
```

Example request:

```bash
curl -N -X POST localhost:8000/generate/stream -H 'content-type: application/json' \
  -d '{"prompt":"Write a short story.\nWords: dragon, cookie, brave\nDialogue: yes\nStory:\n","max_new_tokens":200}'
```

## Repository layout

```
tinygpt/model.py         transformer: RMSNorm, RoPE, SwiGLU, KV cache, weight tying
tinygpt/generate.py      sampling (temperature, top-k, top-p), cached and naive generation
tinygpt/quant.py         per-channel weight-only int8 quantization
tinygpt/sft_data.py      constrained-story prompts and the automatic checker
tinygpt/data_io.py       story iterators (Hugging Face dataset or local text files)
scripts/                 tokenizer, data preparation, training, SFT, evaluation, export, benchmarks, load test
serve/app.py             streaming FastAPI server with queueing and metrics
k8s/                     kind cluster, Deployment, Service, HorizontalPodAutoscaler
tests/                   model, data/SFT and API tests
```

## License

MIT
