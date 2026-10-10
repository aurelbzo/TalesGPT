"""FastAPI inference server for TinyGPT.

Endpoints
  GET  /health           liveness  (process is up)
  GET  /ready            readiness (model loaded + warmed up)
  GET  /metrics          Prometheus text format
  POST /generate         full completion + timing stats
  POST /generate/stream  Server-Sent Events, one event per token

Config (env vars)
  MODEL_PATH          artifacts/model_int8.pt
  TOKENIZER_PATH      artifacts/tokenizer.json
  MAX_CONCURRENCY     generations running at once per process (default 1; the rest queue)
  TORCH_THREADS       intra-op CPU threads (default: torch default). Set to your CPU limit in containers.
  MAX_NEW_TOKENS_CAP  hard cap on max_new_tokens (default 400)
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from tinygpt.generate import generate_stream
from tinygpt.quant import count_params
from tinygpt.utils import load_model, load_tokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_int8.pt")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "artifacts/tokenizer.json")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
MAX_NEW_TOKENS_CAP = int(os.getenv("MAX_NEW_TOKENS_CAP", "400"))
DEVICE = os.getenv("DEVICE", "cpu")
if os.getenv("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

S = {"ready": False}  # model, tok, eos, sem live here


class Metrics:
    def __init__(self):
        self.requests = self.errors = self.tokens = self.in_flight = self.waiting = 0
        self.latency_sum = self.ttft_sum = 0.0

    def render(self) -> str:
        n = max(self.requests, 1)
        return "\n".join([
            "# TYPE tinygpt_requests_total counter", f"tinygpt_requests_total {self.requests}",
            "# TYPE tinygpt_errors_total counter", f"tinygpt_errors_total {self.errors}",
            "# TYPE tinygpt_tokens_generated_total counter", f"tinygpt_tokens_generated_total {self.tokens}",
            "# TYPE tinygpt_in_flight gauge", f"tinygpt_in_flight {self.in_flight}",
            "# TYPE tinygpt_queue_depth gauge", f"tinygpt_queue_depth {self.waiting}",
            "# TYPE tinygpt_latency_seconds_avg gauge", f"tinygpt_latency_seconds_avg {self.latency_sum / n:.4f}",
            "# TYPE tinygpt_ttft_seconds_avg gauge", f"tinygpt_ttft_seconds_avg {self.ttft_sum / n:.4f}",
            "",
        ])


M = Metrics()


class GenerateRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, le=1000)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    model, cfg = load_model(MODEL_PATH, DEVICE)
    tok = load_tokenizer(TOKENIZER_PATH)
    S.update(model=model, cfg=cfg, tok=tok, eos=tok.token_to_id("<|endoftext|>"),
             sem=asyncio.Semaphore(MAX_CONCURRENCY))
    x = torch.tensor([tok.encode("Once upon a time").ids], device=DEVICE)  # warm-up
    list(generate_stream(model, x, 8, temperature=0.8, top_k=40))
    S["ready"] = True
    print(f"model ready in {time.perf_counter() - t0:.1f}s: {count_params(model)/1e6:.2f}M params, "
          f"threads={torch.get_num_threads()}, max_concurrency={MAX_CONCURRENCY}", flush=True)
    yield
    S["ready"] = False


app = FastAPI(title="TinyGPT-Serve", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not S["ready"]:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "params_m": round(count_params(S["model"]) / 1e6, 2)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return M.render()


async def _events(req: GenerateRequest, t_arrival: float):
    """Async generator of dicts: {'token': str} ... then {'done': True, stats}."""
    if not S["ready"]:
        raise HTTPException(503, "model not loaded")
    model, tok, eos, cfg = S["model"], S["tok"], S["eos"], S["cfg"]
    max_new = min(req.max_new_tokens, MAX_NEW_TOKENS_CAP)

    M.waiting += 1
    await S["sem"].acquire()
    M.waiting -= 1
    M.in_flight += 1
    t_start = time.perf_counter()
    ttft, ids, prev, n = None, [], "", 0
    try:
        prompt_ids = tok.encode(req.prompt).ids or [eos]
        prompt_ids = prompt_ids[-max(1, cfg.max_seq_len - max_new):]  # keep the most recent context
        x = torch.tensor([prompt_ids], device=DEVICE)
        gen_device = DEVICE if DEVICE == "cuda" else "cpu"
        gen = torch.Generator(device=gen_device).manual_seed(req.seed if req.seed is not None else random.randrange(2**31))
        stream = generate_stream(model, x, max_new, req.temperature, req.top_k, req.top_p,
                                 eos_id=eos, generator=gen)
        async for nxt in iterate_in_threadpool(stream):
            t = nxt.item()
            if ttft is None:
                ttft = time.perf_counter() - t_arrival
            if t == eos:
                break
            ids.append(t)
            n += 1
            text = tok.decode(ids)
            if text.endswith("�"):  # incomplete UTF-8 sequence: wait for the next token
                continue
            piece, prev = text[len(prev):], text
            if piece:
                yield {"token": piece}
        total = time.perf_counter() - t_arrival
        decode_s = max(time.perf_counter() - t_start, 1e-9)
        M.requests += 1
        M.tokens += n
        M.latency_sum += total
        M.ttft_sum += ttft or 0.0
        yield {"done": True, "tokens": n, "latency_ms": round(total * 1e3, 1),
               "queue_ms": round((t_start - t_arrival) * 1e3, 1),
               "ttft_ms": round((ttft or 0) * 1e3, 1), "tokens_per_s": round(n / decode_s, 1),
               "text": tok.decode(ids)}
    except Exception:
        M.errors += 1
        raise
    finally:
        M.in_flight -= 1
        S["sem"].release()


@app.post("/generate")
async def generate(req: GenerateRequest):
    t0 = time.perf_counter()
    final = None
    async for ev in _events(req, t0):
        if ev.get("done"):
            final = ev
    return JSONResponse({"prompt": req.prompt, "completion": final.pop("text"), **final})


@app.post("/generate/stream")
async def generate_stream_endpoint(req: GenerateRequest):
    t0 = time.perf_counter()

    async def sse():
        async for ev in _events(req, t0):
            ev.pop("text", None)
            yield f"data: {json.dumps(ev)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
