"""End-to-end API test with a tiny randomly-initialised model (no training needed)."""
import json
import os

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tinygpt.model import GPT, GPTConfig
from tinygpt.quant import quantize_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["once upon a time there was a little cat who liked to play"] * 50, trainer=trainer)
    tok.save(str(d / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2, d_model=32, max_seq_len=64)
    qm = quantize_model(GPT(cfg))
    torch.save(dict(state_dict=qm.state_dict(), config=cfg.to_dict(), format="int8"), d / "model_int8.pt")
    from serve import app as appmod
    appmod.MODEL_PATH, appmod.TOKENIZER_PATH = str(d / "model_int8.pt"), str(d / "tokenizer.json")
    with TestClient(appmod.app) as c:
        yield c


def test_health_ready_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    assert "tinygpt_requests_total" in client.get("/metrics").text


def test_generate(client):
    r = client.post("/generate", json={"prompt": "once upon", "max_new_tokens": 10, "seed": 1})
    assert r.status_code == 200
    j = r.json()
    assert j["tokens"] <= 10 and "completion" in j and j["latency_ms"] > 0
    r2 = client.post("/generate", json={"prompt": "once upon", "max_new_tokens": 10, "seed": 1}).json()
    assert r2["completion"] == j["completion"]  # same seed -> same completion


def test_stream(client):
    with client.stream("POST", "/generate/stream", json={"prompt": "a little", "max_new_tokens": 8, "seed": 2}) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    done = json.loads(lines[-2][6:])
    assert done["done"] is True
    assert sum("token" in json.loads(l[6:]) for l in lines[:-2]) <= done["tokens"]


def test_validation_error(client):
    assert client.post("/generate", json={"max_new_tokens": 0}).status_code == 422
