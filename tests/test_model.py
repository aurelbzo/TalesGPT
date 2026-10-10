import copy

import pytest
import torch

from tinygpt.generate import generate, sample_next
from tinygpt.model import GPT, GPTConfig, KVCache, PRESETS, apply_rope, precompute_rope
from tinygpt.quant import QuantLinear, model_size_mb, quantize_model


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=100, n_layer=2, n_head=4, d_model=64, max_seq_len=64)
    return GPT(cfg).eval()


def test_forward_shapes_and_loss(model):
    x = torch.randint(0, 100, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 100)
    assert loss.item() > 0
    last, _ = model(x)
    assert last.shape == (3, 1, 100)


def test_initial_loss_is_about_log_vocab(model):
    # independent random targets (targets == inputs would be 'easy' because of tied embeddings)
    x = torch.randint(0, 100, (8, 32))
    y = torch.randint(0, 100, (8, 32))
    _, loss = model(x, y)
    assert abs(loss.item() - torch.log(torch.tensor(100.0)).item()) < 0.5


def test_weight_tying(model):
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_causality(model):
    """Changing a future token must not change earlier logits."""
    x = torch.randint(0, 100, (1, 20))
    y = x.clone()
    y[0, 12] = (y[0, 12] + 1) % 100
    a, _ = model(x, last_only=False)
    b, _ = model(y, last_only=False)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-5)


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = precompute_rope(16, 32, 10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    assert abs(score(3, 1) - score(10, 8)) < 1e-4  # same offset (2)
    assert abs(score(3, 1) - score(3, 2)) > 1e-3   # different offset


def test_kv_cache_matches_full_forward(model):
    """Prefill + token-by-token decoding must reproduce the logits of one full forward pass."""
    x = torch.randint(0, 100, (2, 24))
    full, _ = model(x, last_only=False)
    cache = KVCache(model.cfg, batch_size=2)
    pre, _ = model(x[:, :10], cache=cache, last_only=False)
    outs = [pre]
    for t in range(10, 24):
        step, _ = model(x[:, t:t + 1], cache=cache, last_only=False)
        outs.append(step)
    cached = torch.cat(outs, dim=1)
    assert torch.allclose(full, cached, atol=1e-4), (full - cached).abs().max()


def test_chunked_prefill_with_existing_context(model):
    x = torch.randint(0, 100, (1, 30))
    full, _ = model(x, last_only=False)
    cache = KVCache(model.cfg, batch_size=1)
    a, _ = model(x[:, :12], cache=cache, last_only=False)
    b, _ = model(x[:, 12:], cache=cache, last_only=False)  # multi-token chunk after context -> mask path
    assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-4)


def test_greedy_cached_equals_naive(model):
    prompt = torch.randint(0, 100, (1, 8))
    a = generate(model, prompt, 30, temperature=0.0, use_cache=True)
    b = generate(model, prompt, 30, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_generation_respects_context_limit(model):
    prompt = torch.randint(0, 100, (1, 60))
    out = generate(model, prompt, 100, temperature=0.0)
    assert out.shape[1] == model.cfg.max_seq_len


def test_sampling_top_k_and_top_p():
    logits = torch.tensor([[10.0, 9.0, 0.0, 0.0, -5.0]])
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        assert sample_next(logits, 1.0, top_k=2, generator=g).item() in (0, 1)
        assert sample_next(logits, 1.0, top_p=0.5, generator=g).item() == 0


def test_presets_param_counts():
    counts = {name: GPT(GPTConfig(vocab_size=8192, **p)).num_params() / 1e6 for name, p in PRESETS.items()}
    assert 1.0 < counts["debug"] < 3.5
    assert 15 < counts["small"] < 20
    assert 27 < counts["base"] < 33


def test_quant_linear_error_is_small():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 128, bias=False)
    q = QuantLinear.from_linear(lin)
    x = torch.randn(8, 256)
    err = (q(x) - lin(x)).norm() / lin(x).norm()
    assert err < 0.02
    assert q.qweight.dtype == torch.int8


def test_quantized_model_close_and_smaller(model):
    qm = quantize_model(copy.deepcopy(model))
    x = torch.randint(0, 100, (2, 16))
    _, la = model(x, x)
    _, lb = qm(x, x)
    assert abs(la.item() - lb.item()) < 0.05
    assert model_size_mb(qm) < model_size_mb(model)


def test_quantized_state_dict_roundtrip(model):
    qm = quantize_model(copy.deepcopy(model))
    fresh = quantize_model(GPT(model.cfg))
    fresh.load_state_dict(qm.state_dict())
    x = torch.randint(0, 100, (1, 12))
    assert torch.allclose(qm(x)[0], fresh(x)[0])