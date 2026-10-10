"""Device / dtype helpers and checkpoint loading shared by all scripts."""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, name: str = "auto"):
    """Returns (autocast dtype or None, resolved name).
    CUDA: bf16 on Ampere+ (native), fp16 otherwise (needs GradScaler). MPS/CPU: fp32 (safest)."""
    if name == "auto":
        if device.type == "cuda":
            name = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
        else:
            name = "fp32"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name], name


def autocast_ctx(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def load_model(path: str | Path, device="cpu", dtype=None):
    """Load a training checkpoint ({'model', 'config'}) or an exported artifact
    ({'state_dict', 'config', 'format'}), including int8-quantized ones."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    if ck.get("format", "fp32") == "int8":
        from .quant import quantize_model
        quantize_model(model)
    sd = ck.get("state_dict", ck.get("model"))
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    if dtype is not None and ck.get("format", "fp32") != "int8":
        model = model.to(dtype)
    return model.to(device).eval(), cfg