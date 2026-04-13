from __future__ import annotations

import torch

import comfy.model_management as model_management
from comfy.ldm.modules import attention as attention_module


def test_is_oom_recognizes_directml_video_memory_error():
    exc = RuntimeError(
        "Could not allocate tensor with 134217728 bytes. There is not enough GPU video memory available!"
    )

    assert model_management.is_oom(exc) is True


def test_attention_sub_quad_falls_back_to_split_on_directml_oom(monkeypatch):
    q = torch.randn(1, 4, 8, dtype=torch.float32)
    k = torch.randn(1, 4, 8, dtype=torch.float32)
    v = torch.randn(1, 4, 8, dtype=torch.float32)
    fallback = torch.full((1, 4, 8), 3.14, dtype=torch.float32)

    def boom(*args, **kwargs):
        raise RuntimeError(
            "Could not allocate tensor with 134217728 bytes. There is not enough GPU video memory available!"
        )

    def fake_split(*args, **kwargs):
        return fallback

    monkeypatch.setattr(attention_module, "efficient_dot_product_attention", boom)
    monkeypatch.setattr(attention_module, "attention_split", fake_split)
    monkeypatch.setattr(
        attention_module.model_management,
        "get_free_memory",
        lambda *args, **kwargs: (1_000_000_000, 1_000_000_000),
    )

    out = attention_module.attention_sub_quad(q, k, v, heads=2)

    assert torch.equal(out, fallback)
