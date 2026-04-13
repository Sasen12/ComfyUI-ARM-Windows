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


def test_attention_split_retries_after_late_oom(monkeypatch):
    q = torch.randn(1, 4, 8, dtype=torch.float32)
    k = torch.randn(1, 4, 8, dtype=torch.float32)
    v = torch.randn(1, 4, 8, dtype=torch.float32)
    expected = torch.zeros_like(q)
    state = {"query_einsum_calls": 0, "raised_once": False}

    def fake_einsum(equation, *operands):
        if equation == "b i d, b j d -> b i j":
            state["query_einsum_calls"] += 1
            if state["query_einsum_calls"] == 2 and not state["raised_once"]:
                state["raised_once"] = True
                raise RuntimeError(
                    "Could not allocate tensor with 70385664 bytes. There is not enough GPU video memory available!"
                )
            query_slice, key_tensor = operands
            return torch.zeros(
                query_slice.shape[0],
                query_slice.shape[1],
                key_tensor.shape[-1],
                dtype=query_slice.dtype,
                device=query_slice.device,
            )
        if equation == "b i j, b j d -> b i d":
            scores, value_tensor = operands
            return torch.zeros(
                scores.shape[0],
                scores.shape[1],
                value_tensor.shape[-1],
                dtype=scores.dtype,
                device=scores.device,
            )
        raise AssertionError(f"Unexpected einsum equation: {equation}")

    monkeypatch.setattr(attention_module, "einsum", fake_einsum)
    monkeypatch.setattr(attention_module.model_management, "get_free_memory", lambda *args, **kwargs: 100)
    monkeypatch.setattr(attention_module.model_management, "soft_empty_cache", lambda *args, **kwargs: None)

    out = attention_module.attention_split(q, k, v, heads=2)

    assert torch.equal(out, expected)
    assert state["raised_once"] is True
    assert state["query_einsum_calls"] >= 4
