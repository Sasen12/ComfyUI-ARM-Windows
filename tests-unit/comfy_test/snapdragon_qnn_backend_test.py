from __future__ import annotations

from types import SimpleNamespace

import torch

from comfy import snapdragon_qnn_backend as qnn_backend


def test_make_calibration_samples_filters_requested_inputs():
    xc = torch.ones(1, 4, 2, 2)
    t = torch.tensor([0.5])
    context = torch.ones(1, 77, 8)

    samples = qnn_backend._make_calibration_samples(xc, t, context, ("xc", "context"))

    assert len(samples) == 3
    assert all("xc" in sample for sample in samples)
    assert all("context" in sample for sample in samples)
    assert all("t" not in sample for sample in samples)


def test_backend_run_only_feeds_exported_inputs(monkeypatch):
    captured = {}

    def fake_run_session(session_id, inputs, output_name=None):
        captured["session_id"] = session_id
        captured["inputs"] = inputs
        captured["output_name"] = output_name
        return torch.ones(1, 4, 2, 2)

    monkeypatch.setattr(qnn_backend.qnn_runtime, "run_session", fake_run_session)

    backend = qnn_backend.SnapdragonQnnUnetBackend(
        model_signature="sig",
        cache_dir=qnn_backend.Path("C:/temp"),
        export_path=qnn_backend.Path("C:/temp/model.onnx"),
        preprocessed_path=qnn_backend.Path("C:/temp/model.pre.onnx"),
        quantized_path=qnn_backend.Path("C:/temp/model.qdq.onnx"),
        session_id="session-1",
        providers=["QNNExecutionProvider", "CPUExecutionProvider"],
        input_names=("xc", "context"),
        shape_signature=((1, 4, 2, 2), ("torch.float32", "torch.float32"), ("torch.float32", "torch.float32"), ("torch.float32", "torch.float32")),
    )

    out = backend.run(
        torch.ones(1, 4, 2, 2),
        torch.tensor([0.5]),
        torch.ones(1, 77, 8),
    )

    assert out.shape == (1, 4, 2, 2)
    assert captured["session_id"] == "session-1"
    assert set(captured["inputs"]) == {"xc", "context"}
    assert captured["output_name"] is None


def test_try_apply_qnn_backend_uses_cached_backend(monkeypatch):
    class FakeBackend:
        def __init__(self):
            self.calls = 0

        def run(self, xc, t, context):
            self.calls += 1
            return xc + 2

    fake_backend = FakeBackend()
    model = SimpleNamespace()
    xc = torch.ones(1, 4, 2, 2)
    t = torch.tensor([0.5])
    context = torch.ones(1, 77, 8)

    monkeypatch.setattr(qnn_backend, "qnn_unet_runtime_enabled", lambda: True)
    monkeypatch.setattr(qnn_backend, "_supports_standard_unet_call", lambda *args, **kwargs: True)
    monkeypatch.setattr(qnn_backend, "_get_backend", lambda *args, **kwargs: fake_backend)

    out = qnn_backend.try_apply_qnn_backend(model, xc, t, context)

    assert fake_backend.calls == 1
    assert torch.allclose(out, xc + 2)


def test_try_apply_qnn_backend_skips_unsupported_calls(monkeypatch):
    model = SimpleNamespace()
    xc = torch.ones(1, 4, 2, 2)
    t = torch.tensor([0.5])
    context = torch.ones(1, 77, 8)

    monkeypatch.setattr(qnn_backend, "qnn_unet_runtime_enabled", lambda: True)

    assert qnn_backend.try_apply_qnn_backend(model, xc, t, context, control=object()) is None
    assert qnn_backend.try_apply_qnn_backend(model, xc, t, context, c_concat=xc) is None
    assert qnn_backend.try_apply_qnn_backend(
        SimpleNamespace(model_type=SimpleNamespace(name="FLUX")),
        xc,
        t,
        context,
    ) is None
