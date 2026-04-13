from __future__ import annotations

import numpy as np
import torch

import comfy.qnn_runtime as qnn_runtime


class FakeSession:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def run(self, output_names, feed):
        self.calls.append((output_names, feed))
        return self.outputs


def test_get_status_reports_missing_onnxruntime(monkeypatch):
    monkeypatch.setattr(qnn_runtime, "ort", None)
    monkeypatch.setattr(qnn_runtime, "onnxruntime_import_error", RuntimeError("missing"))
    monkeypatch.setattr(qnn_runtime, "get_available_providers", lambda: [])
    monkeypatch.setattr(qnn_runtime, "is_windows_on_arm64_host", lambda: True)
    monkeypatch.setattr(qnn_runtime, "is_windows_arm64_process", lambda: True)
    monkeypatch.setattr(qnn_runtime, "preferred_windows_arm_runtime", lambda: "qnn")

    status = qnn_runtime.get_status("auto")

    assert status["onnxruntime_available"] is False
    assert status["qnn_provider_available"] is False
    assert status["runtime_hint"] == "qnn"
    assert "missing" in status["onnxruntime_import_error"]


def test_session_registry_runs_and_converts_tensors(monkeypatch):
    fake_session = FakeSession([np.array([[1.0, 2.0]], dtype=np.float32)])
    handle = qnn_runtime.QnnSessionHandle(
        session_id="session-1",
        session=fake_session,
        model_path="model.onnx",
        backend="htp",
        providers=["QNNExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"backend_type": "htp"}, {}],
        input_names=["input"],
        output_names=["output"],
    )

    qnn_runtime.register_session(handle)
    try:
        output = qnn_runtime.run_session("session-1", {"input": torch.ones((1, 2), dtype=torch.float32)})
    finally:
        qnn_runtime.close_session("session-1")

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2)
    assert torch.allclose(output, torch.tensor([[1.0, 2.0]]))
    assert fake_session.calls[0][0] is None
    assert isinstance(fake_session.calls[0][1]["input"], np.ndarray)


def test_image_round_trip_keeps_batch_layout():
    image = torch.rand(1, 8, 4, 3, dtype=torch.float32)

    np_image = qnn_runtime.image_to_numpy(image, layout="nhwc")
    round_trip = qnn_runtime.numpy_to_image(np_image)

    assert np_image.shape == (1, 8, 4, 3)
    assert round_trip.shape == (1, 8, 4, 3)
    assert torch.all(round_trip >= 0.0)
    assert torch.all(round_trip <= 1.0)
