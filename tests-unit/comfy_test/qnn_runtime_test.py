from __future__ import annotations

from types import SimpleNamespace
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
    assert status["qnn_npu_device_available"] is False
    assert status["runtime_hint"] == "qnn"
    assert "missing" in status["onnxruntime_import_error"]


def test_get_status_reports_qnn_npu_devices(monkeypatch):
    fake_npu_device = SimpleNamespace(
        ep_name="QNNExecutionProvider",
        device=SimpleNamespace(type=SimpleNamespace(name="NPU")),
    )
    fake_cpu_device = SimpleNamespace(
        ep_name="CPUExecutionProvider",
        device=SimpleNamespace(type=SimpleNamespace(name="CPU")),
    )

    monkeypatch.setattr(qnn_runtime, "ort", SimpleNamespace(get_ep_devices=lambda: [fake_cpu_device, fake_npu_device]))
    monkeypatch.setattr(qnn_runtime, "get_available_providers", lambda: ["AzureExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(qnn_runtime, "is_windows_on_arm64_host", lambda: True)
    monkeypatch.setattr(qnn_runtime, "is_windows_arm64_process", lambda: True)
    monkeypatch.setattr(qnn_runtime, "preferred_windows_arm_runtime", lambda: "qnn")

    status = qnn_runtime.get_status("auto")

    assert status["qnn_provider_available"] is True
    assert status["qnn_npu_device_available"] is True
    assert "NPU" in status["qnn_device_types"]


def test_build_session_handle_uses_qnn_ep_devices(monkeypatch, tmp_path):
    fake_npu_device = SimpleNamespace(
        ep_name="QNNExecutionProvider",
        device=SimpleNamespace(type=SimpleNamespace(name="NPU")),
    )
    fake_cpu_device = SimpleNamespace(
        ep_name="CPUExecutionProvider",
        device=SimpleNamespace(type=SimpleNamespace(name="CPU")),
    )

    class FakeSessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.provider_device_calls = []
            self.session_config_entries = {}

        def add_provider_for_devices(self, devices, provider_options):
            self.provider_device_calls.append((list(devices), dict(provider_options)))

        def add_session_config_entry(self, key, value):
            self.session_config_entries[key] = value

    class FakeSession:
        def __init__(self, model_path, sess_options=None):
            self.model_path = model_path
            self.sess_options = sess_options

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def get_providers(self):
            if self.sess_options and self.sess_options.provider_device_calls:
                return ["QNNExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=object()),
        get_ep_devices=lambda: [fake_cpu_device, fake_npu_device],
    )

    monkeypatch.setattr(qnn_runtime, "ort", fake_ort)
    monkeypatch.setattr(qnn_runtime, "get_available_providers", lambda: ["AzureExecutionProvider", "CPUExecutionProvider"])

    model_path = tmp_path / "model.onnx"
    model_path.write_text("fake")

    handle = qnn_runtime.build_session_handle(
        model_path=str(model_path),
        backend="auto",
        backend_path="",
        allow_cpu_fallback=True,
        strict_backend=False,
    )

    assert handle.providers == ["QNNExecutionProvider", "CPUExecutionProvider"]
    assert handle.fallback_used is False
    assert handle.provider_options == [{"backend_type": "htp"}]
    assert handle.session.sess_options.provider_device_calls[0][0][0].ep_name == "QNNExecutionProvider"
    assert handle.session.sess_options.provider_device_calls[0][1] == {"backend_type": "htp"}


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
