"""Experimental Qualcomm Snapdragon QNN helpers.

This module keeps the ONNX Runtime QNN integration isolated from the core
ComfyUI torch runtime so the Snapdragon path can evolve without destabilizing
the existing DirectML / CPU code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import platform
from threading import Lock
from typing import Any
import uuid

import numpy as np
import torch

from comfy.windows_arm import (
    is_windows_arm64_process,
    is_windows_on_arm64_host,
    is_windows_x64_emulated_on_arm64,
    preferred_windows_arm_runtime,
)

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - depends on host package state
    ort = None
    onnxruntime_import_error = exc
else:
    onnxruntime_import_error = None

try:
    import onnxruntime_qnn as ort_qnn
except Exception as exc:  # pragma: no cover - depends on host package state
    ort_qnn = None
    onnxruntime_qnn_import_error = exc
else:
    onnxruntime_qnn_import_error = None

QNN_PROVIDER_NAME = "QNNExecutionProvider"
CPU_PROVIDER_NAME = "CPUExecutionProvider"

if ort is not None and ort_qnn is not None:
    try:
        ort.register_execution_provider_library(QNN_PROVIDER_NAME, ort_qnn.get_library_path())
    except Exception as exc:  # pragma: no cover - depends on host provider state
        logging.debug("Unable to register QNN provider library: %s", exc)

_SESSION_LOCK = Lock()
_SESSIONS: dict[str, "QnnSessionHandle"] = {}


@dataclass(slots=True)
class QnnSessionHandle:
    session_id: str
    session: Any
    model_path: str
    backend: str
    providers: list[str]
    provider_options: list[dict[str, Any]]
    input_names: list[str]
    output_names: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fallback_used: bool = False


def get_runtime_hint() -> str:
    runtime = preferred_windows_arm_runtime()
    if runtime is not None:
        return runtime

    if not is_windows_on_arm64_host():
        return "non-windows-arm"

    if is_windows_x64_emulated_on_arm64():
        return "directml"
    if is_windows_arm64_process():
        return "qnn"
    return "directml"


def get_available_providers() -> list[str]:
    if ort is None:
        return []

    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def is_qnn_provider_available() -> bool:
    return QNN_PROVIDER_NAME in get_available_providers()


def get_status(backend: str | None = None) -> dict[str, Any]:
    available_providers = get_available_providers()
    qnn_available = QNN_PROVIDER_NAME in available_providers
    return {
        "host_arm64": is_windows_on_arm64_host(),
        "process_arm64": is_windows_arm64_process(),
        "runtime_hint": get_runtime_hint(),
        "python_machine": platform.machine().upper(),
        "python_version": platform.python_version(),
        "onnxruntime_available": ort is not None,
        "onnxruntime_import_error": None if onnxruntime_import_error is None else str(onnxruntime_import_error),
        "onnxruntime_qnn_available": ort_qnn is not None,
        "onnxruntime_qnn_import_error": None if onnxruntime_qnn_import_error is None else str(onnxruntime_qnn_import_error),
        "available_providers": available_providers,
        "qnn_provider_available": qnn_available,
        "backend_requested": backend or "auto",
    }


def describe_status(backend: str | None = None) -> str:
    status = get_status(backend)
    host = "ARM64" if status["host_arm64"] else "non-ARM"
    process = "ARM64" if status["process_arm64"] else status["python_machine"]
    providers = ",".join(status["available_providers"]) if status["available_providers"] else "none"
    return (
        f"host={host} process={process} runtime={status['runtime_hint']} "
        f"backend={status['backend_requested']} providers={providers} "
        f"onnxruntime={'available' if status['onnxruntime_available'] else 'missing'}"
    )


def _require_ort():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not available in this environment. "
            "Install the Snapdragon QNN requirements and try again."
        ) from onnxruntime_import_error
    return ort


def _normalize_backend(backend: str | None) -> str:
    candidate = (backend or "auto").strip().lower()
    if candidate in {"", "auto"}:
        return "auto"
    if candidate in {"htp", "gpu", "cpu"}:
        return candidate
    if candidate in {"qnn", "npu"}:
        return "htp"
    raise ValueError(f"Unsupported QNN backend '{backend}'.")


def _resolve_provider_configuration(
    requested_backend: str,
    backend_path: str | None = None,
    allow_cpu_fallback: bool = True,
) -> tuple[list[str], list[dict[str, Any]], str, bool]:
    available_providers = get_available_providers()
    qnn_available = QNN_PROVIDER_NAME in available_providers

    if requested_backend == "cpu":
        return [CPU_PROVIDER_NAME], [{}], "cpu", False

    if requested_backend == "auto":
        if qnn_available:
            requested_backend = "htp"
        elif allow_cpu_fallback:
            return [CPU_PROVIDER_NAME], [{}], "cpu", True
        else:
            raise RuntimeError(
                "QNNExecutionProvider is not available. "
                "Install onnxruntime-qnn on native ARM64 Python 3.11 and try again."
            )

    if not qnn_available:
        if allow_cpu_fallback:
            logging.warning(
                "QNNExecutionProvider is not available; falling back to CPU provider. "
                "Status: %s",
                describe_status(requested_backend),
            )
            return [CPU_PROVIDER_NAME], [{}], "cpu", True
        raise RuntimeError(
            "QNNExecutionProvider is not available in this Python environment."
        )

    provider_options: dict[str, Any] = {"backend_type": requested_backend}
    if backend_path:
        provider_options = {"backend_path": backend_path}

    providers = [QNN_PROVIDER_NAME, CPU_PROVIDER_NAME]
    options = [provider_options, {}]
    return providers, options, requested_backend, False


def build_session_handle(
    model_path: str,
    backend: str = "auto",
    backend_path: str | None = None,
    allow_cpu_fallback: bool = True,
    strict_backend: bool = False,
) -> QnnSessionHandle:
    ort_module = _require_ort()
    normalized_backend = _normalize_backend(backend)
    model_path = os.path.abspath(os.path.expanduser(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    if strict_backend:
        allow_cpu_fallback = False

    providers, provider_options, resolved_backend, fallback_used = _resolve_provider_configuration(
        normalized_backend,
        backend_path=backend_path,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    session_options = ort_module.SessionOptions()
    session_options.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL

    if strict_backend and providers and providers[0] == QNN_PROVIDER_NAME:
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = ort_module.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers,
        provider_options=provider_options,
    )

    input_names = [node.name for node in session.get_inputs()]
    output_names = [node.name for node in session.get_outputs()]
    return QnnSessionHandle(
        session_id=uuid.uuid4().hex,
        session=session,
        model_path=model_path,
        backend=resolved_backend,
        providers=list(providers),
        provider_options=list(provider_options),
        input_names=input_names,
        output_names=output_names,
        fallback_used=fallback_used,
    )


def register_session(handle: QnnSessionHandle) -> str:
    with _SESSION_LOCK:
        _SESSIONS[handle.session_id] = handle
    return handle.session_id


def unregister_session(session_id: str) -> bool:
    with _SESSION_LOCK:
        handle = _SESSIONS.pop(session_id, None)

    if handle is None:
        return False

    try:
        del handle.session
    except Exception:
        pass
    return True


def get_session(session_id: str) -> QnnSessionHandle:
    with _SESSION_LOCK:
        handle = _SESSIONS.get(session_id)
    if handle is None:
        raise KeyError(session_id)
    return handle


def list_sessions() -> list[QnnSessionHandle]:
    with _SESSION_LOCK:
        return list(_SESSIONS.values())


def session_summary(handle: QnnSessionHandle) -> str:
    fallback_note = ", cpu fallback used" if handle.fallback_used else ""
    return (
        f"model={handle.model_path}; backend={handle.backend}; "
        f"providers={handle.providers}; inputs={handle.input_names}; outputs={handle.output_names}"
        f"{fallback_note}"
    )


def describe_session(session_id: str) -> str:
    return session_summary(get_session(session_id))


def close_session(session_id: str) -> bool:
    return unregister_session(session_id)


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            return value
    return value


def _to_torch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            return value
        return torch.from_numpy(np.ascontiguousarray(value))
    try:
        return torch.as_tensor(value)
    except Exception:
        return value


def _ensure_image_batch_layout(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(-1)
    elif tensor.ndim == 3:
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.movedim(0, -1)
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.movedim(1, -1)
    return tensor


def image_to_numpy(image: torch.Tensor, layout: str = "nchw") -> np.ndarray:
    tensor = image.detach().cpu().contiguous()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    layout_key = (layout or "nchw").strip().lower()
    if layout_key == "nhwc":
        if tensor.ndim == 4 and tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.movedim(1, -1)
    else:
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.movedim(-1, 1)

    return np.ascontiguousarray(tensor.numpy())


def numpy_to_image(value: Any) -> torch.Tensor:
    tensor = _to_torch(value)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(np.asarray(tensor))

    tensor = tensor.detach().cpu().contiguous().to(dtype=torch.float32)
    tensor = _ensure_image_batch_layout(tensor)
    return tensor.clamp(0.0, 1.0).contiguous()


def run_session(session_id: str, inputs: dict[str, Any], output_name: str | None = None) -> Any:
    handle = get_session(session_id)
    feed = {name: _to_numpy(value) for name, value in inputs.items()}
    output_names = [output_name] if output_name else None
    outputs = handle.session.run(output_names, feed)

    if output_name is not None:
        if len(outputs) == 0:
            raise RuntimeError("QNN session returned no outputs.")
        return _to_torch(outputs[0])

    converted = [_to_torch(value) for value in outputs]
    if len(converted) == 1:
        return converted[0]
    return converted


def run_image_session(
    session_id: str,
    input_name: str,
    image: torch.Tensor,
    output_name: str | None = None,
    input_layout: str = "nchw",
) -> torch.Tensor:
    handle = get_session(session_id)
    feed = {input_name: image_to_numpy(image, layout=input_layout)}
    output = handle.session.run([output_name] if output_name else None, feed)
    if len(output) == 0:
        raise RuntimeError("QNN image session returned no outputs.")
    return numpy_to_image(output[0])
