from __future__ import annotations

from typing_extensions import override

import torch

from comfy_api.latest import ComfyExtension, io
from comfy import qnn_runtime


_BACKEND_OPTIONS = ["auto", "htp", "gpu", "cpu"]
_IMAGE_LAYOUT_OPTIONS = ["auto", "nchw", "nhwc"]


class SnapdragonRuntimeInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SnapdragonRuntimeInfo",
            display_name="Snapdragon Runtime Info",
            category="snapdragon/qnn",
            is_experimental=True,
            search_aliases=["Snapdragon", "QNN", "NPU", "ONNX Runtime"],
            inputs=[
                io.Combo.Input("backend", options=_BACKEND_OPTIONS, default="auto"),
            ],
            outputs=[
                io.Boolean.Output(display_name="available"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, backend: str) -> io.NodeOutput:
        status = qnn_runtime.get_status(backend)
        available = bool(status["onnxruntime_available"] and status["qnn_provider_available"])
        return io.NodeOutput(available, qnn_runtime.describe_status(backend))


class QNNLoadSession(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QNNLoadSession",
            display_name="Load QNN Session",
            category="snapdragon/qnn",
            is_experimental=True,
            search_aliases=["QNN session", "Snapdragon NPU", "ONNX Runtime"],
            inputs=[
                io.String.Input("model_path", display_name="Model Path", tooltip="Path to an ONNX model file."),
                io.Combo.Input("backend", options=_BACKEND_OPTIONS, default="auto"),
                io.String.Input("backend_path", display_name="Backend DLL Path", optional=True, default=""),
                io.Boolean.Input("allow_cpu_fallback", display_name="Allow CPU Fallback", default=True),
                io.Boolean.Input("strict_backend", display_name="Strict Backend", default=False, advanced=True),
            ],
            outputs=[
                io.String.Output(display_name="session_id"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_path: str,
        backend: str,
        allow_cpu_fallback: bool,
        strict_backend: bool,
        backend_path: str = "",
    ) -> io.NodeOutput:
        handle = qnn_runtime.build_session_handle(
            model_path=model_path,
            backend=backend,
            backend_path=backend_path or None,
            allow_cpu_fallback=allow_cpu_fallback,
            strict_backend=strict_backend,
        )
        session_id = qnn_runtime.register_session(handle)
        return io.NodeOutput(session_id, f"{qnn_runtime.session_summary(handle)}; {qnn_runtime.describe_status(backend)}")


class QNNRunTensor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QNNRunTensor",
            display_name="Run QNN Tensor",
            category="snapdragon/qnn",
            is_experimental=True,
            search_aliases=["QNN", "ONNX Runtime", "tensor inference"],
            inputs=[
                io.String.Input("session_id", display_name="Session ID"),
                io.String.Input("input_name", display_name="Input Name", default="input"),
                io.String.Input("output_name", display_name="Output Name", optional=True, default=""),
                io.MultiType.Input("input_tensor", [io.AnyType], display_name="Input Tensor"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        session_id: str,
        input_name: str,
        input_tensor,
        output_name: str = "",
    ) -> io.NodeOutput:
        output = qnn_runtime.run_session(
            session_id,
            {input_name: input_tensor},
            output_name=output_name or None,
        )
        return io.NodeOutput(output, qnn_runtime.describe_session(session_id))


class QNNRunImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QNNRunImage",
            display_name="Run QNN Image",
            category="snapdragon/qnn",
            is_experimental=True,
            search_aliases=["QNN", "Snapdragon NPU", "image inference"],
            inputs=[
                io.String.Input("session_id", display_name="Session ID"),
                io.String.Input("input_name", display_name="Input Name", default="input"),
                io.String.Input("output_name", display_name="Output Name", optional=True, default=""),
                io.Combo.Input("input_layout", options=_IMAGE_LAYOUT_OPTIONS, default="auto"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        session_id: str,
        input_name: str,
        image: torch.Tensor,
        input_layout: str,
        output_name: str = "",
    ) -> io.NodeOutput:
        layout = input_layout if input_layout != "auto" else "nchw"
        output = qnn_runtime.run_image_session(
            session_id,
            input_name,
            image,
            output_name=output_name or None,
            input_layout=layout,
        )
        return io.NodeOutput(output, qnn_runtime.describe_session(session_id))


class QNNCloseSession(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QNNCloseSession",
            display_name="Close QNN Session",
            category="snapdragon/qnn",
            is_experimental=True,
            inputs=[
                io.String.Input("session_id", display_name="Session ID"),
            ],
            outputs=[
                io.Boolean.Output(display_name="closed"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, session_id: str) -> io.NodeOutput:
        closed = qnn_runtime.close_session(session_id)
        return io.NodeOutput(closed, f"Closed session {session_id}: {closed}")


class SnapdragonQnnExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SnapdragonRuntimeInfo,
            QNNLoadSession,
            QNNRunTensor,
            QNNRunImage,
            QNNCloseSession,
        ]


async def comfy_entrypoint() -> SnapdragonQnnExtension:
    return SnapdragonQnnExtension()
