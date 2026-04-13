from __future__ import annotations

import comfy.memory_management as memory_management
import comfy.model_management as model_management


class OpaqueTensor:
    def __init__(self, nbytes: int = 64):
        self.nbytes = nbytes

    def untyped_storage(self):
        raise NotImplementedError("Cannot access storage of OpaqueTensorImpl")


class DummyModule:
    def __init__(self, tensor):
        self._tensor = tensor

    def state_dict(self):
        return {"weight": self._tensor}


def test_get_untyped_storage_returns_none_for_opaque_tensor():
    tensor = OpaqueTensor()

    assert memory_management.get_untyped_storage(tensor) is None


def test_read_tensor_file_slice_into_returns_false_for_opaque_tensor():
    tensor = OpaqueTensor()

    assert memory_management.read_tensor_file_slice_into(tensor, object()) is False


def test_module_mmap_residency_skips_opaque_tensor_storage():
    module = DummyModule(OpaqueTensor(nbytes=128))

    mmap_touched_mem, module_mem = model_management.module_mmap_residency(module)

    assert mmap_touched_mem == 0
    assert module_mem == 128
