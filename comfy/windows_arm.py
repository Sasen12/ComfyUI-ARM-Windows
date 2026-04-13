"""Helpers for detecting Windows on ARM and Snapdragon-friendly runtime modes."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os
import platform
import sys

IMAGE_FILE_MACHINE_UNKNOWN = 0x0000
IMAGE_FILE_MACHINE_I386 = 0x014C
IMAGE_FILE_MACHINE_ARM = 0x01C0
IMAGE_FILE_MACHINE_IA64 = 0x0200
IMAGE_FILE_MACHINE_AMD64 = 0x8664
IMAGE_FILE_MACHINE_ARM64 = 0xAA64

_ARCHITECTURE_NAMES = {
    IMAGE_FILE_MACHINE_I386: "X86",
    IMAGE_FILE_MACHINE_ARM: "ARM",
    IMAGE_FILE_MACHINE_IA64: "IA64",
    IMAGE_FILE_MACHINE_AMD64: "AMD64",
    IMAGE_FILE_MACHINE_ARM64: "ARM64",
}

_ARCHITECTURE_CODES = {name: code for code, name in _ARCHITECTURE_NAMES.items()}


def is_windows() -> bool:
    return os.name == "nt"


def _normalize_machine_name(machine: str | None) -> str:
    normalized = (machine or "").replace("-", "").replace("_", "").upper()
    if normalized in {"AMD64", "X8664", "X64"}:
        return "AMD64"
    if normalized in {"ARM64", "AARCH64"}:
        return "ARM64"
    if normalized in {"ARM", "ARM32"}:
        return "ARM"
    if normalized in {"X86", "I386", "I686"}:
        return "X86"
    return normalized or "UNKNOWN"


def _machine_code_from_env(value: str | None) -> int | None:
    normalized = _normalize_machine_name(value)
    return _ARCHITECTURE_CODES.get(normalized)


def _machine_code_from_executable(path: str | None) -> int | None:
    if not path:
        return None

    try:
        with open(path, "rb") as executable_file:
            header = executable_file.read(0x1000)
    except Exception:
        return None

    if len(header) < 0x40 or header[:2] != b"MZ":
        return None

    pe_offset = int.from_bytes(header[0x3C:0x40], "little", signed=False)
    if pe_offset + 6 > len(header):
        try:
            with open(path, "rb") as executable_file:
                executable_file.seek(pe_offset)
                signature_and_machine = executable_file.read(6)
        except Exception:
            return None
    else:
        signature_and_machine = header[pe_offset : pe_offset + 6]

    if len(signature_and_machine) < 6 or signature_and_machine[:4] != b"PE\0\0":
        return None

    return int.from_bytes(signature_and_machine[4:6], "little", signed=False)


@lru_cache(maxsize=1)
def _get_windows_machine_type_codes() -> tuple[int | None, int | None]:
    """Return the process and native machine types for Windows."""
    if not is_windows():
        return None, None

    process_code = _machine_code_from_executable(sys.executable)
    if process_code is None:
        process_code = _machine_code_from_env(os.environ.get("PROCESSOR_ARCHITECTURE") or platform.machine())

    native_code = _machine_code_from_env(platform.machine())
    if native_code is None:
        native_code = process_code

    return process_code, native_code


def _get_windows_process_architecture_code() -> int | None:
    process_code, native_code = _get_windows_machine_type_codes()
    if process_code in (None, IMAGE_FILE_MACHINE_UNKNOWN):
        return native_code
    return process_code


def _get_windows_native_architecture_code() -> int | None:
    _, native_code = _get_windows_machine_type_codes()
    return native_code


def get_windows_native_architecture() -> str | None:
    architecture = _get_windows_native_architecture_code()
    if architecture is None:
        return None
    return _ARCHITECTURE_NAMES.get(architecture, f"UNKNOWN_{architecture}")


def get_windows_process_architecture() -> str | None:
    architecture = _get_windows_process_architecture_code()
    if architecture is None:
        return None
    return _ARCHITECTURE_NAMES.get(architecture, f"UNKNOWN_{architecture}")


def is_windows_on_arm64_host() -> bool:
    return get_windows_native_architecture() == "ARM64"


def is_windows_arm64_process() -> bool:
    return get_windows_process_architecture() == "ARM64"


def is_windows_x64_emulated_on_arm64() -> bool:
    return is_windows_on_arm64_host() and get_windows_process_architecture() == "AMD64"


def should_auto_enable_directml() -> bool:
    if not is_windows_on_arm64_host():
        return False

    disabled = os.environ.get("COMFYUI_DISABLE_AUTO_DIRECTML", "").strip().lower()
    return disabled not in {"1", "true", "yes", "on"}


def directml_uses_shared_memory() -> bool:
    return is_windows_on_arm64_host()


def describe_windows_arm_state() -> str:
    if not is_windows_on_arm64_host():
        return "non-Windows-ARM"

    if is_windows_x64_emulated_on_arm64():
        return "Windows ARM64 host with x64 emulated Python"

    if is_windows_arm64_process():
        return "Windows ARM64 host with native ARM64 Python"

    return "Windows ARM64 host"
