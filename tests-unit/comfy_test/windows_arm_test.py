import comfy.windows_arm as windows_arm


class TestWindowsArmDetection:
    def test_windows_arm64_host_with_x64_emulation(self, monkeypatch):
        monkeypatch.setattr(windows_arm, "is_windows", lambda: True)
        monkeypatch.setattr(
            windows_arm,
            "_get_windows_machine_type_codes",
            lambda: (
                windows_arm.IMAGE_FILE_MACHINE_AMD64,
                windows_arm.IMAGE_FILE_MACHINE_ARM64,
            ),
        )
        monkeypatch.delenv("COMFYUI_DISABLE_AUTO_DIRECTML", raising=False)

        assert windows_arm.is_windows_on_arm64_host() is True
        assert windows_arm.is_windows_x64_emulated_on_arm64() is True
        assert windows_arm.is_windows_arm64_process() is False
        assert windows_arm.should_auto_enable_directml() is True
        assert windows_arm.directml_uses_shared_memory() is True
        assert windows_arm.describe_windows_arm_state() == "Windows ARM64 host with x64 emulated Python"

    def test_windows_arm64_native_process(self, monkeypatch):
        monkeypatch.setattr(windows_arm, "is_windows", lambda: True)
        monkeypatch.setattr(
            windows_arm,
            "_get_windows_machine_type_codes",
            lambda: (
                windows_arm.IMAGE_FILE_MACHINE_UNKNOWN,
                windows_arm.IMAGE_FILE_MACHINE_ARM64,
            ),
        )
        monkeypatch.delenv("COMFYUI_DISABLE_AUTO_DIRECTML", raising=False)

        assert windows_arm.is_windows_on_arm64_host() is True
        assert windows_arm.is_windows_x64_emulated_on_arm64() is False
        assert windows_arm.is_windows_arm64_process() is True
        assert windows_arm.preferred_windows_arm_runtime() == "qnn"
        assert windows_arm.should_auto_enable_directml() is False
        assert windows_arm.describe_windows_arm_state() == "Windows ARM64 host with native ARM64 Python"

    def test_non_arm_windows_does_not_auto_enable_directml(self, monkeypatch):
        monkeypatch.setattr(windows_arm, "is_windows", lambda: True)
        monkeypatch.setattr(
            windows_arm,
            "_get_windows_machine_type_codes",
            lambda: (
                windows_arm.IMAGE_FILE_MACHINE_AMD64,
                windows_arm.IMAGE_FILE_MACHINE_AMD64,
            ),
        )
        monkeypatch.delenv("COMFYUI_DISABLE_AUTO_DIRECTML", raising=False)

        assert windows_arm.is_windows_on_arm64_host() is False
        assert windows_arm.is_windows_x64_emulated_on_arm64() is False
        assert windows_arm.should_auto_enable_directml() is False
        assert windows_arm.directml_uses_shared_memory() is False
        assert windows_arm.describe_windows_arm_state() == "non-Windows-ARM"

    def test_auto_directml_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(windows_arm, "is_windows", lambda: True)
        monkeypatch.setattr(
            windows_arm,
            "_get_windows_machine_type_codes",
            lambda: (
                windows_arm.IMAGE_FILE_MACHINE_UNKNOWN,
                windows_arm.IMAGE_FILE_MACHINE_ARM64,
            ),
        )
        monkeypatch.setenv("COMFYUI_DISABLE_AUTO_DIRECTML", "1")

        assert windows_arm.should_auto_enable_directml() is False

    def test_native_arm64_defaults_to_qnn_runtime(self, monkeypatch):
        monkeypatch.setattr(windows_arm, "is_windows", lambda: True)
        monkeypatch.setattr(
            windows_arm,
            "_get_windows_machine_type_codes",
            lambda: (
                windows_arm.IMAGE_FILE_MACHINE_UNKNOWN,
                windows_arm.IMAGE_FILE_MACHINE_ARM64,
            ),
        )
        monkeypatch.delenv("COMFYUI_ARM_RUNTIME", raising=False)
        monkeypatch.delenv("COMFYUI_DISABLE_AUTO_DIRECTML", raising=False)

        assert windows_arm.preferred_windows_arm_runtime() == "qnn"
        assert windows_arm.should_auto_enable_directml() is False

    def test_requested_qnn_runtime_disables_directml_autostart(self, monkeypatch):
        monkeypatch.setattr(windows_arm, "is_windows", lambda: True)
        monkeypatch.setattr(
            windows_arm,
            "_get_windows_machine_type_codes",
            lambda: (
                windows_arm.IMAGE_FILE_MACHINE_AMD64,
                windows_arm.IMAGE_FILE_MACHINE_ARM64,
            ),
        )
        monkeypatch.setenv("COMFYUI_ARM_RUNTIME", "qnn")
        monkeypatch.delenv("COMFYUI_DISABLE_AUTO_DIRECTML", raising=False)

        assert windows_arm.preferred_windows_arm_runtime() == "qnn"
        assert windows_arm.should_auto_enable_directml() is False
