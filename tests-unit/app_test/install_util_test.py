from pathlib import Path
from unittest.mock import mock_open, patch

import utils.install_util as install_util


def test_get_required_packages_versions_ignores_environment_markers():
    install_util.PACKAGE_VERSIONS = {}
    requirements = """\
torch>=2.10.0
comfyui-kitchen>=0.2.8; platform_machine != "ARM64"
# comment
-r other.txt
"""

    with (
        patch("builtins.open", mock_open(read_data=requirements)),
        patch.object(install_util, "requirements_path", Path("requirements.txt")),
    ):
        versions = install_util.get_required_packages_versions()

    assert versions["torch"] == "2.10.0"
    assert versions["comfyui-kitchen"] == "0.2.8"
