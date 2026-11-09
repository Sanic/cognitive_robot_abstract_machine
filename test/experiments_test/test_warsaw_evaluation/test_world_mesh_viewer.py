"""
Check browser-side object selection without a graphics context.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# %% hierarchy and class selection


@pytest.mark.parametrize(
    "script",
    [
        "checks.mjs",
        "interface_checks.mjs",
        "comparison_checks.mjs",
        "comparison_interface_checks.mjs",
        "rendering_checks.mjs",
    ],
)
def test_world_mesh_viewer_object_relationships(script: str) -> None:
    """
    Keep class filtering distinct from the final part hierarchy.
    """
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("The browser model checks require an installed Node.js runtime.")
    checks = Path(__file__).parent / "dataset" / "world_mesh_viewer" / script
    subprocess.run(
        [
            executable,
            "--experimental-default-type=module",
            "--experimental-vm-modules",
            str(checks),
        ],
        check=True,
    )
