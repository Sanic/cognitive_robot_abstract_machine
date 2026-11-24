"""
Focus frames complete objects independently of scan crops and camera panning.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess


# %% Rotated object extents
def test_focus_contains_rotated_selected_object_and_recenters_camera(
    javascript_runtime: Path,
) -> None:
    """
    A stale crop must not truncate the object or bring an unrelated neighbor into focus.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/selection_focus.cjs"),
            str(root / "scan_refiner/web/viewer.js"),
        ],
        check=True,
    )
