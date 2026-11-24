"""
Scene selections wait until Plotly has completed the mouse gesture.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess


# %% Click and orbit separation
def test_drag_does_not_select_or_measure_before_or_after_mouse_release(
    javascript_runtime: Path,
) -> None:
    """
    Only a completed click can request a selection redraw or measurement endpoint.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/scene_clicks.cjs"),
            str(root / "scan_refiner/web/plot_buffers.js"),
        ],
        check=True,
    )
