"""
Foreground view changes show feedback before blocking transformation work.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess


# %% Painted operation status
def test_foreground_change_paints_and_releases_feedback_on_failure(
    javascript_runtime: Path,
) -> None:
    """
    The axis-orientation path uses the same cleanup on success and failure.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/foreground_activity.cjs"),
            str(root / "scan_refiner/web/inspection_tools.js"),
        ],
        check=True,
    )
