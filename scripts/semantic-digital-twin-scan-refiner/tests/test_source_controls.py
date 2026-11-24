"""
Source downloads follow the accepted geometry revision across history traversal.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess


# %% Reviewed source controls
def test_download_returns_when_history_restores_reviewed_geometry(
    javascript_runtime: Path,
) -> None:
    """
    An immutable output is current again after Redo restores its exact geometry.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/source_controls.cjs"),
            str(root / "scan_refiner/web/refinement.js"),
        ],
        check=True,
    )
