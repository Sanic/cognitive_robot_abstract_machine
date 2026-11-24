"""
World overview reuses frozen evidence instead of reading the dense index.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess


# %% Cached evidence
def test_overview_restores_preview_without_dense_region_query(
    javascript_runtime: Path,
) -> None:
    """
    Clearing regional state also prevents a global index query after browser reload.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/overview.cjs"),
            str(root / "scan_refiner/web/viewer.js"),
            str(root / "scan_refiner/web/coordinate_frame.js"),
        ],
        check=True,
    )
