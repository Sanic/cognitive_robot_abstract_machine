"""
Exercise repeated point-budget changes without requiring a graphics driver.
"""

from __future__ import annotations

# %% Imports
import subprocess
from pathlib import Path


# %% Render batches
def test_region_budget_changes_preserve_every_point(javascript_runtime: str) -> None:
    """
    Repeated 50k and 150k loads retain records in bounded render batches.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/scan_cloud.cjs"),
            str(root / "scan_refiner/web/scan_cloud.js"),
        ],
        check=True,
    )


# %% Display filtering
def test_clip_and_section_use_original_scan_coordinates(
    javascript_runtime: str,
) -> None:
    """
    Display-origin compensation applies before clipping and metric slab selection.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/inspection_selection.cjs"),
            str(root / "scan_refiner/web/inspection_tools.js"),
        ],
        check=True,
    )


# %% History controls
def test_reopened_view_restores_history_controls(javascript_runtime: str) -> None:
    """
    Undo is re-enabled after restoration, while pending renders prevent traversal.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/inspection_history.cjs"),
            str(root / "scan_refiner/web/inspection_tools.js"),
        ],
        check=True,
    )


# %% Concurrent layers
def test_scan_and_measurement_insertions_keep_distinct_indices(
    javascript_runtime: str,
) -> None:
    """
    Queued scan and annotation additions receive the indices actually inserted.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/plot_insertions.cjs"),
            str(root / "scan_refiner/web/plot_buffers.js"),
        ],
        check=True,
    )
