"""
Check that distance colours communicate agreement intuitively.
"""

from __future__ import annotations

# %% Imports
import re
import subprocess
from pathlib import Path

from scan_refiner.inspection import InspectionViewer
from scan_refiner.project import RefinementProject
from scan_refiner.records import RegistrationCandidate, RegistrationResult


# %% Palette semantics
def test_zero_distance_is_green_and_large_distance_is_red(
    small_project: RefinementProject,
) -> None:
    """
    The low-residual end is green rather than a dark blue/purple colour.
    """
    candidate = RegistrationCandidate(
        scan_T_root=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        support_fraction=1.0,
        median_distance=0.0,
    )
    small_project.update_registration(
        result=RegistrationResult(candidates=[candidate]), role="visual"
    )
    scene = InspectionViewer(project=small_project).build_scene(heatmap=True)
    scale = scene.figure.data[0].marker.colorscale
    low = [int(channel) for channel in re.findall(r"\d+", scale[0][1])]
    high = [int(channel) for channel in re.findall(r"\d+", scale[-1][1])]
    assert low[1] > low[0] and low[1] > low[2]
    assert high[0] > high[1] and high[0] > high[2]


# %% Interactive palette
def test_interactive_heatmap_reuses_explicit_preview_palette(
    javascript_runtime: str,
) -> None:
    """
    Regional results retain the preview's supported green-to-red colour stops.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/heatmap_palette.cjs"),
            str(root / "scan_refiner/web/viewer.js"),
        ],
        check=True,
    )


# %% RGB preview palette availability
def test_rgb_preview_carries_palette_for_later_regional_results(
    small_project: RefinementProject,
) -> None:
    """
    A regional job needs supported colour stops even without a preview heatmap.
    """
    scale = (
        InspectionViewer(project=small_project)
        .build_scene()
        .figure.data[0]
        .marker.colorscale
    )
    assert scale is not None
    low = [int(channel) for channel in re.findall(r"\d+", scale[0][1])]
    high = [int(channel) for channel in re.findall(r"\d+", scale[-1][1])]
    assert low[1] > low[0] and low[1] > low[2]
    assert high[0] > high[1] and high[0] > high[2]


# %% Plotly nested attribute contract
def test_palette_is_wrapped_as_one_value_for_every_trace(
    javascript_runtime: str,
) -> None:
    """
    Plotly must not interpret individual colour stops as separate trace values.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/plot_palette.cjs"),
            str(root / "scan_refiner/web/plot_buffers.js"),
        ],
        check=True,
    )
