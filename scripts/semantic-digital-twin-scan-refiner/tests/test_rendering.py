"""
Check compact display buffers without changing stored scientific evidence.
"""

from __future__ import annotations

# %% Imports
import numpy as np

from scan_refiner.rendering import DisplayCoordinates
from scan_refiner.inspection import InspectionViewer
from scan_refiner.project import RefinementProject


# %% Display precision
def test_viewer_starts_with_free_camera_rotation(
    small_project: RefinementProject,
) -> None:
    """
    Dragging a newly opened scene must orbit instead of depending on retained modes.
    """
    scene = InspectionViewer(small_project).build_scene()
    assert scene.figure.layout.scene.dragmode == "orbit"


def test_local_float32_display_preserves_small_features_at_large_coordinates() -> None:
    """
    An explicit local origin prevents float32 rendering from erasing millimetre detail.
    """
    points = np.array(((1e6, -2e6, 100), (1e6 + 0.001, -2e6 + 0.002, 100.003)))
    original = points.copy()
    coordinates = DisplayCoordinates.from_points(points)
    assert coordinates.points.dtype == np.float32
    np.testing.assert_allclose(
        coordinates.points.astype(np.float64) + coordinates.origin, original, atol=1e-9
    )
    np.testing.assert_array_equal(points, original)


def test_compact_scene_shares_one_scan_trace(small_project: RefinementProject) -> None:
    """
    Browser geometry uses compact local buffers without duplicating scan positions.
    """
    original = small_project.scan.points.copy()
    scene = InspectionViewer(small_project).build_scene()
    scan_traces = [trace for trace in scene.figure.data if trace.type == "scatter3d"]
    assert len(scan_traces) == 1
    assert scan_traces[0].x.dtype == np.float32
    for trace in scene.figure.data:
        if trace.type == "mesh3d":
            assert trace.customdata is None
            assert trace.opacity == 1.0
    np.testing.assert_array_equal(small_project.scan.points, original)
