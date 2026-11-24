"""
Prove oriented cropping preserves evidence and filters before sampling.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import numpy as np
import pytest
import trimesh
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame, OrientedRegion
from scan_refiner.point_records import PointField, PointBatch, PointRecordLayout
from scan_refiner.spatial_index import SpatialScanIndex


# %% Rigid frames
def test_rotated_box_excludes_scan_axis_bounding_box_corners() -> None:
    """
    A thin diagonal box includes its own points but excludes nearby AABB clutter.
    """
    frame_T_scan = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1])
    frame = CoordinateFrame(frame_T_scan=frame_T_scan)
    region = OrientedRegion(
        lower=np.array([-2.0, -0.1, -0.1]), upper=np.array([2.0, 0.1, 0.1]), frame=frame
    )
    inside = frame.to_scan(frame_points=np.array([[-1, 0, 0], [1, 0, 0]]))
    outside = np.array([[1, 1, 0]])
    assert region.contains(scan_points=inside).tolist() == [True, True]
    assert region.scan_bounds.contains(points=outside).tolist() == [True]
    assert region.contains(scan_points=outside).tolist() == [False]
    np.testing.assert_allclose(
        frame.to_frame(scan_points=inside), [[-1, 0, 0], [1, 0, 0]], atol=1e-12
    )


def test_frame_rejects_scale_in_a_rigid_transform() -> None:
    """
    Display orientation cannot change the measured dimensions of evidence.
    """
    matrix = np.eye(4)
    matrix[0, 0] = 2
    with pytest.raises(InvalidEvidence):
        CoordinateFrame(frame_T_scan=matrix)


# %% Original records
def test_oriented_crop_filters_before_applying_display_budget(tmp_path: Path) -> None:
    """
    Outside points do not consume the sample budget or alter original attributes.
    """
    frame = CoordinateFrame(
        frame_T_scan=trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1])
    )
    region = OrientedRegion(
        lower=np.array([-2.0, -0.1, -0.1]), upper=np.array([2.0, 0.1, 0.1]), frame=frame
    )
    records = np.zeros(108, dtype=PointRecordLayout().dtype)
    records[PointField.POINTS][:8] = frame.to_scan(
        frame_points=np.column_stack([np.linspace(-1.9, 1.9, 8), np.zeros((8, 2))])
    )
    records[PointField.POINTS][8:] = [1, 1, 0]
    records[PointField.SOURCE_INDICES] = np.arange(len(records))
    records[PointField.COLORS] = [12, 34, 56]
    index = SpatialScanIndex.build(
        directory=tmp_path / "index",
        batches=iter([PointBatch(records=records)]),
        source_identity="fixture",
        overview_count=8,
    )
    sample = region.sample(index=index, maximum_count=8)
    assert sorted(sample.records[PointField.SOURCE_INDICES].tolist()) == list(range(8))
    assert sample.records.tobytes() == records[:8].tobytes()
