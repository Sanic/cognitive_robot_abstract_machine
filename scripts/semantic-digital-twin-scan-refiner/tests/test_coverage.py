"""
Distinguish scan residuals from unsupported, cropped and unsampled surfaces.
"""

from __future__ import annotations

# %% Imports
import subprocess
from pathlib import Path
import numpy as np
import pytest
import trimesh
from scan_refiner.coverage import EvidenceComparison
from scan_refiner.frames import CoordinateFrame, OrientedRegion
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh


# %% Labelled separated surfaces
@pytest.fixture
def separated_surfaces() -> list[BodyMesh]:
    """
    Two coplanar square patches have unambiguous nearest-body assignments.
    """
    first = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    second = first.copy()
    second.apply_translation([3, 0, 0])
    return [
        BodyMesh(body_name="near", role=GeometryRole.VISUAL, mesh=first),
        BodyMesh(body_name="unobserved", role=GeometryRole.VISUAL, mesh=second),
    ]


# %% Directional evidence
def test_absent_returns_are_unsupported_without_scan_error(
    separated_surfaces: list[BodyMesh],
) -> None:
    """
    An unscanned patch has no assigned residuals and zero sampled support.
    """
    axis = np.linspace(0, 1, 31)
    x, y = np.meshgrid(axis, axis)
    scan = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    comparison = EvidenceComparison(model_sample_count=1000)
    result = comparison.compare(
        scan_points=scan,
        surfaces=separated_surfaces,
        scan_T_root=np.eye(4),
        tolerance=0.03,
    )
    near, unknown = result.report.bodies
    assert near.scan.sample_count == len(scan)
    assert near.scan.median_distance == 0
    assert near.model.within_tolerance == 1
    assert unknown.scan.sample_count == 0
    assert unknown.scan.median_distance is None
    assert unknown.model.within_tolerance == 0
    assert result.report.supported_area_fraction == 0.5
    assert result.report.model_sample_count == comparison.model_sample_count


def test_crop_excludes_outside_model_from_support_denominator(
    separated_surfaces: list[BodyMesh],
) -> None:
    """
    A box in a rotated frame cannot penalize geometry outside the evidence query.
    """
    comparison = EvidenceComparison(model_sample_count=1000)
    region = OrientedRegion(
        lower=np.array([-0.01, -0.01, -0.01]),
        upper=np.array([1.01, 1.01, 0.01]),
        frame=CoordinateFrame(),
    )
    scan = np.array([[0.5, 0.5, 0]])
    result = comparison.compare(
        scan_points=scan,
        surfaces=separated_surfaces,
        scan_T_root=np.eye(4),
        tolerance=2,
        region=region,
    )
    assert result.report.supported_area_fraction == 1
    assert result.report.bodies[1].model.sample_count == 0
    assert result.report.bodies[1].model.within_tolerance is None
    assert result.report.model_sample_count == 500
    assert result.report.bodies[0].evaluated_area == 1
    assert result.report.bodies[1].evaluated_area == 0


def test_labels_distances_and_support_follow_rigid_placement(
    separated_surfaces: list[BodyMesh],
) -> None:
    """
    All distances are invariant when scan and model share a rigid placement.
    """
    transform = trimesh.transformations.rotation_matrix(0.7, [0, 0, 1])
    transform[:3, 3] = [10, -4, 2]
    root = np.array([[0.2, 0.3, 0.01], [3.7, 0.5, 0.04]])
    comparison = EvidenceComparison(model_sample_count=100)
    expected = comparison.compare(
        scan_points=root,
        surfaces=separated_surfaces,
        scan_T_root=np.eye(4),
        tolerance=0.02,
    )
    actual = comparison.compare(
        scan_points=trimesh.transform_points(root, transform),
        surfaces=separated_surfaces,
        scan_T_root=transform,
        tolerance=0.02,
    )
    np.testing.assert_allclose(
        actual.scan_distances, expected.scan_distances, atol=1e-12
    )
    np.testing.assert_array_equal(actual.scan_body_indices, [0, 1])
    np.testing.assert_allclose(
        actual.model_distances, expected.model_distances, atol=1e-12
    )
    assert actual.report.bodies[0].scan.within_tolerance == 1
    assert actual.report.bodies[1].scan.within_tolerance == 0


def test_surface_samples_are_bounded_and_reproducible(
    separated_surfaces: list[BodyMesh],
) -> None:
    """
    Sampling uses a fixed total budget with original qualified body identities.
    """
    comparison = EvidenceComparison(model_sample_count=101)
    scan = np.array([[0.5, 0.5, 0]])
    first = comparison.compare(
        scan_points=scan,
        surfaces=separated_surfaces,
        scan_T_root=np.eye(4),
        tolerance=0.02,
    )
    second = comparison.compare(
        scan_points=scan,
        surfaces=separated_surfaces,
        scan_T_root=np.eye(4),
        tolerance=0.02,
    )
    np.testing.assert_array_equal(first.model_points, second.model_points)
    assert len(first.model_points) == comparison.model_sample_count
    assert [body.body_name for body in first.report.bodies] == [
        body.body_name for body in separated_surfaces
    ]


# %% Evidence presentation
def test_results_distinguish_absent_and_cropped_evidence(
    javascript_runtime: str,
) -> None:
    """
    Unknown residuals and unevaluated model regions must never display as zero error.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/comparison_report.cjs"),
            str(root / "scan_refiner/web/comparison_report.js"),
        ],
        check=True,
    )


# %% Empty reverse sample
def test_crop_without_model_samples_has_unknown_area_support(
    separated_surfaces: list[BodyMesh],
) -> None:
    """
    A region with evidence but no model samples cannot report zero or perfect support.
    """
    region = OrientedRegion(
        lower=np.array([10, 10, -1]),
        upper=np.array([11, 11, 1]),
        frame=CoordinateFrame(),
    )
    result = EvidenceComparison(model_sample_count=20).compare(
        scan_points=np.array([[10.5, 10.5, 0]]),
        surfaces=separated_surfaces,
        scan_T_root=np.eye(4),
        tolerance=0.02,
        region=region,
    )
    assert result.report.model_sample_count == 0
    assert result.report.supported_area_fraction is None
    assert all(body.model.within_tolerance is None for body in result.report.bodies)
