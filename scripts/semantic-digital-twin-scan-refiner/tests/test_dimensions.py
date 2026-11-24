"""
Box dimension proposals preserve anchors and require the relevant observed faces.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
import trimesh
from scan_refiner.dimensions import (
    BoxBounds,
    DimensionAnchor,
    DimensionConstraints,
    DimensionFitter,
)
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.refinement import FitOutcome
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh, WorldSnapshot
from scan_refiner.dimensions import BodyDimensionInfo


# %% Box evidence
@pytest.fixture
def rough_box() -> BodyMesh:
    """
    The root frame declares the axis directions used by dimension locks.
    """
    return BodyMesh(
        body_name="test/box",
        role=GeometryRole.VISUAL,
        mesh=trimesh.creation.box(extents=[1, 0.6, 0.4]),
    )


@pytest.fixture
def wider_box_returns() -> np.ndarray:
    """
    Opposing faces provide enough interior evidence to recover an increased X extent.
    """
    target = trimesh.creation.box(extents=[1.04, 0.6, 0.4])
    points, _ = trimesh.sample.sample_surface(target, 6000, seed=23)
    return points


# %% Dimension locks and anchors
def test_centered_width_fit_keeps_center_and_other_dimensions(
    rough_box: BodyMesh, wider_box_returns: np.ndarray
) -> None:
    """
    A width-only change cannot alter the center, depth or height.
    """
    result = DimensionFitter().fit(
        body=rough_box,
        scan_points=wider_box_returns,
        scan_T_root=np.eye(4),
        constraints=DimensionConstraints(axes=[True, False, False]),
        tolerance=0.01,
    )
    assert result.proposal.outcome == FitOutcome.PROPOSED
    np.testing.assert_allclose(
        result.proposal.after_box.dimensions, [1.04, 0.6, 0.4], atol=1e-8
    )
    np.testing.assert_array_equal(
        result.proposal.after_box.center, result.proposal.before_box.center
    )
    assert result.proposal.scale[1:] == [1, 1]
    assert result.proposal.offset == [0, 0, 0]


def test_minimum_anchor_preserves_the_original_minimum_face(
    rough_box: BodyMesh,
) -> None:
    """
    One observed opposite face can constrain size when the user fixes its minimum face.
    """
    y, z = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.1, 0.1, 20))
    points = np.column_stack((np.full(y.size, 0.54), y.ravel(), z.ravel()))
    constraints = DimensionConstraints(
        axes=[True, False, False],
        anchors=[
            DimensionAnchor.MINIMUM.value,
            DimensionAnchor.CENTER.value,
            DimensionAnchor.CENTER.value,
        ],
    )
    result = DimensionFitter().fit(
        body=rough_box,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.01,
    )
    assert result.proposal.applicable
    assert result.proposal.after_box.lower[0] == result.proposal.before_box.lower[0]
    np.testing.assert_allclose(result.proposal.after_box.upper[0], 0.54, atol=1e-10)


def test_centered_dimension_rejects_single_sided_evidence(rough_box: BodyMesh) -> None:
    """
    Missing opposing returns do not authorize symmetric growth of an unseen face.
    """
    y, z = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.1, 0.1, 20))
    points = np.column_stack((np.full(y.size, 0.52), y.ravel(), z.ravel()))
    result = DimensionFitter().fit(
        body=rough_box,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=DimensionConstraints(axes=[True, False, False]),
        tolerance=0.01,
    )
    assert result.proposal.outcome == FitOutcome.UNDERCONSTRAINED
    assert not result.proposal.applicable


def test_dimension_limit_produces_rejected_bounded_preview(
    rough_box: BodyMesh, wider_box_returns: np.ndarray
) -> None:
    """
    A larger measured discrepancy is not silently accepted at a smaller movement cap.
    """
    result = DimensionFitter().fit(
        body=rough_box,
        scan_points=wider_box_returns,
        scan_T_root=np.eye(4),
        constraints=DimensionConstraints(
            axes=[True, False, False], maximum_change=0.01
        ),
        tolerance=0.01,
    )
    assert result.proposal.outcome == FitOutcome.AT_BOUND
    assert not result.proposal.applicable
    np.testing.assert_allclose(result.proposal.after_box.dimensions, [1.01, 0.6, 0.4])


# %% Eligibility
def test_rotated_box_is_not_treated_as_its_root_aligned_bounding_box(
    rough_box: BodyMesh,
) -> None:
    """
    The current fitter must not replace oriented geometry with a larger axis-aligned
    box.
    """
    rotated = rough_box.mesh.copy()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(0.3, [0, 0, 1]))
    with pytest.raises(InvalidEvidence):
        BoxBounds.from_mesh(mesh=rotated)


def test_nonbox_geometry_is_rejected() -> None:
    """
    A cylinder's bounding box is not evidence that its actual surface is a box.
    """
    with pytest.raises(InvalidEvidence):
        BoxBounds.from_mesh(mesh=trimesh.creation.cylinder(radius=0.3, height=0.6))


# %% Opposite anchor and placement
def test_maximum_anchor_preserves_the_original_maximum_face(
    rough_box: BodyMesh,
) -> None:
    """
    Fixing the maximum face grows only toward the measured minimum face.
    """
    y, z = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.1, 0.1, 20))
    points = np.column_stack((np.full(y.size, -0.54), y.ravel(), z.ravel()))
    constraints = DimensionConstraints(
        axes=[True, False, False],
        anchors=[
            DimensionAnchor.MAXIMUM.value,
            DimensionAnchor.CENTER.value,
            DimensionAnchor.CENTER.value,
        ],
    )
    result = DimensionFitter().fit(
        body=rough_box,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.01,
    )
    assert result.proposal.applicable
    assert result.proposal.after_box.upper[0] == result.proposal.before_box.upper[0]
    np.testing.assert_allclose(result.proposal.after_box.lower[0], -0.54, atol=1e-10)


def test_dimension_fit_uses_root_axes_under_rotated_scan_registration(
    rough_box: BodyMesh, wider_box_returns: np.ndarray
) -> None:
    """
    Scanner yaw and translation must not become inferred box orientation or dimensions.
    """
    transform = trimesh.transformations.rotation_matrix(0.7, [0, 0, 1])
    transform[:3, 3] = [2, -3, 1]
    result = DimensionFitter().fit(
        body=rough_box,
        scan_points=trimesh.transform_points(wider_box_returns, transform),
        scan_T_root=transform,
        constraints=DimensionConstraints(axes=[True, False, False]),
        tolerance=0.01,
    )
    assert result.proposal.applicable
    np.testing.assert_allclose(
        result.proposal.after_box.dimensions, [1.04, 0.6, 0.4], atol=1e-8
    )


# %% Positive extent floor
def test_dimension_fit_cannot_collapse_geometry_below_positive_floor() -> None:
    """
    A measured smaller extent remains a rejected bounded preview rather than a collapse.
    """
    rough = BodyMesh(
        body_name="thin",
        role=GeometryRole.VISUAL,
        mesh=trimesh.creation.box(extents=[0.004, 0.6, 0.4]),
    )
    y, z = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.1, 0.1, 20))
    points = np.vstack(
        [
            np.column_stack((np.full(y.size, value), y.ravel(), z.ravel()))
            for value in [-0.00025, 0.00025]
        ]
    )
    constraints = DimensionConstraints(axes=[True, False, False])
    result = DimensionFitter().fit(
        body=rough,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.005,
    )
    assert result.proposal.outcome == FitOutcome.AT_BOUND
    assert result.proposal.after_box.dimensions[0] == constraints.minimum_dimension
    assert all(scale > 0 for scale in result.proposal.scale)


# %% Whole-role eligibility
def test_compound_parts_of_one_body_are_checked_together(rough_box: BodyMesh) -> None:
    """
    Several box parts must not be accepted as one box by inspecting only the last part.
    """
    other = rough_box.mesh.copy()
    other.apply_translation([2, 0, 0])
    snapshot = WorldSnapshot(
        meshes=[
            rough_box,
            BodyMesh(body_name=rough_box.body_name, role=rough_box.role, mesh=other),
        ]
    )
    info = BodyDimensionInfo.inspect(
        snapshot=snapshot, body_name=rough_box.body_name, role="visual"
    )
    assert info.bounds is None
