"""
Verified box frames preserve orientation while fitting dimensions in local axes.
"""

from __future__ import annotations

# %% Imports
import subprocess
import hashlib
import json
from pathlib import Path
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.project import RefinementProject
from scan_refiner.jobs import JobBaseline
from scan_refiner.records import RegistrationResult
from scan_refiner.dimensions import BodyDimensions
import numpy as np
import pytest
import trimesh
from scan_refiner.dimensions import (
    BodyDimensionInfo,
    DimensionAnchor,
    DimensionConstraints,
    DimensionFitter,
    DimensionFrame,
    OrientedBoxInspection,
)
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh, WorldSnapshot


# %% Oriented evidence
@pytest.fixture
def box_placement() -> np.ndarray:
    """
    A nontrivial fixed pose makes root-axis scaling an observable error.
    """
    transform = trimesh.transformations.euler_matrix(0.15, -0.2, 0.35)
    transform[:3, 3] = [1, -2, 0.8]
    return transform


@pytest.fixture
def oriented_box(box_placement: np.ndarray) -> BodyMesh:
    """
    The body is an actual cuboid despite its non-axis-aligned root-frame bounds.
    """
    mesh = trimesh.creation.box(extents=[1, 0.6, 0.4])
    mesh.apply_transform(box_placement)
    return BodyMesh(body_name="test/oriented-box", role=GeometryRole.VISUAL, mesh=mesh)


# %% Local-frame dimensions
def test_oriented_width_fit_preserves_center_and_box_directions(
    oriented_box: BodyMesh, box_placement: np.ndarray
) -> None:
    """
    The proposed root-space mapping stretches along the geometric box axes only.
    """
    measured = trimesh.creation.box(extents=[1.04, 0.6, 0.4])
    measured.apply_transform(box_placement)
    points, _ = trimesh.sample.sample_surface(measured, 6000, seed=23)
    constraints = DimensionConstraints(
        axes=[True, False, False], frame_mode=DimensionFrame.BOX.value
    )
    result = DimensionFitter().fit(
        body=oriented_box,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.01,
    )
    assert result.proposal.applicable
    np.testing.assert_allclose(
        result.proposal.after_box.dimensions, [1.04, 0.6, 0.4], atol=1e-8
    )
    matrix = np.array(result.proposal.matrix)
    rotation = box_placement[:3, :3]
    np.testing.assert_allclose(
        matrix @ rotation, rotation @ np.diag([1.04, 1, 1]), atol=1e-8
    )
    center = box_placement[:3, 3]
    np.testing.assert_allclose(
        matrix @ center + result.proposal.offset, center, atol=1e-8
    )
    np.testing.assert_allclose(
        oriented_box.mesh.vertices @ matrix.T + result.proposal.offset,
        measured.vertices,
        atol=1e-8,
    )


def test_oriented_minimum_anchor_holds_the_same_physical_face(
    oriented_box: BodyMesh, box_placement: np.ndarray
) -> None:
    """
    Anchors refer to a geometric box face, not a face of its larger root AABB.
    """
    measured = trimesh.creation.box(extents=[1.04, 0.6, 0.4])
    measured.apply_translation([0.02, 0, 0])
    measured.apply_transform(box_placement)
    points, _ = trimesh.sample.sample_surface(measured, 6000, seed=23)
    constraints = DimensionConstraints(
        axes=[True, False, False],
        anchors=[
            DimensionAnchor.MINIMUM.value,
            DimensionAnchor.CENTER.value,
            DimensionAnchor.CENTER.value,
        ],
        frame_mode=DimensionFrame.BOX.value,
    )
    result = DimensionFitter().fit(
        body=oriented_box,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.01,
    )
    assert result.proposal.applicable
    np.testing.assert_allclose(
        result.proposal.before_box.lower[0],
        result.proposal.after_box.lower[0],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        oriented_box.mesh.vertices @ np.array(result.proposal.matrix).T
        + result.proposal.offset,
        measured.vertices,
        atol=1e-8,
    )


# %% Eligibility and frame stability
def test_oriented_inspection_still_rejects_nonbox_surface() -> None:
    """
    An oriented bounding box alone cannot establish that a cylinder is a box.
    """
    inspected = OrientedBoxInspection().inspect(
        mesh=trimesh.creation.cylinder(radius=0.3, height=0.6)
    )
    assert inspected.bounds is None


def test_role_with_different_box_directions_is_not_resized_along_visual_axes(
    oriented_box: BodyMesh, box_placement: np.ndarray
) -> None:
    """
    A common role mapping requires compatible box directions in both representations.
    """
    collision = trimesh.creation.box(extents=[0.98, 0.58, 0.38])
    collision.apply_transform(trimesh.transformations.rotation_matrix(0.12, [0, 0, 1]))
    collision.apply_transform(box_placement)
    snapshot = WorldSnapshot(
        meshes=[
            oriented_box,
            BodyMesh(
                body_name=oriented_box.body_name,
                role=GeometryRole.COLLISION,
                mesh=collision,
            ),
        ]
    )
    info = BodyDimensionInfo.inspect(
        snapshot=snapshot,
        body_name=oriented_box.body_name,
        role="visual",
        frame_mode=DimensionFrame.BOX.value,
    )
    assert info.bounds is None


def test_canonical_box_axes_survive_vertex_reordering(oriented_box: BodyMesh) -> None:
    """
    Equivalent triangle order must not permute the dimension controls.
    """
    first = OrientedBoxInspection().inspect(mesh=oriented_box.mesh)
    reordered = trimesh.Trimesh(
        vertices=oriented_box.mesh.vertices,
        faces=oriented_box.mesh.faces[::-1],
        process=False,
    )
    second = OrientedBoxInspection().inspect(mesh=reordered)
    assert first.bounds is not None
    assert second.bounds is not None
    np.testing.assert_allclose(
        first.frame.root_T_box, second.frame.root_T_box, atol=1e-8
    )


# %% Oriented revision composition
@pytest.fixture
def oriented_project(
    tmp_path: Path, oriented_box: BodyMesh, box_placement: np.ndarray
) -> RefinementProject:
    """
    Both roles share geometric directions while keeping different original extents.
    """
    collision = trimesh.creation.box(extents=[0.98, 0.58, 0.38])
    collision.apply_transform(box_placement)
    source = tmp_path / "scan.npz"
    np.savez(
        source,
        points=oriented_box.mesh.vertices,
        colors=np.full(oriented_box.mesh.vertices.shape, 128, dtype=np.uint8),
    )
    return RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=source,
        snapshot=WorldSnapshot(
            meshes=[
                oriented_box,
                BodyMesh(
                    body_name=oriented_box.body_name,
                    role=GeometryRole.COLLISION,
                    mesh=collision,
                ),
            ]
        ),
        model_source="oriented-fixture",
    )


def test_oriented_resize_composes_with_root_translation_and_undo(
    oriented_project: RefinementProject, box_placement: np.ndarray
) -> None:
    """
    A geometric-frame resize preserves a prior root translation and all role
    differences.
    """
    body = oriented_project.snapshot.meshes[0].body_name
    oriented_project.apply_translation(
        body_name=body, translation=[0.1, -0.03, 0], source_job="pose"
    )
    before = [
        surface.mesh.vertices.copy()
        for surface in oriented_project.effective_snapshot.meshes
    ]
    rotation = box_placement[:3, :3]
    matrix = rotation @ np.diag([1.04, 1, 1]) @ rotation.T
    center = box_placement[:3, 3] + [0.1, -0.03, 0]
    offset = center - matrix @ center
    oriented_project.apply_dimensions(
        body_name=body,
        scale=[1, 1, 1],
        offset=offset.tolist(),
        source_job="dimensions",
        matrix=matrix.tolist(),
    )
    for surface, vertices in zip(oriented_project.effective_snapshot.meshes, before):
        np.testing.assert_allclose(
            surface.mesh.vertices, vertices @ matrix.T + offset, atol=1e-8
        )
    oriented_project.apply_translation(
        body_name=body, translation=[0, 0, 0.01], source_job="later-pose"
    )
    for surface, vertices in zip(oriented_project.effective_snapshot.meshes, before):
        np.testing.assert_allclose(
            surface.mesh.vertices,
            vertices @ matrix.T + offset + [0, 0, 0.01],
            atol=1e-8,
        )
    oriented_project.undo()
    oriented_project.undo()
    for surface, vertices in zip(oriented_project.effective_snapshot.meshes, before):
        np.testing.assert_allclose(surface.mesh.vertices, vertices, atol=1e-8)


def test_empty_matrix_extension_keeps_legacy_dimension_fingerprint(
    oriented_project: RefinementProject,
) -> None:
    """
    Existing saved diagonal-only dimension revisions retain their prior job identity.
    """
    override = BodyDimensions(
        body_name="legacy", scale=[1.04, 1, 1], offset=[0, 0, 0], source_job="old-job"
    )
    baseline = JobBaseline(
        world_sha256="world",
        scan_sha256="scan",
        registration=RegistrationResult(),
        dimension_overrides=[override],
    )
    encoded = DataclassJSONSerializer.to_json(instance=baseline)
    encoded.pop("overrides")
    encoded["dimension_overrides"]["__items__"][0].pop("matrix", None)
    expected = hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest()
    assert baseline.signature == expected


# %% Browser geometric axes
def test_browser_uses_full_stretch_matrix_for_effective_and_preview_geometry(
    javascript_runtime: str,
) -> None:
    """
    Off-diagonal components must not be dropped when a box is rotated in the root.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/oriented_dimension_buffers.cjs"),
            str(root / "scan_refiner/web/viewer.js"),
        ],
        check=True,
    )


# %% Controls and input validation
def test_oriented_dimension_controls_expose_geometric_frame() -> None:
    """
    The fit frame must be explicit instead of silently interpreting locks as root axes.
    """
    page = (Path(__file__).parents[1] / "scan_refiner/web/viewer.html").read_text()
    assert 'id="dimension-frame"' in page
    assert '<option value="box"' in page


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_dimension_mapping_rejects_rotation_shear_or_reflection(
    matrix: list[list[float]],
) -> None:
    """
    The stored affine mapping cannot introduce an unrelated shape or pose change.
    """
    with pytest.raises(InvalidEvidence):
        BodyDimensions(
            body_name="box", scale=[1, 1, 1], offset=[0, 0, 0], matrix=matrix
        ).validate()


def test_canonical_axes_survive_extent_order_changes(
    oriented_box: BodyMesh, box_placement: np.ndarray
) -> None:
    """
    A resize across another extent must not swap the user's dimension axis identities.
    """
    before = OrientedBoxInspection().inspect(mesh=oriented_box.mesh)
    changed = trimesh.creation.box(extents=[0.5, 0.6, 0.4])
    changed.apply_transform(box_placement)
    after = OrientedBoxInspection().inspect(mesh=changed)
    np.testing.assert_allclose(before.frame.basis, after.frame.basis, atol=1e-8)
