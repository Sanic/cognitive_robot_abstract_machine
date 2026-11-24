"""
Verify durable project baselines and explicit transform acceptance.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scan_refiner.exceptions import ExistingProject, InvalidEvidence
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh, WorldSnapshot


# %% Baseline persistence
def test_project_reopens_with_original_geometry_and_scan(tmp_path: Path) -> None:
    """
    Project reopening uses saved inputs rather than a mutable external model.
    """
    mesh = trimesh.creation.box()
    snapshot = WorldSnapshot(
        meshes=[BodyMesh(body_name="room/box", role=GeometryRole.VISUAL, mesh=mesh)]
    )
    source = tmp_path / "preview.npz"
    points = mesh.vertices.copy()
    np.savez(source, points=points, colors=np.full(points.shape, 128, dtype=np.uint8))
    directory = tmp_path / "project"
    project = RefinementProject.create(
        directory=directory,
        scan_path=source,
        snapshot=snapshot,
        model_source="test-world",
    )
    mesh.apply_translation((10, 0, 0))
    reopened = RefinementProject(directory=directory)
    np.testing.assert_array_equal(reopened.scan.points, points)
    np.testing.assert_array_equal(reopened.snapshot.meshes[0].mesh.vertices, points)
    assert reopened.record.scan_sha256 == project.record.scan_sha256
    assert reopened.record.model_source == "test-world"
    assert reopened.record.accepted_candidate is None


def test_project_does_not_overwrite_an_existing_directory(tmp_path: Path) -> None:
    """
    Initialization never replaces an existing project or unrelated directory.
    """
    with pytest.raises(ExistingProject):
        RefinementProject.create(
            directory=tmp_path,
            scan_path=tmp_path / "unused",
            snapshot=WorldSnapshot(meshes=[]),
            model_source="test",
        )


def test_scan_rejects_non_metric_units_without_explicit_conversion(
    tmp_path: Path,
) -> None:
    """
    Coordinate interpretation must be supplied explicitly for each import.
    """
    with pytest.raises(InvalidEvidence):
        RefinementProject.create(
            directory=tmp_path / "project",
            scan_path=tmp_path / "unused",
            snapshot=WorldSnapshot(meshes=[]),
            model_source="test",
            scan_scale=0,
        )


def test_acceptance_requires_a_current_candidate(tmp_path: Path) -> None:
    """
    An arbitrary candidate number cannot become an accepted registration.
    """
    source = tmp_path / "preview.npz"
    snapshot = WorldSnapshot(
        meshes=[
            BodyMesh(
                body_name="box", role=GeometryRole.VISUAL, mesh=trimesh.creation.box()
            )
        ]
    )
    np.savez(source, points=snapshot.meshes[0].mesh.vertices)
    project = RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=source,
        snapshot=snapshot,
        model_source="test",
    )
    with pytest.raises(InvalidEvidence):
        project.accept(candidate_index=0)
