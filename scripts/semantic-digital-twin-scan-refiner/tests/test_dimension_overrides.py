"""
Dimensional overrides compose with translations without rewriting baseline geometry.
"""

from __future__ import annotations

# %% Imports
from copy import deepcopy
import pytest
from scan_refiner.history import ProjectState
from scan_refiner.dimensions import BodyDimensions
from scan_refiner.refinement import BodyTranslation
import subprocess
from pathlib import Path
import numpy as np
from scan_refiner.jobs import JobManager
from scan_refiner.project import RefinementProject
from scan_refiner.records import ProjectFile


# %% Affine revision order
def test_resizing_current_geometry_composes_with_prior_and_later_translation(
    small_project: RefinementProject,
) -> None:
    """
    Resize anchors refer to the current root frame, independently of previous pose
    edits.
    """
    small_project.apply_translation(
        body_name="test/box", translation=[0.1, 0, 0], source_job="pose"
    )
    current = small_project.effective_snapshot.meshes[0].mesh.vertices.copy()
    small_project.apply_dimensions(
        body_name="test/box", scale=[1.1, 1, 1], offset=[-0.01, 0, 0], source_job="size"
    )
    expected = current * [1.1, 1, 1] + [-0.01, 0, 0]
    np.testing.assert_allclose(
        small_project.effective_snapshot.meshes[0].mesh.vertices, expected
    )
    small_project.apply_translation(
        body_name="test/box", translation=[0.02, 0, 0], source_job="later-pose"
    )
    np.testing.assert_allclose(
        small_project.effective_snapshot.meshes[0].mesh.vertices,
        expected + [0.02, 0, 0],
    )
    small_project.undo()
    small_project.undo()
    np.testing.assert_allclose(
        small_project.effective_snapshot.meshes[0].mesh.vertices, current
    )


def test_dimension_override_changes_fingerprint_and_restores_exactly_with_undo(
    small_project: RefinementProject,
) -> None:
    """
    Baseline bytes and valid old analysis identities survive a full resize/Undo cycle.
    """
    baseline = (small_project.directory / ProjectFile.WORLD).read_bytes()
    vertices = small_project.snapshot.meshes[0].mesh.vertices.copy()
    manager = JobManager(project=small_project)
    signature = manager.baseline.signature
    small_project.apply_dimensions(
        body_name="test/box", scale=[1.04, 1, 1], offset=[0, 0, 0], source_job="resize"
    )
    assert manager.baseline.signature != signature
    assert (small_project.directory / ProjectFile.WORLD).read_bytes() == baseline
    small_project.undo()
    assert manager.baseline.signature == signature
    np.testing.assert_array_equal(
        small_project.effective_snapshot.meshes[0].mesh.vertices, vertices
    )
    small_project.redo()
    assert small_project.current_state.dimension_overrides[0].scale == [1.04, 1, 1]


# %% Browser deformation order
def test_browser_composes_dimensions_pose_and_preview_from_baseline(
    javascript_runtime: str,
) -> None:
    """
    The rendered affine order must match the effective Python snapshot after Undo/Redo.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/dimension_buffers.cjs"),
            str(root / "scan_refiner/web/viewer.js"),
        ],
        check=True,
    )


# %% Coherent revision fingerprints
def test_analysis_fingerprint_reads_one_complete_geometry_revision(
    small_project: RefinementProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Pose and dimension fields from different revisions must never be mixed in one
    fingerprint.
    """
    first = deepcopy(small_project.current_state)
    first.overrides = [
        BodyTranslation(
            body_name="test/box", translation=[0.01, 0, 0], source_job="first"
        )
    ]
    first.dimension_overrides = [
        BodyDimensions(
            body_name="test/box",
            scale=[1.01, 1, 1],
            offset=[0, 0, 0],
            source_job="first",
        )
    ]
    second = deepcopy(first)
    second.dimension_overrides = [
        BodyDimensions(
            body_name="test/box",
            scale=[1.02, 1, 1],
            offset=[0, 0, 0],
            source_job="second",
        )
    ]
    revisions = iter([first, second])

    def read_state(project: RefinementProject) -> ProjectState:
        """
        Expose successive atomic journals as concurrent model edits would do.
        """
        return next(revisions)

    monkeypatch.setattr(RefinementProject, "current_state", property(read_state))
    baseline = JobManager(project=small_project).baseline
    assert baseline.overrides == first.overrides
    assert baseline.dimension_overrides == first.dimension_overrides
