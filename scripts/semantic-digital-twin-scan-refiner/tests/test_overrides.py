"""
Undoable selected-body changes preserve immutable baselines and other bodies.
"""

from __future__ import annotations

# %% Imports
import subprocess
from pathlib import Path
import numpy as np
from scan_refiner.project import RefinementProject
from scan_refiner.world import BodyMesh
from scan_refiner.records import GeometryRole
from scan_refiner.records import ProjectFile
from scan_refiner.jobs import JobManager


# %% Immutable world revisions
def test_translation_override_is_reversible_without_rewriting_baseline(
    small_project: RefinementProject,
) -> None:
    """
    Applied displacements alter effective geometry and restore exactly through history.
    """
    baseline = (small_project.directory / ProjectFile.WORLD).read_bytes()
    vertices = small_project.snapshot.meshes[0].mesh.vertices.copy()
    signature = JobManager(project=small_project).baseline.signature
    small_project.apply_translation(
        body_name="test/box", translation=[0, 0, 0.02], source_job="proposal"
    )
    np.testing.assert_allclose(
        small_project.effective_snapshot.meshes[0].mesh.vertices,
        vertices + [0, 0, 0.02],
    )
    assert JobManager(project=small_project).baseline.signature != signature
    assert (small_project.directory / ProjectFile.WORLD).read_bytes() == baseline
    small_project.undo()
    np.testing.assert_array_equal(
        small_project.effective_snapshot.meshes[0].mesh.vertices, vertices
    )
    assert JobManager(project=small_project).baseline.signature == signature
    small_project.redo()
    assert small_project.current_state.overrides[0].translation == [0, 0, 0.02]


def test_sequential_overrides_accumulate_relative_to_baseline(
    small_project: RefinementProject,
) -> None:
    """
    A second proposal moves the current revision rather than replacing the first delta.
    """
    small_project.apply_translation(
        body_name="test/box", translation=[0.01, 0, 0.02], source_job="first"
    )
    small_project.apply_translation(
        body_name="test/box", translation=[0, 0.01, -0.01], source_job="second"
    )
    assert small_project.current_state.overrides[0].translation == [0.01, 0.01, 0.01]
    assert len(small_project.current_state.overrides) == 1
    small_project.undo()
    assert small_project.current_state.overrides[0].source_job == "first"


# %% Browser model revision
def test_browser_rebuilds_overrides_from_original_geometry(
    javascript_runtime: str,
) -> None:
    """
    Undo must not leave accumulated displacement in the browser's mesh buffers.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/body_overrides.cjs"),
            str(root / "scan_refiner/web/viewer.js"),
        ],
        check=True,
    )


# %% Representation consistency
def test_selected_body_moves_all_roles_but_not_other_bodies(
    small_project: RefinementProject, tmp_path: Path
) -> None:
    """
    A body-level override preserves visual/collision alignment and explicit scope.
    """
    snapshot = small_project.snapshot
    mesh = snapshot.meshes[0].mesh
    snapshot.meshes.extend(
        [
            BodyMesh(
                body_name="test/box", role=GeometryRole.COLLISION, mesh=mesh.copy()
            ),
            BodyMesh(
                body_name="test/fixed", role=GeometryRole.VISUAL, mesh=mesh.copy()
            ),
        ]
    )
    project = RefinementProject.create(
        directory=tmp_path / "roles",
        scan_path=small_project.directory / ProjectFile.SCAN,
        snapshot=snapshot,
        model_source="fixture-roles",
    )
    project.apply_translation(
        body_name="test/box", translation=[0, 0, 0.02], source_job="move"
    )
    for surface in project.effective_snapshot.meshes:
        np.testing.assert_allclose(
            surface.mesh.vertices,
            mesh.vertices
            + ([0, 0, 0.02] if surface.body_name == "test/box" else [0, 0, 0]),
        )


# %% Atomic application
def test_browser_locks_history_during_override_application(
    javascript_runtime: str,
) -> None:
    """
    Undo cannot interleave with model application or its state refresh.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/refinement_atomic.cjs"),
            str(root / "scan_refiner/web/refinement.js"),
        ],
        check=True,
    )
