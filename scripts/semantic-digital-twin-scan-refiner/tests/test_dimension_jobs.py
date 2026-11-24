"""
Dimension jobs remain proposals and retain each geometry role's relative shape.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import numpy as np
import pytest
import trimesh
from typing_extensions import Iterator
from scan_refiner.dimensions import (
    DimensionConstraints,
    DimensionProposal,
    DimensionFrame,
    OrientedBoxInspection,
)
from scan_refiner.job_worker import JobWorker
from scan_refiner.jobs import JobRecord
from scan_refiner.jobs import JobKind, JobManager, JobRequest, JobStatus
from scan_refiner.project import RefinementProject
from scan_refiner.records import (
    GeometryRole,
    ProjectFile,
    RegistrationCandidate,
    RegistrationResult,
)
from scan_refiner.world import BodyMesh, WorldSnapshot


# %% Frozen box scenario
@pytest.fixture
def dimension_placement() -> np.ndarray:
    """
    Existing root-box cases keep their original geometry frame.
    """
    return np.eye(4)


@pytest.fixture
def dimension_manager(
    tmp_path: Path, dimension_placement: np.ndarray
) -> Iterator[JobManager]:
    """
    A visual and deliberately smaller collision share a body with dense measured faces.
    """
    visual = trimesh.creation.box(extents=[1, 0.6, 0.4])
    collision = trimesh.creation.box(extents=[0.98, 0.58, 0.38])
    target = trimesh.creation.box(extents=[1.04, 0.6, 0.4])
    for mesh in (visual, collision, target):
        mesh.apply_transform(dimension_placement)
    points, _ = trimesh.sample.sample_surface(target, 6000, seed=23)
    source = tmp_path / "scan.npz"
    np.savez(source, points=points, colors=np.full(points.shape, 128, dtype=np.uint8))
    project = RefinementProject.create(
        directory=tmp_path / "dimension-project",
        scan_path=source,
        snapshot=WorldSnapshot(
            meshes=[
                BodyMesh(body_name="box", role=GeometryRole.VISUAL, mesh=visual),
                BodyMesh(body_name="box", role=GeometryRole.COLLISION, mesh=collision),
            ]
        ),
        model_source="synthetic-box-width",
    )
    project.update_registration(
        result=RegistrationResult(
            candidates=[
                RegistrationCandidate(
                    scan_T_root=np.eye(4).tolist(),
                    support_fraction=1,
                    median_distance=0,
                )
            ]
        ),
        role="visual",
    )
    manager = JobManager(project=project)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def dimension_job(dimension_manager: JobManager) -> tuple[JobManager, str]:
    """
    Complete a bounded X-width fit without applying its geometry.
    """
    request = JobRequest(
        kind=JobKind.DIMENSIONS.value,
        candidate_index=0,
        body_name="box",
        lower=[-1, -1, -1],
        upper=[1, 1, 1],
        maximum_count=6000,
        dimension_constraints=DimensionConstraints(axes=[True, False, False]),
    )
    record = dimension_manager.start(request=request)
    dimension_manager.processes[record.identifier].wait(timeout=30)
    assert (
        dimension_manager.get(identifier=record.identifier).status
        == JobStatus.SUCCEEDED
    )
    return dimension_manager, record.identifier


# %% Explicit application
def test_dimension_job_preserves_baseline_until_reviewed_application(
    dimension_job: tuple[JobManager, str],
) -> None:
    """
    Applying one shared affine mapping preserves a smaller collision rather than
    equalizing it.
    """
    manager, identifier = dimension_job
    proposal = manager.read_proposal(identifier=identifier)
    assert isinstance(proposal, DimensionProposal)
    assert proposal.applicable
    assert manager.project.current_state.dimension_overrides == []
    baseline = (manager.project.directory / ProjectFile.WORLD).read_bytes()
    original = [
        surface.mesh.vertices.copy()
        for surface in manager.project.effective_snapshot.meshes
    ]
    manager.apply(identifier=identifier)
    for surface, vertices in zip(manager.project.effective_snapshot.meshes, original):
        np.testing.assert_allclose(
            surface.mesh.vertices, vertices * np.array(proposal.scale) + proposal.offset
        )
    lengths = [
        surface.mesh.extents[0] for surface in manager.project.effective_snapshot.meshes
    ]
    np.testing.assert_allclose(lengths, [1.04, 0.98 * 1.04], atol=1e-8)
    assert (manager.project.directory / ProjectFile.WORLD).read_bytes() == baseline
    manager.project.undo()
    for surface, vertices in zip(manager.project.effective_snapshot.meshes, original):
        np.testing.assert_array_equal(surface.mesh.vertices, vertices)


def test_worker_pins_dimensional_geometry_at_submission(
    dimension_job: tuple[JobManager, str],
) -> None:
    """
    A later resize cannot enter a job that captured the earlier model revision.
    """
    manager, identifier = dimension_job
    manager.apply(identifier=identifier)
    before = [
        surface.mesh.vertices.copy()
        for surface in manager.project.effective_snapshot.meshes
    ]
    record = manager.start(request=JobRequest(candidate_index=0, maximum_count=8))
    manager.cancel(identifier=record.identifier)
    manager.project.apply_dimensions(
        body_name="box", scale=[1.01, 1, 1], offset=[0, 0, 0], source_job="later-size"
    )
    frozen = JobRecord.read(directory=manager.directory(identifier=record.identifier))
    worker = JobWorker(
        project=manager.project,
        directory=manager.directory(identifier=record.identifier),
    )
    for surface, vertices in zip(worker.load_geometry(record=frozen).meshes, before):
        np.testing.assert_allclose(surface.mesh.vertices, vertices)


# %% Oriented worker and publication
@pytest.mark.parametrize(
    "dimension_placement", [trimesh.transformations.euler_matrix(0.15, -0.2, 0.35)]
)
def test_oriented_job_freezes_stretch_and_undo_restores_both_roles(
    dimension_manager: JobManager,
) -> None:
    """
    The complete job path retains geometric axes, original evidence and frozen matrices.
    """
    manager = dimension_manager
    original = [
        surface.mesh.vertices.copy()
        for surface in manager.project.effective_snapshot.meshes
    ]
    request = JobRequest(
        kind=JobKind.DIMENSIONS.value,
        candidate_index=0,
        body_name="box",
        lower=[-1, -1, -1],
        upper=[1, 1, 1],
        maximum_count=6000,
        dimension_constraints=DimensionConstraints(
            axes=[True, False, False], frame_mode=DimensionFrame.BOX.value
        ),
    )
    record = manager.start(request=request)
    manager.processes[record.identifier].wait(timeout=30)
    assert manager.get(identifier=record.identifier).status == JobStatus.SUCCEEDED
    proposal = manager.read_proposal(identifier=record.identifier)
    assert proposal.applicable
    assert manager.project.current_state.dimension_overrides == []
    np.testing.assert_allclose(
        proposal.after_box.dimensions, [1.04, 0.6, 0.4], atol=1e-8
    )
    manager.apply(identifier=record.identifier)
    before = [
        surface.mesh.vertices.copy()
        for surface in manager.project.effective_snapshot.meshes
    ]
    for vertices, baseline in zip(before, original):
        np.testing.assert_allclose(
            vertices,
            baseline @ np.array(proposal.matrix).T + proposal.offset,
            atol=1e-8,
        )
    widths = [
        OrientedBoxInspection().inspect(mesh=surface.mesh).bounds.dimensions[0]
        for surface in manager.project.effective_snapshot.meshes
    ]
    np.testing.assert_allclose(widths, [1.04, 0.98 * 1.04], atol=1e-8)
    frozen_job = manager.start(request=JobRequest(candidate_index=0, maximum_count=8))
    manager.cancel(identifier=frozen_job.identifier)
    manager.project.undo()
    frozen = JobRecord.read(
        directory=manager.directory(identifier=frozen_job.identifier)
    )
    worker = JobWorker(
        project=manager.project,
        directory=manager.directory(identifier=frozen_job.identifier),
    )
    for surface, vertices in zip(worker.load_geometry(record=frozen).meshes, before):
        np.testing.assert_allclose(surface.mesh.vertices, vertices, atol=1e-8)
    for surface, vertices in zip(manager.project.effective_snapshot.meshes, original):
        np.testing.assert_array_equal(surface.mesh.vertices, vertices)
