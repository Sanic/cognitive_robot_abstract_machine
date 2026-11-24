"""
Exercise bounded jobs, cancellation and stale-result protection.
"""

from __future__ import annotations

# %% Imports
from scan_refiner.refinement import TranslationConstraints, TranslationProposal
import json
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.coverage import ComparisonReport, ComparisonArray
from scan_refiner.jobs import JobRecord
from scan_refiner.job_worker import JobWorker
from scan_refiner.jobs import JobFile
from copy import deepcopy
from time import monotonic, sleep
import numpy as np
import pytest
from typing_extensions import Iterator
from scan_refiner.jobs import JobManager, JobRequest, JobKind, JobStatus
from scan_refiner.project import RefinementProject
from scan_refiner.records import RegistrationCandidate, RegistrationResult
from scan_refiner.registration import RegistrationSearch
from scan_refiner.scan_sources import PreviewPointSource
from scan_refiner.spatial_index import SpatialScanIndex
from scan_refiner.records import ProjectFile


# %% Owned workers
@pytest.fixture
def job_manager(small_project: RefinementProject) -> Iterator[JobManager]:
    """
    Own worker lifetimes independently of the caller's project decisions.
    """
    candidate = RegistrationCandidate(
        scan_T_root=np.eye(4).tolist(), support_fraction=1.0, median_distance=0.0
    )
    small_project.update_registration(
        result=RegistrationResult(candidates=[candidate]), role="visual"
    )
    manager = JobManager(project=small_project)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def completed_heatmap(job_manager: JobManager) -> tuple[JobManager, str]:
    """
    Run a tiny surface comparison with exact source point identities.
    """
    job = job_manager.start(
        request=JobRequest(
            kind=JobKind.HEATMAP.value, candidate_index=0, maximum_count=8
        )
    )
    deadline = monotonic() + 30
    while (
        job_manager.get(identifier=job.identifier).status
        in (JobStatus.QUEUED, JobStatus.RUNNING)
        and monotonic() < deadline
    ):
        sleep(0.05)
    assert job_manager.get(identifier=job.identifier).status == JobStatus.SUCCEEDED
    return job_manager, job.identifier


# %% Results and provenance
def test_heatmap_job_preserves_project_decisions(
    completed_heatmap: tuple[JobManager, str],
) -> None:
    """
    Computing derived distances cannot accept or replace a registration.
    """
    manager, identifier = completed_heatmap
    job = manager.get(identifier=identifier)
    assert job.completed == job.total == 8
    with np.load(
        manager.result_path(identifier=identifier), allow_pickle=False
    ) as result:
        np.testing.assert_array_equal(result["distances"], np.zeros(8))
        assert sorted(result["records"]["source_indices"].tolist()) == list(range(8))
    assert manager.project.record.accepted_candidate is None
    assert len(manager.project.history.load().past) == 1


def test_changed_registration_marks_completed_heatmap_stale(
    completed_heatmap: tuple[JobManager, str],
) -> None:
    """
    A result cannot be presented against a different registration revision.
    """
    manager, identifier = completed_heatmap
    registration = deepcopy(manager.project.registration)
    registration.candidates[0].scan_T_root[0][3] = 1.0
    manager.project.update_registration(result=registration, role="visual")
    assert manager.get(identifier=identifier).status == JobStatus.STALE


def test_cancelling_worker_keeps_prior_project_state(job_manager: JobManager) -> None:
    """
    Cancelling a computation terminates its own worker without changing decisions.
    """
    before = job_manager.project.current_state
    job = job_manager.start(
        request=JobRequest(
            kind=JobKind.HEATMAP.value, candidate_index=0, maximum_count=8
        )
    )
    job_manager.cancel(identifier=job.identifier)
    assert job_manager.get(identifier=job.identifier).status == JobStatus.CANCELLED
    assert job_manager.project.current_state == before


def test_registration_job_requires_explicit_application(
    job_manager: JobManager,
) -> None:
    """
    Candidate generation is separate from undoable replacement of prior proposals.
    """
    before = job_manager.project.registration
    request = JobRequest(
        kind=JobKind.REGISTRATION.value,
        upright=True,
        search=RegistrationSearch(
            candidate_count=2, yaw_step_degrees=90, refinement_iterations=2
        ),
    )
    job = job_manager.start(request=request)
    deadline = monotonic() + 30
    while (
        job_manager.get(identifier=job.identifier).status
        in (JobStatus.QUEUED, JobStatus.RUNNING)
        and monotonic() < deadline
    ):
        sleep(0.05)
    assert job_manager.get(identifier=job.identifier).status == JobStatus.SUCCEEDED
    assert job_manager.project.registration == before
    job_manager.apply(identifier=job.identifier)
    assert (
        len(job_manager.project.registration.candidates)
        == request.search.candidate_count
    )
    job_manager.project.undo()
    assert job_manager.project.registration == before


# %% Dense source revisions
def test_changed_index_identity_marks_result_stale(
    completed_heatmap: tuple[JobManager, str],
) -> None:
    """
    Changing the dense evidence source invalidates an older preview comparison.
    """
    manager, identifier = completed_heatmap
    SpatialScanIndex.build(
        directory=manager.project.directory / ProjectFile.SCAN_INDEX,
        batches=PreviewPointSource(preview=manager.project.scan).batches(),
        source_identity="different-scan-index",
    )
    assert manager.get(identifier=identifier).status == JobStatus.STALE


# %% Bidirectional report artifacts
def test_comparison_publishes_labelled_directional_report(
    completed_heatmap: tuple[JobManager, str],
) -> None:
    """
    A report and numeric labels belong to the same immutable comparison job.
    """
    manager, identifier = completed_heatmap
    record = manager.get(identifier=identifier)
    report = DataclassJSONSerializer.from_json(
        data=json.loads(
            (manager.directory(identifier=identifier) / JobFile.REPORT).read_text()
        ),
        clazz=ComparisonReport,
    )
    assert report.scan.sample_count == record.total
    assert report.scan.median_distance == record.median_distance
    assert report.tolerance == record.request.tolerance
    assert [body.body_name for body in report.bodies] == ["test/box"]
    assert report.model_sample_count == record.request.model_sample_count
    with np.load(
        manager.result_path(identifier=identifier), allow_pickle=False
    ) as arrays:
        np.testing.assert_array_equal(
            arrays[ComparisonArray.SCAN_BODY_INDICES], np.zeros(record.total)
        )
        assert len(arrays[ComparisonArray.MODEL_POINTS]) == report.model_sample_count
        np.testing.assert_array_equal(
            arrays[ComparisonArray.MODEL_BODY_INDICES],
            np.zeros(report.model_sample_count),
        )


# %% Resource contracts
@pytest.mark.parametrize("count", [0, 50001, 1.5])
def test_model_sample_budget_must_be_bounded_integer(count: int | float) -> None:
    """
    Malformed sample counts are rejected before they can start a worker.
    """
    assert JobRequest(candidate_index=0, model_sample_count=count).problem is not None


# %% Reviewed geometry proposals
def test_refinement_job_can_only_apply_reviewable_translation(
    job_manager: JobManager,
) -> None:
    """
    Computing a translation leaves geometry unchanged until explicit undoable
    application.
    """
    scan = job_manager.project.scan
    scan.points[:, 2] += 0.02
    scan.save(path=job_manager.project.directory / ProjectFile.SCAN)
    request = JobRequest(
        kind=JobKind.REFINEMENT.value,
        candidate_index=0,
        body_name="test/box",
        lower=[-1, -1, -1],
        upper=[1, 1, 1],
        maximum_count=8,
        constraints=TranslationConstraints(axes=[False, False, True], minimum_points=3),
    )
    before = job_manager.project.effective_snapshot.meshes[0].mesh.vertices.copy()
    job = job_manager.start(request=request)
    deadline = monotonic() + 30
    while (
        job_manager.get(identifier=job.identifier).status
        in (JobStatus.QUEUED, JobStatus.RUNNING)
        and monotonic() < deadline
    ):
        sleep(0.05)
    assert job_manager.get(identifier=job.identifier).status == JobStatus.SUCCEEDED
    np.testing.assert_array_equal(
        job_manager.project.effective_snapshot.meshes[0].mesh.vertices, before
    )
    proposal = DataclassJSONSerializer.from_json(
        data=json.loads(
            (
                job_manager.directory(identifier=job.identifier) / JobFile.PROPOSAL
            ).read_text()
        ),
        clazz=TranslationProposal,
    )
    assert proposal.applicable
    job_manager.apply(identifier=job.identifier)
    np.testing.assert_allclose(
        job_manager.project.current_state.overrides[0].translation,
        [0, 0, 0.02],
        atol=1e-6,
    )
    job_manager.project.undo()
    np.testing.assert_array_equal(
        job_manager.project.effective_snapshot.meshes[0].mesh.vertices, before
    )


# %% Model revision invalidation
def test_body_override_invalidates_comparison_until_undo(
    completed_heatmap: tuple[JobManager, str],
) -> None:
    """
    All geometry-derived results depend on the effective model revision.
    """
    manager, identifier = completed_heatmap
    manager.project.apply_translation(
        body_name="test/box", translation=[0, 0, 0.01], source_job="move"
    )
    assert manager.get(identifier=identifier).status == JobStatus.STALE
    manager.project.undo()
    assert manager.get(identifier=identifier).status == JobStatus.SUCCEEDED


# %% Frozen worker inputs
def test_worker_uses_geometry_captured_at_submission(job_manager: JobManager) -> None:
    """
    Later model edits cannot change the geometry computed under an earlier job
    fingerprint.
    """
    before = job_manager.project.effective_snapshot.meshes[0].mesh.vertices.copy()
    job = job_manager.start(request=JobRequest(candidate_index=0, maximum_count=8))
    job_manager.cancel(identifier=job.identifier)
    job_manager.project.apply_translation(
        body_name="test/box", translation=[0, 0, 0.02], source_job="later"
    )
    record = JobRecord.read(directory=job_manager.directory(identifier=job.identifier))
    geometry = JobWorker(
        project=job_manager.project,
        directory=job_manager.directory(identifier=job.identifier),
    ).load_geometry(record=record)
    np.testing.assert_array_equal(geometry.meshes[0].mesh.vertices, before)
