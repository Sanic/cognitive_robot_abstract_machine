"""
An agent uses existing fitting and journal operations within a frozen server scope.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from unittest.mock import patch
from scan_refiner.agent.models import JobInspection
from scan_refiner.jobs import JobRecord, JobRequest
import numpy as np
import pytest
import threading
import subprocess
import sys
import json
from urllib.request import urlopen
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.agent.client import AgentClient
from scan_refiner.agent.codec import AgentCodec
from scan_refiner.agent.models import ProjectSummary
from scan_refiner.web_service import InspectionService, InspectionServer
import trimesh
from scan_refiner.agent.service import AgentService
from scan_refiner.agent.models import (
    AgentScope,
    AgentMode,
    ChangeSetQuery,
    ChangeSetDecisionQuery,
    ChangeSetStatus,
    ChangeSetHeatmapQuery,
    TranslationEdit,
    ProjectQuery,
    RegionQuery,
    RegionSampleQuery,
    EvidenceResolution,
    RefinementQuery,
    Reasoning,
    ApplyQuery,
    HistoryQuery,
    BodyQuery,
    JobQuery,
    RegionRenderQuery,
    RenderView,
)
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.project import RefinementProject
from scan_refiner.jobs import JobManager, JobStatus
from scan_refiner.records import GeometryRole, RegistrationCandidate, RegistrationResult
from scan_refiner.refinement import TranslationConstraints
from scan_refiner.world import WorldSnapshot, BodyMesh
from typing_extensions import Iterator
from semantic_digital_twin.adapters.urdf import URDFParser


# %% Known geometric discrepancy
@pytest.fixture
def agent_service(tmp_path: Path) -> Iterator[AgentService]:
    """
    A translated scan of a cuboid provides bounded, observable synthetic evidence.
    """
    model = trimesh.creation.box(extents=[1, 0.6, 0.4])
    points, _ = trimesh.sample.sample_surface(model, 5000, seed=12)
    points[:, 0] += 0.02
    source = tmp_path / "scan.npz"
    np.savez(source, points=points, colors=np.full(points.shape, 128, dtype=np.uint8))
    project = RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=source,
        snapshot=WorldSnapshot(
            meshes=[
                BodyMesh(body_name="test/box", role=GeometryRole.VISUAL, mesh=model)
            ]
        ),
        model_source="synthetic-agent-test",
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
    service = AgentService(
        project=project,
        jobs=manager,
        scope=AgentScope(mode=AgentMode.REFINE, body_names=["test/box"]),
    )
    try:
        yield service
    finally:
        manager.close()


# %% Reusable evidence and authority
def test_region_is_reusable_and_explicit_about_preview_evidence(
    agent_service: AgentService,
) -> None:
    """
    A scan-coordinate region retains provenance without returning its raw record arrays.
    """
    revision = agent_service.inspect_project(request=ProjectQuery()).revision
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    summary = agent_service.inspect_region(
        request=RegionSampleQuery(
            region_id=region.identifier,
            maximum_points=1000,
            resolution=EvidenceResolution.PREVIEW,
        )
    )
    assert summary.region.identifier == region.identifier
    assert summary.region.created_revision == revision
    assert summary.resolution == EvidenceResolution.PREVIEW
    assert summary.count <= summary.requested_count
    assert Path(summary.evidence_path).is_file()


def test_stale_revision_is_rejected_before_starting_a_worker(
    agent_service: AgentService,
) -> None:
    """
    Proposal input revisions must match the state the agent actually inspected.
    """
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    with pytest.raises(InvalidEvidence):
        agent_service.propose_refinement(
            request=RefinementQuery(
                expected_revision="stale",
                region_id=region.identifier,
                body_name="test/box",
                reasoning=Reasoning(explanation="Fit observed X offset"),
                translation=TranslationConstraints(axes=[True, False, False]),
            )
        )
    assert agent_service.jobs.processes == {}


def test_agent_scope_blocks_unlisted_bodies(agent_service: AgentService) -> None:
    """
    An agent cannot broaden its server-granted object scope through request fields.
    """
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    agent_service.scope.body_names = ["other/body"]
    with pytest.raises(InvalidEvidence):
        agent_service.propose_refinement(
            request=RefinementQuery(
                expected_revision=agent_service.inspect_project(
                    request=ProjectQuery()
                ).revision,
                region_id=region.identifier,
                body_name="test/box",
                reasoning=Reasoning(explanation="Fit X"),
                translation=TranslationConstraints(axes=[True, False, False]),
            )
        )


def test_agent_fit_apply_undo_redo_uses_existing_journal(
    agent_service: AgentService,
) -> None:
    """
    A reviewed correction applies once, retains its explanation and traverses exact
    history states.
    """
    before = agent_service.project.effective_snapshot.meshes[0].mesh.vertices.copy()
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    job = agent_service.propose_refinement(
        request=RefinementQuery(
            expected_revision=agent_service.inspect_project(
                request=ProjectQuery()
            ).revision,
            region_id=region.identifier,
            body_name="test/box",
            reasoning=Reasoning(
                explanation="Correct the supported X offset",
                assumptions=["Depth and height remain fixed"],
            ),
            translation=TranslationConstraints(axes=[True, False, False]),
            maximum_points=5000,
        )
    )
    agent_service.jobs.processes[job.identifier].wait(timeout=30)
    assert (
        agent_service.inspect_job(request=JobQuery(job_id=job.identifier)).record.status
        == JobStatus.SUCCEEDED
    )
    action = agent_service.apply_proposal(
        request=ApplyQuery(
            job_id=job.identifier,
            expected_revision=agent_service.inspect_project(
                request=ProjectQuery()
            ).revision,
            reasoning=Reasoning(
                explanation="Same-record residual improved within the X limit"
            ),
        )
    )
    np.testing.assert_allclose(
        agent_service.project.effective_snapshot.meshes[0].mesh.vertices,
        before + [0.02, 0, 0],
        atol=0.002,
    )
    agent_service.undo_action(
        request=HistoryQuery(
            action_id=action.identifier, expected_revision=action.after_revision
        )
    )
    np.testing.assert_array_equal(
        agent_service.project.effective_snapshot.meshes[0].mesh.vertices, before
    )
    agent_service.redo_action(
        request=HistoryQuery(
            action_id=action.identifier, expected_revision=action.before_revision
        )
    )
    assert (
        agent_service.inspect_project(request=ProjectQuery()).revision
        == action.after_revision
    )


def test_agent_cannot_undo_an_intervening_human_action(
    agent_service: AgentService,
) -> None:
    """
    History traversal does not silently undo someone else's latest decision.
    """
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    job = agent_service.propose_refinement(
        request=RefinementQuery(
            expected_revision=agent_service.inspect_project(
                request=ProjectQuery()
            ).revision,
            region_id=region.identifier,
            body_name="test/box",
            reasoning=Reasoning(explanation="Fit X"),
            translation=TranslationConstraints(axes=[True, False, False]),
            maximum_points=5000,
        )
    )
    agent_service.jobs.processes[job.identifier].wait(timeout=30)
    action = agent_service.apply_proposal(
        request=ApplyQuery(
            job_id=job.identifier,
            expected_revision=agent_service.inspect_project(
                request=ProjectQuery()
            ).revision,
            reasoning=Reasoning(explanation="Supported fit"),
        )
    )
    inspection = agent_service.project.current_state.inspection
    inspection.view.point_size = 3
    agent_service.project.update_inspection(
        inspection=inspection, label="Human changed point size"
    )
    with pytest.raises(InvalidEvidence):
        agent_service.undo_action(
            request=HistoryQuery(
                action_id=action.identifier, expected_revision=action.after_revision
            )
        )


# %% Registration decision provenance
def test_acceptance_changes_the_agent_revision(agent_service: AgentService) -> None:
    """
    The default placement must not change behind a caller's observed revision token.
    """
    observed = agent_service.inspect_project(request=ProjectQuery()).revision
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    agent_service.project.accept(candidate_index=0)
    with pytest.raises(InvalidEvidence):
        agent_service.propose_refinement(
            request=RefinementQuery(
                expected_revision=observed,
                region_id=region.identifier,
                body_name="test/box",
                reasoning=Reasoning(explanation="Fit inspected placement"),
                translation=TranslationConstraints(axes=[True, False, False]),
            )
        )


# %% Rendered evidence and unsupported permissions
def test_agent_renders_bounded_html_without_changing_inspection(
    agent_service: AgentService,
) -> None:
    """
    A focused image source retains its raw crop and declared display origin.
    """
    state = agent_service.project.current_state
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    artifact = agent_service.render_region(
        request=RegionRenderQuery(
            region_id=region.identifier,
            resolution=EvidenceResolution.PREVIEW,
            image=False,
            maximum_points=1000,
            view=RenderView.TOP,
        )
    )
    assert Path(artifact.html_path).is_file()
    assert artifact.image_path == ""
    assert artifact.region_id == region.identifier
    np.testing.assert_allclose(
        artifact.display_origin, (np.array(region.query.lower) + region.query.upper) / 2
    )
    assert agent_service.project.current_state == state


def test_unsupported_semantic_constraint_is_not_ignored() -> None:
    """
    Unknown semantic constraints must fail rather than become an unconstrained fit.
    """
    with pytest.raises(ValidationError):
        TypeAdapter(RefinementQuery).validate_python(
            {
                "expected_revision": "revision",
                "region_id": "region",
                "body_name": "body",
                "reasoning": {"explanation": "Center the handle"},
                "translation": {"axes": [True, False, False], "center_on": "drawer"},
            }
        )


def test_inspect_only_scope_rejects_application(agent_service: AgentService) -> None:
    """
    Inspect mode can analyze evidence but cannot mutate model history.
    """
    agent_service.scope.mode = AgentMode.INSPECT
    before = agent_service.project.current_state
    with pytest.raises(InvalidEvidence):
        agent_service.apply_proposal(
            request=ApplyQuery(
                job_id="0" * 32,
                expected_revision=agent_service.inspect_project(
                    request=ProjectQuery()
                ).revision,
                reasoning=Reasoning(explanation="Attempt a change"),
            )
        )
    assert agent_service.project.current_state == before


# %% Shared HTTP client
def test_change_set_moves_fixed_parent_and_child_once_and_undoes_as_one(
    agent_service: AgentService,
    tmp_path: Path,
) -> None:
    """
    The proposed deltas are root-relative and shared history restores the group.
    """
    source = Path(__file__).parent / "dataset/source_world.urdf"
    snapshot = WorldSnapshot.from_world(
        world=URDFParser.from_file(file_path=str(source)).create_world()
    )
    baseline = agent_service.project
    project = RefinementProject.create(
        directory=tmp_path / "group-project",
        scan_path=baseline.directory / "scan.npz",
        snapshot=snapshot,
        model_source=str(source),
    )
    agent_service.project = agent_service.jobs.project = project
    parent, child = "test/box", "test/handle"
    revision = agent_service.revision
    shift = [0.0, 0.012, 0.0]
    request = ChangeSetQuery(
        expected_revision=revision,
        edits=[
            TranslationEdit(body_name=name, translation=shift)
            for name in (parent, child)
        ],
        reasoning=Reasoning(
            explanation="Move the fixed pair while preserving its spacing."
        ),
    )
    agent_service.scope.body_names = [parent, child]
    before = project.history.load()
    proposal = agent_service.submit_change_set(request=request)
    assert project.history.load() == before
    decision = ChangeSetDecisionQuery(
        identifier=proposal.identifier, expected_revision=revision, accept=True
    )
    agent_service.scope.body_names = [parent]
    with pytest.raises(InvalidEvidence):
        agent_service.decide_change_set(request=decision)
    assert project.history.load() == before
    agent_service.scope.body_names = [parent, child]
    result = agent_service.decide_change_set(request=decision)
    assert len(project.history.load().past) == len(before.past) + 1
    for original, actual in zip(snapshot.meshes, project.effective_snapshot.meshes):
        np.testing.assert_allclose(
            actual.mesh.vertices,
            original.mesh.vertices
            + (
                np.array(shift)
                if original.body_name in (parent, child)
                else np.zeros(3)
            ),
        )
    agent_service.undo_action(
        request=HistoryQuery(
            action_id=result.action.identifier, expected_revision=agent_service.revision
        )
    )
    assert project.current_state == before.current
    (review,) = agent_service.list_change_sets(request=ProjectQuery()).reviews
    assert review.status == ChangeSetStatus.UNDONE
    with pytest.raises(InvalidEvidence):
        agent_service.decide_change_set(request=decision)
    agent_service.redo_action(
        request=HistoryQuery(
            action_id=result.action.identifier, expected_revision=agent_service.revision
        )
    )
    assert agent_service.revision == result.action.after_revision


@pytest.fixture
def agent_server(agent_service: AgentService) -> Iterator[InspectionServer]:
    """
    A listener owns one shared worker manager for both browser and agent requests.
    """
    with InspectionService(
        project=agent_service.project, agent_scope=agent_service.scope
    ).create_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=10)


def test_investigation_uses_browser_revision_and_rejects_stale_context(
    agent_server: InspectionServer,
) -> None:
    """
    Copied oriented bounds are reusable only against the observed revision.
    """
    url = f"http://127.0.0.1:{agent_server.server_port}"
    client = AgentClient(base_url=url)
    with urlopen(url + "/api/state", timeout=5) as response:
        revision = json.load(response)["revision"]
    query = RegionQuery(
        expected_revision=revision,
        candidate_index=0,
        lower=[0, 0, 0],
        upper=[1, 2, 3],
        frame_T_scan=[[0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]],
    )
    stored = client.create_region(request=query)
    assert stored.query == query
    assert (
        stored.created_revision
        == client.inspect_project(request=ProjectQuery()).revision
    )
    agent_server.agent.project.accept(candidate_index=0)
    with pytest.raises(InvalidEvidence):
        client.create_region(request=query)


def test_rejected_change_set_stays_declined_over_the_shared_transport(
    agent_server: InspectionServer,
) -> None:
    """
    A decline updates the proposal queue without changing the model or journal.
    """
    client = AgentClient(base_url=f"http://127.0.0.1:{agent_server.server_port}")
    revision = client.inspect_project(request=ProjectQuery()).revision
    proposal = client.submit_change_set(
        request=ChangeSetQuery(
            expected_revision=revision,
            edits=[TranslationEdit(body_name="test/box", translation=[0.01, 0, 0])],
            reasoning=Reasoning(explanation="Inspect a proposed movement."),
        )
    )
    decision = ChangeSetDecisionQuery(
        identifier=proposal.identifier, expected_revision=revision, accept=False
    )
    before = agent_server.agent.project.history.load()
    client.decide_change_set(request=decision)
    (record,) = client.list_change_sets(request=ProjectQuery()).reviews
    assert record.status == ChangeSetStatus.REJECTED
    decision.accept = True
    with pytest.raises(InvalidEvidence):
        client.decide_change_set(request=decision)
    assert agent_server.agent.project.history.load() == before


def test_change_set_heatmap_colours_the_same_records_without_applying(
    agent_server: InspectionServer,
) -> None:
    """
    Both binary distance arrays describe the same records at the submitted placement.
    """
    url = f"http://127.0.0.1:{agent_server.server_port}"
    client = AgentClient(base_url=url)
    revision = client.inspect_project(request=ProjectQuery()).revision
    region = client.create_region(request=RegionQuery(body_name="test/box"))
    proposal = client.submit_change_set(
        request=ChangeSetQuery(
            expected_revision=revision,
            region_id=region.identifier,
            edits=[TranslationEdit(body_name="test/box", translation=[0.02, 0, 0])],
            reasoning=Reasoning(explanation="Correct the known translation."),
        )
    )
    before = agent_server.agent.project.history.load()
    job = client.compute_change_set_heatmap(
        request=ChangeSetHeatmapQuery(
            identifier=proposal.identifier,
            expected_revision=revision,
            candidate_index=0,
        )
    )
    agent_server.jobs.processes[job.identifier].wait(timeout=30)
    for path in ("distances", "proposed-distances"):
        with urlopen(f"{url}/api/jobs/{job.identifier}/{path}", timeout=5) as response:
            distances = np.frombuffer(response.read(), dtype="<f4")
        assert len(distances) == 5000
        if path == "distances":
            np.testing.assert_allclose(distances.max(), 0.02, atol=1e-6)
        else:
            np.testing.assert_allclose(distances, 0, atol=1e-6)
    assert agent_server.agent.project.history.load() == before


def test_http_agent_client_shares_server_jobs_and_journal(
    agent_server: InspectionServer,
) -> None:
    """
    Typed transport preserves region identities and delegates to the UI's worker owner.
    """
    client = AgentClient(base_url=f"http://127.0.0.1:{agent_server.server_port}")
    assert agent_server.agent.jobs is agent_server.jobs
    summary = client.inspect_project(request=ProjectQuery())
    assert summary.revision == agent_server.agent.revision
    region = client.create_region(request=RegionQuery(body_name="test/box"))
    sample = client.inspect_region(
        request=RegionSampleQuery(
            region_id=region.identifier,
            resolution=EvidenceResolution.PREVIEW,
            maximum_points=1000,
        )
    )
    assert sample.region.identifier == region.identifier
    assert (
        client.find_bodies(request=BodyQuery(search="box")).bodies[0].body_name
        == "test/box"
    )


def test_cli_reads_the_same_project_revision(agent_server: InspectionServer) -> None:
    """
    Shell automation uses the running service rather than becoming a second writer.
    """
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "agent_cli.py"),
        "inspect_project",
        "--url",
        f"http://127.0.0.1:{agent_server.server_port}",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=30
    )
    summary = DataclassJSONSerializer.from_json(
        data=json.loads(result.stdout), clazz=ProjectSummary
    )
    assert summary.revision == agent_server.agent.revision


def test_codec_refuses_unknown_types_and_ignored_fields() -> None:
    """
    Transport deserialization cannot select executable classes or silently drop
    constraints.
    """
    codec = AgentCodec()
    payload = DataclassJSONSerializer.to_json(instance=ProjectQuery())
    payload["unexpected_constraint"] = "center_on"
    with pytest.raises(InvalidEvidence):
        codec.decode_request(
            payload=json.dumps(payload).encode(), request_type=ProjectQuery
        )
    payload = {"__json_type__": "os.system", "command": "ignored"}
    with pytest.raises(InvalidEvidence):
        codec.decode_request(
            payload=json.dumps(payload).encode(), request_type=ProjectQuery
        )


# %% Serialization enum validation
def test_codec_rejects_unknown_enum_members() -> None:
    """
    Invalid enum values are request errors before repository deserialization.
    """
    from scan_refiner.agent.models import AgentMode

    payload = {
        "__json_type__": AgentMode.__module__ + "." + AgentMode.__qualname__,
        "name": "UNSUPPORTED",
    }
    with pytest.raises(InvalidEvidence):
        AgentCodec().validate_type_tags(
            value=payload, known={payload["__json_type__"]: AgentMode}
        )


# %% Readable tool refusals
def test_http_client_receives_stale_revision_error(
    agent_server: InspectionServer,
) -> None:
    """
    A declined action remains a structured error over the shared connection.
    """
    client = AgentClient(base_url=f"http://127.0.0.1:{agent_server.server_port}")
    with pytest.raises(InvalidEvidence):
        client.apply_proposal(
            request=ApplyQuery(
                job_id="0" * 32,
                expected_revision="stale",
                reasoning=Reasoning(explanation="Try obsolete revision"),
            )
        )
    assert (
        client.inspect_project(request=ProjectQuery()).revision
        == agent_server.agent.revision
    )


# %% Missing opaque evidence
def test_unknown_region_is_a_readable_request_error(
    agent_service: AgentService,
) -> None:
    """
    A well-formed unknown identifier must not escape as an incidental filesystem error.
    """
    with pytest.raises(InvalidEvidence):
        agent_service.inspect_region(
            request=RegionSampleQuery(
                region_id="0" * 32, resolution=EvidenceResolution.PREVIEW
            )
        )


# %% Unapplied registration candidates
def test_render_can_inspect_a_new_job_candidate_before_registration_is_applied(
    agent_service: AgentService,
) -> None:
    """
    Candidate identities in a new job are independent of the current placement table.
    """
    region = agent_service.create_region(request=RegionQuery(body_name="test/box"))
    registration = RegistrationResult(
        candidates=[
            RegistrationCandidate(
                scan_T_root=np.eye(4).tolist(), support_fraction=1, median_distance=0
            ),
            RegistrationCandidate(
                scan_T_root=np.eye(4).tolist(),
                support_fraction=0.9,
                median_distance=0.01,
            ),
        ]
    )
    record = JobRecord(
        identifier="1" * 32,
        request=JobRequest(kind="registration"),
        baseline_signature=agent_service.jobs.baseline.signature,
        scan_T_root=np.eye(4).tolist(),
        status=JobStatus.SUCCEEDED,
    )
    inspection = JobInspection(record=record, registration=registration)
    with patch.object(agent_service, "inspect_job", return_value=inspection):
        artifact = agent_service.render_region(
            request=RegionRenderQuery(
                region_id=region.identifier,
                placement_job=record.identifier,
                candidate_index=1,
                resolution=EvidenceResolution.PREVIEW,
                image=False,
                maximum_points=1000,
            )
        )
    assert Path(artifact.html_path).is_file()
    assert len(agent_service.project.registration.candidates) == 1
