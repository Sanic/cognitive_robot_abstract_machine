"""
An agent correction can leave the service as a validated native-source bundle.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import threading
import numpy as np
import trimesh
from semantic_digital_twin.adapters.urdf import URDFParser
from scan_refiner.agent.client import AgentClient
from scan_refiner.agent.models import (
    AgentScope,
    AgentMode,
    ProjectQuery,
    RegionQuery,
    RefinementQuery,
    Reasoning,
    JobQuery,
    ApplyQuery,
    HistoryQuery,
    SourceQuery,
    ProposalQuery,
)
from scan_refiner.jobs import JobStatus
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole, RegistrationCandidate, RegistrationResult
from scan_refiner.refinement import TranslationConstraints
from scan_refiner.source_application import SourceArtifact
from scan_refiner.web_service import InspectionService
from scan_refiner.world import WorldSnapshot


# %% Native source round trip
def test_agent_correction_exports_validated_urdf_without_editing_original(
    tmp_path: Path,
) -> None:
    """
    Typed HTTP actions share the journal and export only accepted, reloadable geometry.
    """
    source = Path(__file__).parent / "dataset/source_world.urdf"
    original = source.read_bytes()
    snapshot = WorldSnapshot.from_world(
        world=URDFParser.from_file(file_path=str(source)).create_world()
    )
    body = next(
        body
        for body in snapshot.meshes
        if body.body_name == "test/box" and body.role == GeometryRole.VISUAL
    )
    points, _ = trimesh.sample.sample_surface(body.mesh, 5000, seed=19)
    points[:, 0] += 0.02
    scan = tmp_path / "scan.npz"
    np.savez(scan, points=points, colors=np.full(points.shape, 128, dtype=np.uint8))
    project = RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=scan,
        snapshot=snapshot,
        model_source=str(source.absolute()),
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
    scope = AgentScope(mode=AgentMode.REFINE, body_names=["test/box"])
    with InspectionService(project=project, agent_scope=scope).create_server(
        port=0
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = AgentClient(base_url=f"http://127.0.0.1:{server.server_port}")
            region = client.create_region(request=RegionQuery(body_name="test/box"))
            job = client.propose_refinement(
                request=RefinementQuery(
                    expected_revision=client.inspect_project(
                        request=ProjectQuery()
                    ).revision,
                    region_id=region.identifier,
                    body_name="test/box",
                    reasoning=Reasoning(
                        explanation="Fit the known observable X discrepancy"
                    ),
                    translation=TranslationConstraints(axes=[True, False, False]),
                    maximum_points=5000,
                )
            )
            server.jobs.processes[job.identifier].wait(timeout=30)
            assert (
                client.inspect_job(
                    request=JobQuery(job_id=job.identifier)
                ).record.status
                == JobStatus.SUCCEEDED
            )
            comparison = client.compare_proposal(
                request=ProposalQuery(job_id=job.identifier)
            )
            np.testing.assert_allclose(
                comparison.inspection.proposal.translation, [0.02, 0, 0], atol=0.002
            )
            action = client.apply_proposal(
                request=ApplyQuery(
                    job_id=job.identifier,
                    expected_revision=client.inspect_project(
                        request=ProjectQuery()
                    ).revision,
                    reasoning=Reasoning(
                        explanation="Apply the supported X correction after comparing the proposal"
                    ),
                )
            )
            client.undo_action(
                request=HistoryQuery(
                    action_id=action.identifier, expected_revision=action.after_revision
                )
            )
            client.redo_action(
                request=HistoryQuery(
                    action_id=action.identifier,
                    expected_revision=action.before_revision,
                )
            )
            review = client.review_source(
                request=SourceQuery(expected_revision=action.after_revision)
            )
            result = client.export_source(
                request=SourceQuery(
                    expected_revision=action.after_revision, review_id=review.identifier
                )
            )
            assert result.validated is True
            assert (Path(result.directory) / SourceArtifact.URDF).is_file()
            assert (Path(result.directory).parent / SourceArtifact.ARCHIVE).is_file()
            assert source.read_bytes() == original
        finally:
            server.shutdown()
            thread.join(timeout=10)
