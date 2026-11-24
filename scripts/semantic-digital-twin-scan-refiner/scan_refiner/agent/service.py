"""
Revision-checked agent operations over the refiner's existing workers and journal.
"""

from __future__ import annotations

# %% Imports
from scan_refiner.agent.interfaces import AgentOperations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from uuid import uuid4
import numpy as np
import trimesh
from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood.adapters.json_field import JSONField
from scan_refiner.agent.rendering import AgentRenderer
from scan_refiner.agent.models import (
    AgentScope,
    AgentMode,
    ProjectQuery,
    ProjectSummary,
    BodyQuery,
    BodySummary,
    BodySearchResult,
    RegionQuery,
    EvidenceRegion,
    RegionSampleQuery,
    SampleSummary,
    EvidenceResolution,
    RegionRenderQuery,
    RenderArtifact,
    RegistrationQuery,
    RefinementQuery,
    ChangeSet,
    ChangeSetQuery,
    ChangeSetDecisionQuery,
    ChangeSetList,
    ChangeSetResult,
    ChangeSetHeatmapQuery,
    ChangeSetReview,
    ChangeSetStatus,
    JobQuery,
    JobInspection,
    ProposalQuery,
    ProposalComparison,
    ApplyQuery,
    AcceptanceQuery,
    HistoryQuery,
    SourceQuery,
    AgentAction,
    AgentOperation,
    AgentFile,
    Reasoning,
    AgentRevision,
    AgentJobNote,
    ComparisonQuery,
    AgentRevisionState,
)
from scan_refiner.history import ActionHistory, ProjectState
from scan_refiner.project import RefinementProject
from scan_refiner.jobs import (
    JobManager,
    JobRequest,
    JobRecord,
    JobKind,
    JobStatus,
    JobFile,
)
from scan_refiner.job_worker import JobWorker
from scan_refiner.records import GeometryRole, RegistrationResult, ProjectFile
from scan_refiner.dimensions import (
    BodyDimensionInfo,
    DimensionFrame,
    DimensionProposal,
    BodyDimensions,
)
from scan_refiner.frames import OrientedRegion, CoordinateFrame
from scan_refiner.refinement import TranslationProposal, BodyTranslation
from scan_refiner.world import WorldSnapshot
from scan_refiner.coverage import ComparisonReport, ComparisonArray
from scan_refiner.spatial_index import IndexFile, SpatialScanIndex
from scan_refiner.point_records import PointField
from scan_refiner.scan_sources import PreviewPointSource
from scan_refiner.source_application import (
    SourceApplication,
    SourceReview,
    SourceResult,
    WorldStructure,
)
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.source_service import SourceProcess


# %% Shared service
@dataclass(kw_only=True)
class AgentService(AgentOperations):
    """
    Typed operations sharing the local UI's project, worker owner and action lock.
    """

    project: RefinementProject
    """
    Authoritative project; derived agent artifacts never replace its baselines.
    """

    jobs: JobManager
    """
    The existing service-owned worker manager, rather than a competing worker pool.
    """

    scope: AgentScope
    """
    Server-defined capabilities and movement limits.
    """

    browser_executable: str | None = None
    """
    Existing Chromium executable used only when a PNG is explicitly requested.
    """

    @property
    def revision(self) -> str:
        """
        Fingerprint the source and default placement as well as worker geometry inputs.
        """
        state = self.project.current_state
        revision = AgentRevision(
            geometry_signature=self.jobs.baseline.signature,
            model_source=state.record.model_source,
            accepted_candidate=state.record.accepted_candidate,
        )
        return hashlib.sha256(
            json.dumps(
                DataclassJSONSerializer.to_json(instance=revision), sort_keys=True
            ).encode()
        ).hexdigest()

    def revision_state(self) -> AgentRevisionState:
        """
        Expose a small polling notification without rereading mesh or scan geometry.
        """
        return AgentRevisionState(revision=self.revision, mode=self.scope.mode)

    def inspect_project(self, *, request: ProjectQuery) -> ProjectSummary:
        state = self.project.current_state
        action_directory = (
            self.project.directory / AgentFile.DIRECTORY / AgentFile.ACTIONS
        )
        paths = sorted(
            action_directory.glob("*/record.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:10]
        actions = [
            DataclassJSONSerializer.from_json(
                data=json.loads(path.read_text()), clazz=AgentAction
            )
            for path in paths
        ]
        return ProjectSummary(
            revision=self.revision,
            model_source=state.record.model_source,
            scan_source=state.record.scan_source,
            dense_index=(
                self.project.directory / ProjectFile.SCAN_INDEX / IndexFile.RECORD
            ).exists(),
            body_count=len({mesh.body_name for mesh in self.project.snapshot.meshes}),
            scope=deepcopy(self.scope),
            accepted_candidate=state.record.accepted_candidate,
            candidate_count=len(state.registration.candidates),
            recent_actions=actions,
        )

    def find_bodies(self, *, request: BodyQuery) -> BodySearchResult:
        if not 1 <= request.limit <= 200:
            raise InvalidEvidence(reason="Body search limit must be 1–200.")
        snapshot = self.project.effective_snapshot
        structure = None
        if request.include_source:
            world = SourceApplication.load_world(
                identifier=self.project.record.model_source
            )
            SourceApplication.check_geometry(
                expected=self.project.snapshot,
                actual=WorldSnapshot.from_world(world=world),
            )
            structure = WorldStructure.capture(world=world)
        names = {mesh.body_name for mesh in snapshot.meshes}
        if structure:
            names.update(structure.frame_names)
        semantic_matches = (
            {
                body
                for annotation in structure.annotations
                if request.search.lower() in annotation.annotation_type.lower()
                for body in annotation.body_names
            }
            if structure
            else set()
        )
        names = sorted(
            name
            for name in names
            if request.search.lower() in name.lower() or name in semantic_matches
        )
        bodies = []
        for name in names[: request.limit]:
            surfaces = [mesh for mesh in snapshot.meshes if mesh.body_name == name]
            roles = sorted({str(mesh.role) for mesh in surfaces})
            bounds = np.array([mesh.mesh.bounds for mesh in surfaces])
            role = (
                GeometryRole.VISUAL
                if GeometryRole.VISUAL in roles
                else GeometryRole.COLLISION
            )
            problem = (
                BodyDimensionInfo.inspect(
                    snapshot=snapshot,
                    body_name=name,
                    role=str(role),
                    frame_mode=DimensionFrame.BOX.value,
                ).problem
                if surfaces
                else "This source frame has no geometry."
            )
            incoming = (
                next(
                    (
                        joint
                        for joint in structure.connections
                        if joint.child_name == name
                    ),
                    None,
                )
                if structure
                else None
            )
            semantics = (
                [
                    annotation.annotation_type
                    for annotation in structure.annotations
                    if name in annotation.body_names
                ]
                if structure
                else []
            )
            bodies.append(
                BodySummary(
                    body_name=name,
                    roles=roles,
                    lower=bounds[:, 0, :].min(axis=0).tolist() if surfaces else [],
                    upper=bounds[:, 1, :].max(axis=0).tolist() if surfaces else [],
                    box_problem=problem,
                    parent_name=incoming.parent_name if incoming else None,
                    joint_type=incoming.joint_type if incoming else None,
                    semantic_types=semantics,
                )
            )
        return BodySearchResult(bodies=bodies, matched_count=len(names))

    def create_region(self, *, request: RegionQuery) -> EvidenceRegion:
        if request.expected_revision is not None:
            self.check_revision(expected=request.expected_revision)
        query = deepcopy(request)
        frame = CoordinateFrame(frame_T_scan=np.array(query.frame_T_scan))
        if not np.isfinite(query.margin_m) or query.margin_m < 0:
            raise InvalidEvidence(
                reason="Region context margin must be finite and nonnegative."
            )
        if query.lower or query.upper:
            if query.body_name:
                raise InvalidEvidence(
                    reason="Choose either explicit bounds or body-derived bounds."
                )
        elif query.body_name:
            candidate = self.select_candidate(candidate=query.candidate_index)
            placement = (
                np.eye(4)
                if candidate is None
                else np.array(
                    self.project.registration.candidates[candidate].scan_T_root
                )
            )
            surfaces = [
                mesh.mesh
                for mesh in self.project.effective_snapshot.meshes
                if mesh.body_name == query.body_name
            ]
            if not surfaces:
                raise InvalidEvidence(reason="Choose an existing geometric body.")
            points = trimesh.transform_points(
                np.concatenate([mesh.vertices for mesh in surfaces]),
                frame.frame_T_scan @ placement,
            )
            query.lower = (points.min(axis=0) - query.margin_m).tolist()
            query.upper = (points.max(axis=0) + query.margin_m).tolist()
            query.candidate_index = candidate
        else:
            raise InvalidEvidence(reason="Supply metric bounds or one existing body.")
        OrientedRegion(
            lower=np.array(query.lower), upper=np.array(query.upper), frame=frame
        )
        region = EvidenceRegion(
            identifier=uuid4().hex,
            query=query,
            scan_sha256=self.project.record.scan_sha256,
            created_revision=self.revision,
        )
        directory = self.artifact_directory(
            kind=AgentFile.REGIONS, identifier=region.identifier
        )
        directory.mkdir(parents=True)
        (directory / AgentFile.RECORD).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=region), indent=2)
        )
        return region

    def inspect_region(self, *, request: RegionSampleQuery) -> SampleSummary:
        if not 1 <= request.maximum_points <= self.scope.maximum_points:
            raise InvalidEvidence(
                reason="Choose a point budget within the server limit."
            )
        region = self.read_region(identifier=request.region_id)
        key = hashlib.sha256(
            f"{request.maximum_points}:{request.resolution}".encode()
        ).hexdigest()[:16]
        path = self.artifact_directory(
            kind=AgentFile.REGIONS, identifier=region.identifier
        ) / (key + ".npz")
        if not path.exists():
            box = OrientedRegion(
                lower=np.array(region.query.lower),
                upper=np.array(region.query.upper),
                frame=CoordinateFrame(frame_T_scan=np.array(region.query.frame_T_scan)),
            )
            if request.resolution == EvidenceResolution.DENSE:
                index = SpatialScanIndex(
                    directory=self.project.directory / ProjectFile.SCAN_INDEX
                )
                if not (index.directory / IndexFile.RECORD).exists():
                    raise InvalidEvidence(
                        reason="No dense index is attached; request preview evidence explicitly."
                    )
                records = box.sample(
                    index=index, maximum_count=request.maximum_points
                ).records
            else:
                preview = self.project.scan
                indices = preview.sample_indices(
                    spacing=0.01, maximum_count=request.maximum_points
                )
                records = next(
                    PreviewPointSource(
                        preview=preview, batch_capacity=len(preview.points)
                    ).batches()
                ).records[indices]
                records = records[box.contains(scan_points=records[PointField.POINTS])]
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, records=records)
            temporary.replace(path)
        with np.load(path, allow_pickle=False) as stored:
            count = len(stored[ComparisonArray.RECORDS])
        return SampleSummary(
            region=region,
            count=count,
            requested_count=request.maximum_points,
            resolution=request.resolution,
            evidence_path=str(path.absolute()),
        )

    def render_region(self, *, request: RegionRenderQuery) -> RenderArtifact:
        registration = self.project.registration
        candidate = None
        if request.placement_job:
            inspection = self.inspect_job(
                request=JobQuery(job_id=request.placement_job)
            )
            if inspection.registration is None:
                raise InvalidEvidence(
                    reason="Choose a current successful registration job for temporary placement review."
                )
            registration = inspection.registration
            candidate = (
                request.candidate_index if request.candidate_index is not None else 0
            )
            if not 0 <= candidate < len(registration.candidates):
                raise InvalidEvidence(
                    reason="Choose an existing candidate from this registration job."
                )
        else:
            candidate = self.select_candidate(candidate=request.candidate_index)
        if request.comparison_job:
            inspection = self.inspect_job(
                request=JobQuery(job_id=request.comparison_job)
            )
            record = inspection.record
            if inspection.report is None:
                raise InvalidEvidence(
                    reason="Choose a current successful comparison job."
                )
            region = self.read_region(identifier=request.region_id)
            if (
                region.query.lower != record.request.lower
                or region.query.upper != record.request.upper
                or region.query.frame_T_scan != record.request.frame_T_scan
            ):
                raise InvalidEvidence(
                    reason="The image region must match the comparison evidence crop and frame."
                )
            if request.geometry_role != record.request.geometry_role:
                raise InvalidEvidence(
                    reason="Render the geometry role used by this comparison."
                )
            path = self.jobs.result_path(identifier=record.identifier)
            with np.load(path, allow_pickle=False) as arrays:
                count = len(arrays[ComparisonArray.RECORDS])
            sample = SampleSummary(
                region=region,
                count=count,
                requested_count=record.request.maximum_count,
                resolution=EvidenceResolution.DENSE
                if self.jobs.baseline.index_identity
                else EvidenceResolution.PREVIEW,
                evidence_path=str(path.absolute()),
            )
            candidate = record.request.candidate_index
        else:
            sample = self.inspect_region(
                request=RegionSampleQuery(
                    region_id=request.region_id,
                    maximum_points=request.maximum_points,
                    resolution=request.resolution,
                )
            )
        return AgentRenderer(
            project=self.project, browser_executable=self.browser_executable
        ).render_evidence(
            query=request,
            sample=sample,
            snapshot=self.project.effective_snapshot,
            registration=registration,
            candidate=candidate,
            revision=self.revision,
        )

    def propose_comparison(self, *, request: ComparisonQuery) -> JobRecord:
        self.check_revision(expected=request.expected_revision)
        self.check_reasoning(reasoning=request.reasoning)
        if not 1 <= request.maximum_points <= self.scope.maximum_points:
            raise InvalidEvidence(reason="Point budget exceeds the server scope.")
        region = self.read_region(identifier=request.region_id)
        candidate = self.select_candidate(candidate=request.candidate_index)
        if candidate is None:
            raise InvalidEvidence(
                reason="Choose a registration candidate before comparing geometry."
            )
        record = self.jobs.start(
            request=JobRequest(
                kind=JobKind.HEATMAP.value,
                candidate_index=candidate,
                geometry_role=request.geometry_role,
                body_name=request.body_name,
                lower=region.query.lower,
                upper=region.query.upper,
                frame_T_scan=region.query.frame_T_scan,
                maximum_count=request.maximum_points,
                model_sample_count=request.model_samples,
                tolerance=request.tolerance_m,
            )
        )
        self.save_job_reasoning(
            record=record, reasoning=request.reasoning, region_id=region.identifier
        )
        return record

    def compare_proposal(self, *, request: ProposalQuery) -> ProposalComparison:
        inspection = self.inspect_job(request=JobQuery(job_id=request.job_id))
        if inspection.proposal is None:
            raise InvalidEvidence(
                reason="Choose a current successful translation or dimension proposal."
            )
        result = ProposalComparison(inspection=inspection)
        if not request.image or not inspection.proposal.matched_points:
            return result
        record = inspection.record
        region = (
            self.read_region(identifier=inspection.region_id)
            if inspection.region_id
            else self.create_region(
                request=RegionQuery(
                    lower=record.request.lower,
                    upper=record.request.upper,
                    frame_T_scan=record.request.frame_T_scan,
                )
            )
        )
        path = self.jobs.result_path(identifier=record.identifier)
        sample = SampleSummary(
            region=region,
            count=inspection.proposal.matched_points,
            requested_count=record.request.maximum_count,
            resolution=EvidenceResolution.DENSE
            if self.jobs.baseline.index_identity
            else EvidenceResolution.PREVIEW,
            evidence_path=str(path.absolute()),
        )
        before = JobWorker(
            project=self.project,
            directory=self.jobs.directory(identifier=record.identifier),
        ).load_geometry(record=record)
        after = JobWorker(
            project=self.project,
            directory=self.jobs.directory(identifier=record.identifier),
        ).load_geometry(record=record)
        proposal = inspection.proposal
        if isinstance(proposal, TranslationProposal):
            after.apply_translations(
                overrides=[
                    BodyTranslation(
                        body_name=proposal.body_name, translation=proposal.translation
                    )
                ]
            )
        else:
            after.apply_dimensions(
                overrides=[
                    BodyDimensions(
                        body_name=proposal.body_name,
                        scale=proposal.scale,
                        offset=proposal.offset,
                        matrix=proposal.matrix,
                    )
                ]
            )
        render = RegionRenderQuery(
            region_id=region.identifier,
            view=request.view,
            maximum_points=min(record.request.maximum_count, 20000),
            resolution=sample.resolution,
            geometry_role=record.request.geometry_role,
            image=True,
        )
        renderer = AgentRenderer(
            project=self.project, browser_executable=self.browser_executable
        )
        result.before = renderer.render_evidence(
            query=render,
            sample=sample,
            snapshot=before,
            registration=self.project.registration,
            candidate=record.request.candidate_index,
            revision=self.revision,
        )
        result.after = renderer.render_evidence(
            query=render,
            sample=sample,
            snapshot=after,
            registration=self.project.registration,
            candidate=record.request.candidate_index,
            revision=self.revision,
        )
        return result

    def propose_registration(self, *, request: RegistrationQuery) -> JobRecord:
        self.check_revision(expected=request.expected_revision)
        self.check_reasoning(reasoning=request.reasoning)
        record = self.jobs.start(
            request=JobRequest(
                kind=JobKind.REGISTRATION.value,
                upright=request.upright,
                geometry_role=request.geometry_role,
                search=request.search,
            )
        )
        self.save_job_reasoning(record=record, reasoning=request.reasoning)
        return record

    def propose_refinement(self, *, request: RefinementQuery) -> JobRecord:
        self.check_revision(expected=request.expected_revision)
        self.check_body_scope(body_name=request.body_name)
        self.check_reasoning(reasoning=request.reasoning)
        if request.operation not in ("translation", "dimensions"):
            raise InvalidEvidence(
                reason="Only translation and verified box dimensions are implemented; shared/semantic constraints must not be assumed."
            )
        if not 1 <= request.maximum_points <= self.scope.maximum_points:
            raise InvalidEvidence(reason="Point budget exceeds the agent scope.")
        if (
            request.operation == "translation"
            and request.translation.maximum_translation
            > self.scope.maximum_translation_m
        ):
            raise InvalidEvidence(
                reason="The proposed movement limit exceeds the server scope."
            )
        if (
            request.operation == "dimensions"
            and request.dimensions.maximum_change > self.scope.maximum_extent_change_m
        ):
            raise InvalidEvidence(
                reason="The proposed extent-change limit exceeds the server scope."
            )
        region = self.read_region(identifier=request.region_id)
        candidate = self.select_candidate(candidate=request.candidate_index)
        if candidate is None:
            raise InvalidEvidence(
                reason="Choose a registration candidate before fitting geometry."
            )
        job = JobRequest(
            kind=JobKind.REFINEMENT.value
            if request.operation == "translation"
            else JobKind.DIMENSIONS.value,
            candidate_index=candidate,
            body_name=request.body_name,
            geometry_role=request.geometry_role,
            lower=region.query.lower,
            upper=region.query.upper,
            frame_T_scan=region.query.frame_T_scan,
            maximum_count=request.maximum_points,
            tolerance=request.tolerance_m,
            constraints=request.translation,
            dimension_constraints=request.dimensions,
        )
        record = self.jobs.start(request=job)
        self.save_job_reasoning(
            record=record, reasoning=request.reasoning, region_id=region.identifier
        )
        return record

    def inspect_job(self, *, request: JobQuery) -> JobInspection:
        record = self.jobs.get(identifier=request.job_id)
        note = self.jobs.directory(identifier=request.job_id) / AgentFile.JOB_NOTE
        annotation = (
            DataclassJSONSerializer.from_json(
                data=json.loads(note.read_text()), clazz=AgentJobNote
            )
            if note.exists()
            else None
        )
        result = JobInspection(
            record=record,
            current_revision=self.revision,
            region_id=annotation.region_id if annotation else "",
            reasoning=annotation.reasoning if annotation else None,
        )
        if record.status != JobStatus.SUCCEEDED:
            return result
        if record.request.kind in (JobKind.REFINEMENT, JobKind.DIMENSIONS):
            result.proposal = self.jobs.read_proposal(identifier=request.job_id)
        elif record.request.kind == JobKind.REGISTRATION:
            result.registration = DataclassJSONSerializer.from_json(
                data=json.loads(
                    (
                        self.jobs.directory(identifier=request.job_id)
                        / JobFile.CANDIDATES
                    ).read_text()
                ),
                clazz=RegistrationResult,
            )
        elif record.request.kind == JobKind.HEATMAP:
            result.report = DataclassJSONSerializer.from_json(
                data=json.loads(
                    (
                        self.jobs.directory(identifier=request.job_id) / JobFile.REPORT
                    ).read_text()
                ),
                clazz=ComparisonReport,
            )
        return result

    def cancel_job(self, *, request: JobQuery) -> JobRecord:
        if not (
            self.jobs.directory(identifier=request.job_id) / AgentFile.JOB_NOTE
        ).is_file():
            raise InvalidEvidence(
                reason="Agent cancellation is limited to jobs requested through the agent interface."
            )
        return self.jobs.cancel(identifier=request.job_id)

    def apply_proposal(self, *, request: ApplyQuery) -> AgentAction:
        self.check_revision(expected=request.expected_revision)
        self.check_reasoning(reasoning=request.reasoning)
        if self.scope.mode != AgentMode.REFINE:
            raise InvalidEvidence(
                reason="This agent connection is inspect-only; proposal application is disabled."
            )
        record = self.jobs.get(identifier=request.job_id)
        if record.request.kind == JobKind.REGISTRATION:
            if not self.scope.allow_registration:
                raise InvalidEvidence(
                    reason="Registration changes are outside the granted agent scope."
                )
        else:
            self.check_body_scope(body_name=record.request.body_name)
            proposal = self.jobs.read_proposal(identifier=request.job_id)
            self.check_proposal_scope(proposal=proposal)
        before = self.history_signature()
        before_revision = self.revision
        self.jobs.apply(identifier=request.job_id)
        return self.record_action(
            operation=AgentOperation.APPLY,
            reasoning=request.reasoning,
            job_id=request.job_id,
            before_revision=before_revision,
            before_state=before,
        )

    # %% Grouped proposal review
    def submit_change_set(self, *, request: ChangeSetQuery) -> ChangeSet:
        self.check_revision(expected=request.expected_revision)
        self.check_reasoning(reasoning=request.reasoning)
        self.check_change_set_scope(request=request)
        names = {mesh.body_name for mesh in self.project.snapshot.meshes}
        if any(edit.body_name not in names for edit in request.edits):
            raise InvalidEvidence(
                reason="Every edited body must exist in the frozen model."
            )
        evidence = (
            self.read_region(identifier=request.region_id)
            if request.region_id
            else None
        )
        if (
            evidence is not None
            and evidence.created_revision != request.expected_revision
        ):
            raise InvalidEvidence(
                reason="Reopen the evidence crop against the current revision."
            )
        proposal = ChangeSet(identifier=uuid4().hex, request=request, evidence=evidence)
        self.save_change_set(proposal=proposal)
        return proposal

    def list_change_sets(self, *, request: ProjectQuery) -> ChangeSetList:
        revision = self.revision
        directory = self.project.directory / AgentFile.DIRECTORY / AgentFile.CHANGE_SETS
        paths = sorted(
            directory.glob(f"*/{AgentFile.RECORD}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        history = self.project.history.load()
        applied = {
            item.body_name: item.source_job for item in history.current.overrides
        }
        undone = self.redo_proposal_ids(history=history)
        return ChangeSetList(
            revision=revision,
            can_decide=self.scope.mode == AgentMode.REFINE,
            reviews=[
                self.change_set_review(
                    proposal=self.read_change_set(identifier=path.parent.name),
                    revision=revision,
                    applied=applied,
                    undone=undone,
                )
                for path in paths[:30]
            ],
        )

    def decide_change_set(self, *, request: ChangeSetDecisionQuery) -> ChangeSetResult:
        self.check_revision(expected=request.expected_revision)
        if self.scope.mode != AgentMode.REFINE:
            raise InvalidEvidence(
                reason="This connection is inspect-only; change-set decisions require refine mode."
            )
        proposal = self.read_change_set(identifier=request.identifier)
        self.check_revision(expected=proposal.request.expected_revision)
        if proposal.rejected:
            raise InvalidEvidence(reason="This proposal has already been rejected.")
        history = self.project.history.load()
        if proposal.identifier in self.redo_proposal_ids(history=history):
            raise InvalidEvidence(
                reason="This change set was undone. Use Redo to restore it."
            )
        self.check_change_set_scope(request=proposal.request)
        if not request.accept:
            proposal.rejected = True
            self.save_change_set(proposal=proposal)
            return ChangeSetResult(
                review=ChangeSetReview(
                    proposal=proposal, status=ChangeSetStatus.REJECTED
                )
            )
        before = self.state_signature(state=history.current)
        self.project.apply_translations(
            edits=[
                BodyTranslation(
                    body_name=edit.body_name,
                    translation=edit.translation,
                    source_job=proposal.identifier,
                )
                for edit in proposal.request.edits
            ],
            label=f"Apply change set ({len(proposal.request.edits)} bodies)",
        )
        action = self.record_action(
            operation=AgentOperation.DECIDE_CHANGE_SET,
            reasoning=proposal.request.reasoning,
            job_id=proposal.identifier,
            before_revision=request.expected_revision,
            before_state=before,
        )
        return ChangeSetResult(
            review=ChangeSetReview(proposal=proposal, status=ChangeSetStatus.APPLIED),
            action=action,
        )

    def compute_change_set_heatmap(
        self, *, request: ChangeSetHeatmapQuery
    ) -> JobRecord:
        self.check_revision(expected=request.expected_revision)
        proposal = self.read_change_set(identifier=request.identifier)
        self.check_revision(expected=proposal.request.expected_revision)
        evidence = proposal.evidence
        if proposal.rejected or evidence is None:
            raise InvalidEvidence(
                reason="Choose a pending proposal with a linked scan crop."
            )
        if not 1 <= request.maximum_points <= self.scope.maximum_points:
            raise InvalidEvidence(
                reason="Choose a point budget within the configured scope."
            )
        self.check_change_set_scope(request=proposal.request)
        query = evidence.query
        if (
            query.candidate_index is not None
            and query.candidate_index != request.candidate_index
        ):
            raise InvalidEvidence(
                reason="Select the registration candidate retained with this crop."
            )
        return self.jobs.start(
            request=JobRequest(
                kind=JobKind.CHANGE_SET_HEATMAP.value,
                candidate_index=request.candidate_index,
                lower=query.lower,
                upper=query.upper,
                frame_T_scan=query.frame_T_scan,
                maximum_count=request.maximum_points,
                change_set_id=proposal.identifier,
                comparison_edits=[
                    BodyTranslation(
                        body_name=edit.body_name, translation=edit.translation
                    )
                    for edit in proposal.request.edits
                ],
            )
        )

    def check_change_set_scope(self, *, request: ChangeSetQuery) -> None:
        """
        Validate membership and movement before any grouped edit is published.
        """
        names = {edit.body_name for edit in request.edits}
        if not 1 <= len(request.edits) <= 50 or len(names) != len(request.edits):
            raise InvalidEvidence(
                reason="Provide translations for 1–50 distinct bodies."
            )
        for edit in request.edits:
            self.check_body_scope(body_name=edit.body_name)
            values = np.asarray(edit.translation)
            if (
                values.shape != (3,)
                or not np.isfinite(values).all()
                or not np.any(values)
                or np.max(np.abs(values)) > self.scope.maximum_translation_m
            ):
                raise InvalidEvidence(
                    reason="Each translation must be a finite, nonzero XYZ delta within the granted movement limit."
                )

    def change_set_review(
        self,
        *,
        proposal: ChangeSet,
        revision: str,
        applied: dict[str, str],
        undone: set[str],
    ) -> ChangeSetReview:
        """
        Classify a proposal using the current journal, including after Undo.
        """
        if all(
            applied.get(edit.body_name) == proposal.identifier
            for edit in proposal.request.edits
        ):
            status = ChangeSetStatus.APPLIED
        elif proposal.identifier in undone:
            status = ChangeSetStatus.UNDONE
        elif proposal.rejected:
            status = ChangeSetStatus.REJECTED
        elif proposal.request.expected_revision == revision:
            status = ChangeSetStatus.PENDING
        else:
            status = ChangeSetStatus.STALE
        return ChangeSetReview(proposal=proposal, status=status)

    @staticmethod
    def redo_proposal_ids(*, history: ActionHistory) -> set[str]:
        """
        Find proposal identities that can be restored through existing history.
        """
        return {
            edit.source_job
            for state in history.future
            for edit in state.overrides
            if edit.source_job
        }

    def read_change_set(self, *, identifier: str) -> ChangeSet:
        """
        Read a project-owned proposal without following arbitrary request paths.
        """
        path = (
            self.artifact_directory(kind=AgentFile.CHANGE_SETS, identifier=identifier)
            / AgentFile.RECORD
        )
        if not path.is_file():
            raise InvalidEvidence(reason="Choose an existing change set.")
        return DataclassJSONSerializer.from_json(
            data=json.loads(path.read_text()), clazz=ChangeSet
        )

    def save_change_set(self, *, proposal: ChangeSet) -> None:
        """
        Publish a complete proposal or rejection record in one rename.
        """
        directory = self.artifact_directory(
            kind=AgentFile.CHANGE_SETS, identifier=proposal.identifier
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / AgentFile.RECORD
        temporary = path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=proposal), indent=2)
        )
        temporary.replace(path)

    def accept_registration(self, *, request: AcceptanceQuery) -> AgentAction:
        self.check_revision(expected=request.expected_revision)
        self.check_reasoning(reasoning=request.reasoning)
        if self.scope.mode != AgentMode.REFINE or not self.scope.allow_registration:
            raise InvalidEvidence(
                reason="Registration acceptance is outside the granted agent scope."
            )
        before = self.history_signature()
        revision = self.revision
        self.project.accept(candidate_index=request.candidate_index)
        return self.record_action(
            operation=AgentOperation.ACCEPT,
            reasoning=request.reasoning,
            job_id="",
            before_revision=revision,
            before_state=before,
        )

    def undo_action(self, *, request: HistoryQuery) -> ProjectSummary:
        self.check_revision(expected=request.expected_revision)
        action = self.read_action(identifier=request.action_id)
        if (
            self.scope.mode != AgentMode.REFINE
            or self.history_signature() != action.after_state
        ):
            raise InvalidEvidence(
                reason="The exact agent action is no longer the latest history state; intervening decisions are preserved."
            )
        self.check_action_scope(action=action)
        self.project.undo()
        return self.inspect_project(request=ProjectQuery())

    def redo_action(self, *, request: HistoryQuery) -> ProjectSummary:
        self.check_revision(expected=request.expected_revision)
        action = self.read_action(identifier=request.action_id)
        if (
            self.scope.mode != AgentMode.REFINE
            or self.history_signature() != action.before_state
        ):
            raise InvalidEvidence(
                reason="Redo requires the exact state before this agent action."
            )
        self.check_action_scope(action=action)
        future = self.project.history.load().future
        if not future or self.state_signature(state=future[-1]) != action.after_state:
            raise InvalidEvidence(reason="This action is not the next Redo revision.")
        self.project.redo()
        return self.inspect_project(request=ProjectQuery())

    def review_source(self, *, request: SourceQuery) -> SourceReview:
        self.check_revision(expected=request.expected_revision)
        return SourceProcess(directory=self.project.directory).request_source_action(
            publish=False
        )

    def export_source(self, *, request: SourceQuery) -> SourceResult:
        self.check_revision(expected=request.expected_revision)
        return SourceProcess(directory=self.project.directory).request_source_action(
            publish=True, identifier=request.review_id
        )

    def check_revision(self, *, expected: str) -> None:
        """
        Fail before work or mutation when the caller inspected another model revision.
        """
        if expected != self.revision:
            raise InvalidEvidence(
                reason="The project geometry or registration changed; inspect the current revision first."
            )

    def check_body_scope(self, *, body_name: str) -> None:
        """
        Reject body requests outside the server-defined object boundary.
        """
        if self.scope.body_names and body_name not in self.scope.body_names:
            raise InvalidEvidence(reason="This body is outside the agent scope.")

    @staticmethod
    def check_reasoning(*, reasoning: Reasoning) -> None:
        """
        Require a bounded explanation rather than recording an unexplained action.
        """
        if (
            not reasoning.explanation.strip()
            or len(reasoning.explanation) > 8000
            or len(reasoning.assumptions) > 50
        ):
            raise InvalidEvidence(
                reason="Provide a nonempty bounded explanation and explicit assumptions."
            )

    def select_candidate(self, *, candidate: int | None) -> int | None:
        """
        Resolve an explicit or accepted provisional placement without inventing
        registration.
        """
        state = self.project.current_state
        if candidate is None:
            candidate = (
                state.record.accepted_candidate
                if state.record.accepted_candidate is not None
                else (0 if state.registration.candidates else None)
            )
        if candidate is not None and not 0 <= candidate < len(
            state.registration.candidates
        ):
            raise InvalidEvidence(reason="Choose an existing registration candidate.")
        return candidate

    def artifact_directory(self, *, kind: Path, identifier: str) -> Path:
        """
        Resolve a local opaque artifact identity without accepting filesystem paths.
        """
        if len(identifier) != 32 or any(
            character not in "0123456789abcdef" for character in identifier
        ):
            raise InvalidEvidence(
                reason="Choose an existing opaque agent artifact identifier."
            )
        return self.project.directory / AgentFile.DIRECTORY / kind / identifier

    def read_region(self, *, identifier: str) -> EvidenceRegion:
        """
        Restore a retained region only for its original frozen scan capture.
        """
        path = (
            self.artifact_directory(kind=AgentFile.REGIONS, identifier=identifier)
            / AgentFile.RECORD
        )
        if not path.is_file():
            raise InvalidEvidence(reason="Choose an existing agent region identifier.")
        region = DataclassJSONSerializer.from_json(
            data=json.loads(path.read_text()), clazz=EvidenceRegion
        )
        if region.scan_sha256 != self.project.record.scan_sha256:
            raise InvalidEvidence(
                reason="This region belongs to a different scan baseline."
            )
        return region

    def save_job_reasoning(
        self, *, record: JobRecord, reasoning: Reasoning, region_id: str = ""
    ) -> None:
        """
        Retain the agent's interpretation beside the worker's independent numeric
        result.
        """
        (
            self.jobs.directory(identifier=record.identifier) / AgentFile.JOB_NOTE
        ).write_text(
            json.dumps(
                DataclassJSONSerializer.to_json(
                    instance=AgentJobNote(reasoning=reasoning, region_id=region_id)
                ),
                indent=2,
            )
        )

    @staticmethod
    def state_signature(*, state: ProjectState) -> str:
        """
        Fingerprint the complete journal state, including intervening inspection
        decisions.
        """
        data = DataclassJSONSerializer.to_json(instance=state)
        inspection = data["inspection"]
        if not state.inspection.saved_assemblies:
            inspection.pop("saved_assemblies", None)
        views = [inspection["view"]]
        views.extend(item["view"] for item in inspection["bookmarks"][JSONField.ITEMS])
        for view in views:
            if not view["selected_assembly"]:
                view.pop("selected_assembly")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def history_signature(self) -> str:
        """
        Capture the current complete journal state for exact Undo/Redo ownership checks.
        """
        return self.state_signature(state=self.project.current_state)

    def record_action(
        self,
        *,
        operation: AgentOperation,
        reasoning: Reasoning,
        job_id: str,
        before_revision: str,
        before_state: str,
    ) -> AgentAction:
        """
        Persist the explanation and the exact before/after journal revision
        fingerprints.
        """
        action = AgentAction(
            identifier=uuid4().hex,
            operation=operation,
            reasoning=reasoning,
            job_id=job_id,
            before_revision=before_revision,
            after_revision=self.revision,
            before_state=before_state,
            after_state=self.history_signature(),
        )
        directory = self.artifact_directory(
            kind=AgentFile.ACTIONS, identifier=action.identifier
        )
        directory.mkdir(parents=True)
        (directory / AgentFile.RECORD).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=action), indent=2)
        )
        return action

    def read_action(self, *, identifier: str) -> AgentAction:
        """
        Restore an existing agent action and its original model application provenance.
        """
        path = (
            self.artifact_directory(kind=AgentFile.ACTIONS, identifier=identifier)
            / AgentFile.RECORD
        )
        if not path.is_file():
            raise InvalidEvidence(reason="Choose an existing agent action identifier.")
        return DataclassJSONSerializer.from_json(
            data=json.loads(path.read_text()), clazz=AgentAction
        )

    def check_action_scope(self, *, action: AgentAction) -> None:
        """
        Recheck authority before traversing a previously authorized action.
        """
        if action.operation == AgentOperation.DECIDE_CHANGE_SET:
            proposal = self.read_change_set(identifier=action.job_id)
            self.check_change_set_scope(request=proposal.request)
            return
        if action.operation == AgentOperation.ACCEPT:
            if not self.scope.allow_registration:
                raise InvalidEvidence(
                    reason="Registration history is outside the current scope."
                )
            return
        record = self.jobs.get(identifier=action.job_id)
        if record.request.kind == JobKind.REGISTRATION:
            if not self.scope.allow_registration:
                raise InvalidEvidence(
                    reason="Registration history is outside the current scope."
                )
        else:
            self.check_body_scope(body_name=record.request.body_name)
            proposal_type = (
                TranslationProposal
                if record.request.kind == JobKind.REFINEMENT
                else DimensionProposal
            )
            proposal = DataclassJSONSerializer.from_json(
                data=json.loads(
                    (
                        self.jobs.directory(identifier=action.job_id) / JobFile.PROPOSAL
                    ).read_text()
                ),
                clazz=proposal_type,
            )
            self.check_proposal_scope(proposal=proposal)

    def check_proposal_scope(
        self, *, proposal: TranslationProposal | DimensionProposal
    ) -> None:
        """
        Recheck actual movement against current authority, including after a scope
        reduction.
        """
        movement = (
            np.max(np.abs(proposal.translation))
            if isinstance(proposal, TranslationProposal)
            else np.max(
                np.abs(proposal.after_box.dimensions - proposal.before_box.dimensions)
            )
        )
        limit = (
            self.scope.maximum_translation_m
            if isinstance(proposal, TranslationProposal)
            else self.scope.maximum_extent_change_m
        )
        if movement > limit + 1e-9:
            raise InvalidEvidence(
                reason="The actual proposed change exceeds the granted scope."
            )
