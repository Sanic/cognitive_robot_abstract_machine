"""Owned bounded workers with explicit result application and stale checks."""

from __future__ import annotations

# %% Imports
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import uuid4
import numpy as np
from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood.adapters.json_field import JSONField
from krrood.patterns.path_enumeration import PathEnumeration
from scan_refiner.dimensions import (
    BodyDimensions,
    BodyDimensionInfo,
    DimensionConstraints,
    DimensionProposal,
)
from scan_refiner.refinement import (
    BodyTranslation,
    TranslationConstraints,
    TranslationProposal,
)
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole, ProjectFile, RegistrationResult
from scan_refiner.registration import RegistrationSearch
from scan_refiner.spatial_index import IndexFile, SpatialScanIndex


# %% Job vocabulary
class JobKind(StrEnum):
    """Bounded analyses available from the inspection workspace."""

    HEATMAP = "heatmap"
    CHANGE_SET_HEATMAP = "change-set-heatmap"
    REGISTRATION = "registration"
    REFINEMENT = "refinement"
    DIMENSIONS = "dimensions"


class JobStatus(StrEnum):
    """Lifecycle and applicability of a derived analysis result."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"
    APPLIED = "applied"


class JobFile(PathEnumeration):
    """Artifacts belonging to one computation, independent of project history."""

    STATE = "job.json"
    RESULT = "result.npz"
    REPORT = "report.json"
    PROPOSAL = "proposal.json"
    CANDIDATES = "candidates.json"
    LOG = "worker.log"


class BaselineField(StrEnum):
    """Optional revision fields omitted from legacy geometry fingerprints."""

    OVERRIDES = "overrides"
    DIMENSIONS = "dimension_overrides"
    MATRIX = "matrix"


# %% Request and result records
@dataclass(kw_only=True)
class JobRequest:
    """Declared computation scope and budgets referring to frozen evidence."""

    kind: str = JobKind.HEATMAP.value
    """Requested analysis operation."""
    candidate_index: int | None = None
    """Placement used for a surface comparison."""
    geometry_role: str = GeometryRole.VISUAL.value
    """Explicit model surface representation, without fallback."""
    body_name: str = ""
    """Optional body scope; an empty name includes all surfaces of the role."""
    lower: list[float] = field(default_factory=list)
    """Optional lower crop bounds in the declared inspection frame."""
    upper: list[float] = field(default_factory=list)
    """Optional upper crop bounds in the declared inspection frame."""
    frame_T_scan: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    """Rigid scan-to-crop-frame transform, frozen when the job starts."""
    maximum_count: int = 50000
    """Maximum original records used by the comparison job."""
    model_sample_count: int = 20000
    """Total reproducible model surface samples before regional crop filtering."""
    tolerance: float = 0.02
    """Distance threshold in metres for the reported supported-point fraction."""
    upright: bool = False
    """Explicit confirmation of shared gravity axes for the current registration method."""
    constraints: TranslationConstraints = field(default_factory=TranslationConstraints)
    """Unlocked translation axes, bounded displacement and evidence gates."""
    dimension_constraints: DimensionConstraints = field(
        default_factory=DimensionConstraints
    )
    """Unlocked box extents, fixed face anchors and evidence thresholds."""
    search: RegistrationSearch = field(default_factory=RegistrationSearch)
    """Bounded search settings for upright registration."""
    change_set_id: str = ""
    """Retained proposal associated with a paired heatmap."""
    comparison_edits: list[BodyTranslation] = field(default_factory=list)
    """Frozen unapplied translations used by the proposed-side comparison."""

    @property
    def problem(self) -> str | None:
        """Describe an invalid request without starting or modifying a computation."""
        if (
            self.kind not in list(JobKind)
            or self.geometry_role not in list(GeometryRole)
            or not 0 < self.maximum_count <= 150000
            or not isinstance(self.model_sample_count, int)
            or not 1 <= self.model_sample_count <= 50000
            or not np.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            return "Choose a supported job, geometry role and bounded point budget."
        if not CoordinateFrame.accepts(matrix=np.array(self.frame_T_scan)):
            return "Choose a finite rigid crop frame."
        if self.lower or self.upper:
            if (
                len(self.lower) != 3
                or len(self.upper) != 3
                or not np.isfinite([self.lower, self.upper]).all()
                or np.any(np.array(self.lower) > self.upper)
            ):
                return "Supply ordered finite crop bounds."
        if self.kind == JobKind.CHANGE_SET_HEATMAP and (
            not self.change_set_id or not self.lower or not self.comparison_edits
        ):
            return "A paired heatmap requires a change set and bounded scan crop."
        if self.kind == JobKind.REFINEMENT:
            if not self.body_name or not self.lower:
                return "Select one body and a bounded scan crop for refinement."
            if self.constraints.problem is not None:
                return self.constraints.problem
        if self.kind == JobKind.DIMENSIONS:
            if not self.body_name or not self.lower:
                return "Select one box body and a bounded scan crop for dimensional refinement."
            if self.dimension_constraints.problem is not None:
                return self.dimension_constraints.problem
        if self.kind == JobKind.REGISTRATION and (
            not self.upright
            or not 1 <= self.search.yaw_step_degrees <= 180
            or not 0.01 <= self.search.cell_width <= 10
            or not 1 <= self.search.candidate_count <= 12
            or not 1 <= self.search.refinement_iterations <= 100
            or not 1 <= self.search.maximum_grid_cells <= 1000000
        ):
            return "Confirm shared vertical axes and bounded registration settings."
        return None

    def validate(self) -> None:
        """Reject an illegal persisted request before a worker consumes evidence."""
        if self.problem is not None:
            raise InvalidEvidence(reason=self.problem)


@dataclass(kw_only=True)
class JobBaseline:
    """Immutable input identities and the placement revision used by an analysis."""

    world_sha256: str
    """Frozen world geometry identity."""
    scan_sha256: str
    """Frozen scan preview identity."""
    registration: RegistrationResult
    """Placement proposals against which the job is valid."""

    index_identity: str = ""
    """Dense index source identity, or empty when evidence comes from the preview."""

    overrides: list[BodyTranslation] = field(default_factory=list)
    """Geometry revision whose changes invalidate all dependent analysis results."""

    dimension_overrides: list[BodyDimensions] = field(default_factory=list)
    """Dimensional geometry revision used by the derived analysis."""

    @property
    def signature(self) -> str:
        """Identify the revision without relying on timestamps or candidate indices alone."""
        data = DataclassJSONSerializer.to_json(instance=self)
        if not self.overrides:
            data.pop(BaselineField.OVERRIDES)
        if not self.dimension_overrides:
            data.pop(BaselineField.DIMENSIONS)
        else:
            for encoded, override in zip(
                data[BaselineField.DIMENSIONS][JSONField.ITEMS],
                self.dimension_overrides,
            ):
                if not override.matrix:
                    encoded.pop(BaselineField.MATRIX)
        payload = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(kw_only=True)
class JobRecord:
    """Persistent lifecycle and result summary for one bounded computation."""

    identifier: str
    """Unique local computation identity."""
    request: JobRequest
    """Frozen settings and evidence scope."""
    baseline_signature: str
    """Fingerprint that prevents applying a result to another registration."""
    scan_T_root: list[list[float]]
    """Comparison placement captured at submission time."""
    status: str = JobStatus.QUEUED.value
    """Current computation lifecycle state."""
    message: str = "Queued"
    """Current phase or concrete failure diagnostic."""
    completed: int = 0
    """Processed records for phases with a known total."""
    total: int = 0
    """Record total when measurable; zero denotes an indeterminate phase."""
    median_distance: float | None = None
    """Median unsigned scan-to-surface distance in metres."""
    percentile90_distance: float | None = None
    """Ninetieth percentile distance in metres."""
    within_tolerance: float | None = None
    """Fraction of evaluated scan records inside the declared tolerance."""

    overrides: list[BodyTranslation] = field(default_factory=list)
    """Effective model revision frozen at submission, independent of later edits."""

    dimension_overrides: list[BodyDimensions] = field(default_factory=list)
    """Submitted dimensional model revision, independent of subsequent project edits."""

    def save(self, *, directory: Path) -> None:
        """Atomically publish phase progress after corresponding artifacts are ready."""
        path = directory / JobFile.STATE
        temporary = path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=self), indent=2)
        )
        temporary.replace(path)

    @classmethod
    def read(cls, *, directory: Path) -> JobRecord:
        """Restore a saved request and its current computation state."""
        return DataclassJSONSerializer.from_json(
            data=json.loads((directory / JobFile.STATE).read_text()), clazz=cls
        )


# %% Owned process lifecycle
@dataclass(kw_only=True)
class JobManager:
    """Keep expensive work outside HTTP threads and serialize its resource budget."""

    project: RefinementProject
    """Project whose immutable evidence is analyzed."""
    processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    """Workers owned by this service instance."""
    log_tail_bytes: int = 2048
    """Maximum failure diagnostic read from a worker's stderr log."""

    @property
    def baseline(self) -> JobBaseline:
        """Capture current evidence identities and registration proposals."""
        state = self.project.current_state
        record = state.record
        index = SpatialScanIndex(
            directory=self.project.directory / ProjectFile.SCAN_INDEX
        )
        identity = (
            index.record.source_identity
            if (index.directory / IndexFile.RECORD).exists()
            else ""
        )
        return JobBaseline(
            world_sha256=record.world_sha256,
            scan_sha256=record.scan_sha256,
            registration=state.registration,
            index_identity=identity,
            overrides=state.overrides,
            dimension_overrides=state.dimension_overrides,
        )

    def directory(self, *, identifier: str) -> Path:
        """Resolve a local job identity without allowing path traversal."""
        if len(identifier) != 32 or any(
            character not in "0123456789abcdef" for character in identifier
        ):
            raise InvalidEvidence(reason="Choose an existing local job identifier.")
        return self.project.directory / ProjectFile.JOBS / identifier

    def request_problem(self, *, request: JobRequest) -> str | None:
        """Check declared settings and their applicability to this project's geometry."""
        if request.problem is not None:
            return request.problem
        if any(process.poll() is None for process in self.processes.values()):
            return "Wait for or cancel the running job before starting another."
        if request.kind in (
            JobKind.HEATMAP,
            JobKind.CHANGE_SET_HEATMAP,
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ) and (
            request.candidate_index is None
            or not 0
            <= request.candidate_index
            < len(self.project.registration.candidates)
        ):
            return "Select a registration candidate for comparison."
        surfaces = [
            surface
            for surface in self.project.effective_snapshot.meshes
            if surface.role == request.geometry_role
        ]
        if not surfaces:
            return f"The world has no {request.geometry_role} surfaces."
        if request.body_name and not any(
            surface.body_name == request.body_name for surface in surfaces
        ):
            return f"The selected body has no {request.geometry_role} geometry."
        if request.kind == JobKind.DIMENSIONS:
            return (
                BodyDimensionInfo.inspect(
                    snapshot=self.project.effective_snapshot,
                    body_name=request.body_name,
                    role=request.geometry_role,
                    frame_mode=request.dimension_constraints.frame_mode,
                ).problem
                or None
            )
        return None

    def start(self, *, request: JobRequest) -> JobRecord:
        """Start one isolated worker without replacing any project decisions."""
        problem = self.request_problem(request=request)
        if problem is not None:
            raise InvalidEvidence(reason=problem)
        transform = np.eye(4)
        if request.kind in (
            JobKind.HEATMAP,
            JobKind.CHANGE_SET_HEATMAP,
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ):
            transform = np.array(
                self.project.registration.candidates[
                    request.candidate_index
                ].scan_T_root
            )
        identifier = uuid4().hex
        directory = self.directory(identifier=identifier)
        directory.mkdir(parents=True)
        baseline = self.baseline
        record = JobRecord(
            identifier=identifier,
            request=request,
            baseline_signature=baseline.signature,
            scan_T_root=transform.tolist(),
            overrides=baseline.overrides,
            dimension_overrides=baseline.dimension_overrides,
        )
        record.save(directory=directory)
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with (directory / JobFile.LOG).open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).parents[1] / "run_job.py"),
                    "--project",
                    str(self.project.directory),
                    "--job",
                    identifier,
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.processes[identifier] = process
        return record

    def get(self, *, identifier: str) -> JobRecord:
        """Return progress, worker failures and revision applicability."""
        directory = self.directory(identifier=identifier)
        record = JobRecord.read(directory=directory)
        process = self.processes.get(identifier)
        if (
            process is not None
            and process.poll() is not None
            and record.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        ):
            record.status = JobStatus.FAILED.value
            path = directory / JobFile.LOG
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - self.log_tail_bytes))
                record.message = log.read().decode(errors="replace")
            record.save(directory=directory)
        if (
            record.status == JobStatus.SUCCEEDED
            and record.baseline_signature != self.baseline.signature
        ):
            record.status = JobStatus.STALE.value
            record.message = "Geometry or registration changed; this result belongs to an earlier revision."
        return record

    @property
    def records(self) -> list[JobRecord]:
        """Expose persisted jobs, newest directories first."""
        directory = self.project.directory / ProjectFile.JOBS
        if not directory.exists():
            return []
        paths = sorted(
            directory.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True
        )
        return [
            self.get(identifier=path.name)
            for path in paths
            if path.is_dir() and (path / JobFile.STATE).exists()
        ]

    def cancel(self, *, identifier: str) -> JobRecord:
        """Stop this service's worker and keep preceding decisions unchanged."""
        record = self.get(identifier=identifier)
        process = self.processes.get(identifier)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
            record.status = JobStatus.CANCELLED.value
            record.message = "Cancelled without changing project decisions."
            record.save(directory=self.directory(identifier=identifier))
        return record

    def result_path(self, *, identifier: str) -> Path:
        """Require a current successful result before exposing comparison arrays."""
        record = self.get(identifier=identifier)
        if record.status != JobStatus.SUCCEEDED or record.request.kind not in (
            JobKind.HEATMAP,
            JobKind.CHANGE_SET_HEATMAP,
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ):
            raise InvalidEvidence(
                reason="Only current successful analyses have displayable arrays."
            )
        return self.directory(identifier=identifier) / JobFile.RESULT

    def read_proposal(
        self, *, identifier: str
    ) -> TranslationProposal | DimensionProposal:
        """Restore a current successful geometry proposal with its declared fit type."""
        record = self.get(identifier=identifier)
        if record.status != JobStatus.SUCCEEDED or record.request.kind not in (
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ):
            raise InvalidEvidence(
                reason="Only a current successful geometry proposal can be reviewed."
            )
        proposal_type = (
            DimensionProposal
            if record.request.kind == JobKind.DIMENSIONS
            else TranslationProposal
        )
        return DataclassJSONSerializer.from_json(
            data=json.loads(
                (self.directory(identifier=identifier) / JobFile.PROPOSAL).read_text()
            ),
            clazz=proposal_type,
        )

    def apply(self, *, identifier: str) -> None:
        """Apply a reviewed registration or geometry proposal as an undoable revision."""
        record = self.get(identifier=identifier)
        if record.status != JobStatus.SUCCEEDED or record.request.kind not in (
            JobKind.REGISTRATION,
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ):
            raise InvalidEvidence(
                reason="Only current successful proposals can be applied."
            )
        if record.request.kind in (JobKind.REFINEMENT, JobKind.DIMENSIONS):
            proposal = self.read_proposal(identifier=identifier)
            if not proposal.applicable:
                raise InvalidEvidence(
                    reason="This fit is rejected; choose supported constraints and evidence."
                )
            if record.request.kind == JobKind.DIMENSIONS:
                self.project.apply_dimensions(
                    body_name=proposal.body_name,
                    scale=proposal.scale,
                    offset=proposal.offset,
                    source_job=identifier,
                    matrix=proposal.matrix,
                )
            else:
                self.project.apply_translation(
                    body_name=proposal.body_name,
                    translation=proposal.translation,
                    source_job=identifier,
                )
            record.status = JobStatus.APPLIED.value
            record.message = "Selected-body override added to Undo history."
            record.save(directory=self.directory(identifier=identifier))
            return
        result = DataclassJSONSerializer.from_json(
            data=json.loads(
                (self.directory(identifier=identifier) / JobFile.CANDIDATES).read_text()
            ),
            clazz=RegistrationResult,
        )
        self.project.update_registration(
            result=result, role=record.request.geometry_role
        )
        record.status = JobStatus.APPLIED.value
        record.message = "Candidate proposals added to project history."
        record.save(directory=self.directory(identifier=identifier))

    def close(self) -> None:
        """Terminate only workers owned by this service when its listener closes."""
        for identifier, process in self.processes.items():
            if process.poll() is None:
                self.cancel(identifier=identifier)
