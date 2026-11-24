"""Explicit requests, bounded evidence references and revision-bound agent decisions."""

from __future__ import annotations

# %% Imports
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from enum import StrEnum
import numpy as np
from krrood.patterns.path_enumeration import PathEnumeration
from scan_refiner.coverage import ComparisonReport
from scan_refiner.jobs import JobRecord
from scan_refiner.records import RegistrationResult
from scan_refiner.dimensions import DimensionConstraints, DimensionProposal
from scan_refiner.refinement import TranslationConstraints, TranslationProposal
from scan_refiner.registration import RegistrationSearch
from scan_refiner.exceptions import InvalidEvidence


# %% Vocabulary
class AgentMode(StrEnum):
    """Permission to inspect evidence or apply explicitly scoped changes."""

    INSPECT = "inspect"
    REFINE = "refine"


class EvidenceResolution(StrEnum):
    """Frozen coarse preview or bounded original indexed scan records."""

    PREVIEW = "preview"
    DENSE = "dense"


class RenderView(StrEnum):
    """Identical-camera projections available for evidence and proposal inspection."""

    PERSPECTIVE = "perspective"
    TOP = "top"
    FRONT = "front"
    SIDE = "side"


class AgentOperation(StrEnum):
    """Stable tool names shared by the HTTP transport, CLI and MCP adapter."""

    PROJECT = "inspect_project"
    BODIES = "find_bodies"
    REGION = "create_region"
    SAMPLE = "inspect_region"
    RENDER = "render_region"
    REGISTER = "propose_registration"
    REFINE = "propose_refinement"
    SUBMIT_CHANGE_SET = "submit_change_set"
    LIST_CHANGE_SETS = "list_change_sets"
    DECIDE_CHANGE_SET = "decide_change_set"
    COMPUTE_CHANGE_SET_HEATMAP = "compute_change_set_heatmap"
    JOB = "inspect_job"
    CANCEL = "cancel_job"
    COMPARE = "compare_proposal"
    HEATMAP = "propose_comparison"
    APPLY = "apply_proposal"
    ACCEPT = "accept_registration"
    UNDO = "undo_action"
    REDO = "redo_action"
    SOURCE_REVIEW = "review_source"
    EXPORT = "export_source"


class AgentFile(PathEnumeration):
    """Project-local evidence, image and explanation artifacts."""

    DIRECTORY = "agent"
    REGIONS = "regions"
    RENDERS = "renders"
    ACTIONS = "actions"
    CHANGE_SETS = "change-sets"
    RECORD = "record.json"
    JOB_NOTE = "agent-reasoning.json"
    POINTS = "points.npz"
    IMAGE = "view.png"
    PAGE = "view.html"
    MODEL = "model"


# %% Scope and inputs
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AgentScope:
    """Server-owned limits; clients cannot grant themselves broader authority."""

    mode: AgentMode = AgentMode.INSPECT
    """Inspect mode rejects every project-history mutation."""
    body_names: list[str] = field(default_factory=list)
    """Refinable bodies; an empty list allows all bodies only in explicit refine mode."""
    maximum_translation_m: float = 0.05
    """Maximum absolute per-axis movement in one accepted proposal."""
    maximum_extent_change_m: float = 0.05
    """Maximum absolute change of each fitted extent in one accepted proposal."""
    allow_registration: bool = False
    """Whether refine mode may replace/accept registration candidates."""
    maximum_points: int = 150000
    """Upper bound for bounded regional analysis and image requests."""

    def __post_init__(self) -> None:
        """Reject unsupported modes and nonpositive scope bounds."""
        if (
            self.mode not in list(AgentMode)
            or not np.isfinite(
                [self.maximum_translation_m, self.maximum_extent_change_m]
            ).all()
            or min(self.maximum_translation_m, self.maximum_extent_change_m) <= 0
            or not 1 <= self.maximum_points <= 150000
        ):
            raise InvalidEvidence(
                reason="Choose a supported agent scope with finite positive edit limits and a bounded point budget."
            )


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ProjectQuery:
    """Inspect a project without changing its decisions."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class BodyQuery:
    """Search geometry identities with optional verified source semantics and joints."""

    search: str = ""
    """Case-insensitive substring of the qualified body name."""
    limit: int = 50
    """Maximum returned bodies, capped at 200."""
    include_source: bool = False
    """Reload and verify the original source before reporting semantic/joint metadata."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class RegionQuery:
    """A reusable metric evidence box, directly specified or derived from one body."""

    expected_revision: str | None = None
    """Viewer revision to verify before retaining evidence copied from a selection."""

    body_name: str = ""
    """Body whose current bounds supply a contextual box when explicit bounds are absent."""
    lower: list[float] = field(default_factory=list)
    """Minimum XYZ in the declared inspection frame, in metres."""
    upper: list[float] = field(default_factory=list)
    """Maximum XYZ in the same frame, in metres."""
    frame_T_scan: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    """Rigid scan-to-inspection placement for this crop."""
    candidate_index: int | None = None
    """Placement used to derive body bounds; absent selects accepted or first candidate."""
    margin_m: float = 0.15
    """Nonnegative context around body bounds, independent of geometry fitting."""
    label: str = ""
    """Human-readable purpose retained with the evidence definition."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class RegionSampleQuery:
    """Bounded sampling of a retained evidence region without returning raw points."""

    region_id: str
    """Previously created region identity."""
    maximum_points: int = 50000
    """Maximum sampled original records."""
    resolution: EvidenceResolution = EvidenceResolution.DENSE
    """Dense indexed evidence or explicitly coarse preview evidence."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class RegionRenderQuery:
    """Reproducible scan/model inspection with bounded image and evidence sizes."""

    region_id: str
    """Frozen crop shared by images and numerical fitting."""
    maximum_points: int = 20000
    """Maximum original records before display-only voxel reduction."""
    resolution: EvidenceResolution = EvidenceResolution.DENSE
    """Declared evidence source; no silent fallback from dense to preview."""
    view: RenderView = RenderView.PERSPECTIVE
    """Camera projection, identical for proposal before/after images."""
    geometry_role: str = "visual"
    """Visual, collision or both; scan records remain visible."""
    candidate_index: int | None = None
    """Current placement, or a candidate from placement_job when supplied."""
    placement_job: str = ""
    """Unapplied registration job whose placements may be inspected temporarily."""
    comparison_job: str = ""
    """Current heatmap job whose exact evidence and distances should be rendered."""
    image: bool = True
    """Produce a PNG using the configured existing headless browser, in addition to HTML."""
    display_spacing_m: float = 0.001
    """Display-only sampling spacing; original fitting records are never replaced."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class Reasoning:
    """Agent interpretation kept separate from measured fit evidence."""

    explanation: str
    """Why this operation was selected and what is expected to improve."""
    assumptions: list[str] = field(default_factory=list)
    """Explicit physical assumptions, not claimed as measured facts."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class RegistrationQuery:
    """Automatic upright search whose result remains an unapplied proposal."""

    expected_revision: str
    """Current model/registration revision observed before requesting work."""
    reasoning: Reasoning
    """Purpose and declared assumptions attached to the job."""
    upright: bool
    """Explicit confirmation that both inputs already share vertical Z."""
    geometry_role: str = "visual"
    """Actual surface representation used for the automatic search."""
    search: RegistrationSearch = field(default_factory=RegistrationSearch)
    """Existing bounded occupancy/correspondence search settings."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class RefinementQuery:
    """One existing-body fit with explicit evidence and parameter permissions."""

    expected_revision: str
    """Revision against which the requested body geometry was inspected."""
    region_id: str
    """Evidence crop; original records are resampled within the declared budget."""
    body_name: str
    """Single qualified body whose both roles will change together on application."""
    reasoning: Reasoning
    """Intent and physical assumptions attached to the proposal."""
    operation: str = "translation"
    """Translation or dimensions; unsupported shared constraints are not inferred."""
    geometry_role: str = "visual"
    """Surface used for correspondences, without role fallback."""
    candidate_index: int | None = None
    """Current placement; absent selects accepted or first candidate."""
    maximum_points: int = 50000
    """Bounded original records used by the existing fit worker."""
    tolerance_m: float = 0.02
    """Declared metric tolerance for in-sample before/after diagnostics."""
    translation: TranslationConstraints = field(default_factory=TranslationConstraints)
    """Explicit root-axis locks, association gate and movement bounds."""
    dimensions: DimensionConstraints = field(default_factory=DimensionConstraints)
    """Explicit box frame, extent locks, anchors and evidence requirements."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class JobQuery:
    """Read or cancel one service-owned bounded worker."""

    job_id: str
    """Existing local job identity."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ProposalQuery:
    """Compare one successful fit on the same admitted original records."""

    job_id: str
    """Current successful translation or dimension proposal."""
    view: RenderView = RenderView.PERSPECTIVE
    """Identical camera for current/proposed evidence views."""
    image: bool = False
    """Include bounded PNG/HTML comparison views when requested."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ApplyQuery:
    """Apply an exact proposal after reviewing its current evidence and revision."""

    job_id: str
    """Successful unapplied proposal; rejected fits cannot apply."""
    expected_revision: str
    """Observed current revision; stale requests fail before mutation."""
    reasoning: Reasoning
    """Agent recommendation and declared assumptions retained with the action."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AcceptanceQuery:
    """Accept an already reviewed current registration candidate."""

    candidate_index: int
    """Existing provisional placement index."""
    expected_revision: str
    """Current geometry/registration revision observed by the caller."""
    reasoning: Reasoning
    """Evidence and assumptions supporting acceptance."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class HistoryQuery:
    """Traverse one agent action without undoing intervening human decisions."""

    action_id: str
    """Agent action whose exact before/after history state must be current."""
    expected_revision: str
    """Current geometry/registration revision observed before traversal."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class SourceQuery:
    """Review or export accepted source changes without overwriting original files."""

    expected_revision: str
    """Revision whose accepted changes should appear in the source copy."""
    review_id: str = ""
    """Existing source review for export; empty when preparing a review."""


# %% Grouped proposal review
class ChangeSetStatus(StrEnum):
    """Whether a retained proposal can still be reviewed or applied."""

    PENDING = "pending"
    APPLIED = "applied"
    UNDONE = "undone"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class TranslationEdit:
    """An explicit root-frame displacement from a reviewed revision."""

    body_name: str
    """A qualified geometry identity. The edit covers every geometry role."""
    translation: list[float]
    """XYZ delta in metres, applied once to this body's world geometry."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSetQuery:
    """A hypothesis packaged as explicit edits for one human decision."""

    expected_revision: str
    """Model and registration revision used to plan the change."""
    edits: list[TranslationEdit]
    """One edit per affected body, including any attached parts that should follow."""
    reasoning: Reasoning
    """Explanation, assumptions and uncertainty for the reviewer."""
    region_id: str = ""
    """Optional retained crop used to investigate these edits."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSet:
    """A retained grouped proposal independent of project history."""

    identifier: str
    """Project-local proposal identity."""
    request: ChangeSetQuery
    """Original edits and their planning context."""
    evidence: EvidenceRegion | None = None
    """Frozen crop when the proposal links an investigation region."""
    rejected: bool = False
    """Whether a reviewer declined this proposal."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSetDecisionQuery:
    """Review a specific proposal against the currently displayed revision."""

    identifier: str
    """Proposal shown to the reviewer."""
    expected_revision: str
    """Observed revision for this decision."""
    accept: bool
    """Apply the complete proposal when true, otherwise decline it."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSetReview:
    """A proposal and its applicability in the current workspace."""

    proposal: ChangeSet
    """Edits displayed during review."""
    status: ChangeSetStatus
    """Current relation to the journal and baseline."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSetList:
    """The recent proposal queue as of one project revision."""

    revision: str
    """Current model and registration revision."""
    can_decide: bool
    """Whether the service is configured to permit proposal decisions."""
    reviews: list[ChangeSetReview]
    """Recent proposals, newest first."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSetResult:
    """Outcome of deciding a single grouped proposal."""

    review: ChangeSetReview
    """Resulting status and original proposal."""
    action: AgentAction | None = None
    """Journal action for an accepted group."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ChangeSetHeatmapQuery:
    """Evaluate an unapplied change set against its retained crop."""

    identifier: str
    """Pending proposal whose before and after geometry will be compared."""
    expected_revision: str
    """Browser or agent revision that anchors the review."""
    candidate_index: int
    """Currently displayed registration candidate for both distance arrays."""
    maximum_points: int = 50000
    """Evidence budget shared by both sides of the comparison."""


# %% Region and artifact results
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class EvidenceRegion:
    """Persistent scan-coordinate evidence definition independent of point-cloud resolution."""

    identifier: str
    """Reusable project-local region identity."""
    query: RegionQuery
    """Explicit frozen bounds, frame and their original purpose."""
    scan_sha256: str
    """Scan baseline identity; a region cannot transfer silently to another capture."""
    created_revision: str
    """Model/registration revision used when the crop was chosen."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class SampleSummary:
    """Bounded evidence metadata; no raw point arrays are returned to the agent."""

    region: EvidenceRegion
    """Frozen metric region represented by this sample."""
    count: int
    """Retained original records, not the unknown total region population."""
    requested_count: int
    """Explicit sampling ceiling."""
    resolution: EvidenceResolution
    """Evidence source actually used."""
    evidence_path: str
    """Project-local NPZ containing complete original point records."""
    coordinate_frame: str = "scan"
    """Stored points remain in original scan coordinates."""
    units: str = "m"
    """Metric coordinates, separate from display-only origin compensation."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class RenderArtifact:
    """Immutable inspection view with camera, evidence and geometry provenance."""

    identifier: str
    """Project-local image artifact identity."""
    region_id: str
    """Evidence definition shared by the view and fitting requests."""
    revision: str
    """Geometry and registration revision represented by the image."""
    sample_count: int
    """Original records available to the display sampler."""
    display_count: int
    """Actual scan markers rendered after display-only reduction."""
    view: RenderView
    """Camera orientation and projection."""
    html_path: str
    """Self-contained interactive image source."""
    image_path: str
    """PNG path, empty only when no PNG was requested."""
    display_origin: list[float]
    """Inspection-frame origin subtracted from display-only float32 coordinates."""
    resolution: EvidenceResolution
    """Dense or preview evidence source."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AgentAction:
    """Explanation and exact journal states associated with an agent model change."""

    identifier: str
    """Persistent project-local action identity."""
    operation: AgentOperation
    """Proposal application or registration acceptance."""
    reasoning: Reasoning
    """Agent judgement and assumptions, distinct from worker metrics."""
    job_id: str
    """Originating worker, empty for candidate acceptance."""
    before_revision: str
    """Observed geometry/registration revision before mutation."""
    after_revision: str
    """Resulting geometry/registration revision."""
    before_state: str
    """Full journal-state fingerprint preventing traversal through human actions."""
    after_state: str
    """Full journal state immediately after the agent action."""


# %% Project and object results
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ProjectSummary:
    """Current project identity, capabilities and server-owned permissions."""

    revision: str
    """Geometry/registration fingerprint required by every model-changing request."""
    model_source: str
    """Original source description identity."""
    scan_source: str
    """Original capture/preview identity retained for provenance."""
    dense_index: bool
    """Whether original indexed regional evidence is available."""
    body_count: int
    """Geometric bodies in the frozen effective world."""
    scope: AgentScope
    """Limits the agent cannot expand through tool arguments."""
    accepted_candidate: int | None
    """Current explicitly accepted registration, if any."""
    candidate_count: int
    """Current provisional placement count."""
    recent_actions: list[AgentAction]
    """Most recent agent-authored model changes with their explanations."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class BodySummary:
    """Root-frame geometry summary with optional source-backed semantic context."""

    body_name: str
    """Qualified body identity used by all operations."""
    roles: list[str]
    """Actual available geometry representations."""
    lower: list[float]
    """Minimum model-root XYZ bounds over both roles, in metres."""
    upper: list[float]
    """Maximum model-root XYZ bounds, in metres."""
    box_problem: str
    """Verified oriented-box exclusion; empty means the body can use dimension fitting."""
    parent_name: str | None = None
    """Verified original source parent, unknown unless source metadata was requested."""
    joint_type: str | None = None
    """Verified source incoming connection type."""
    semantic_types: list[str] = field(default_factory=list)
    """Source annotation classes referencing this frame; never inferred from its name."""
    coordinate_frame: str = "model_root"
    """Bounds are independent of scanner placement and crop yaw."""
    units: str = "m"
    """Metric bounds rather than display coordinates."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class BodySearchResult:
    """Bounded object search with explicit truncation."""

    bodies: list[BodySummary]
    """Matching geometric objects."""
    matched_count: int
    """Number matching the search before the output limit."""


# %% Bounded proposal results
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class JobInspection:
    """Existing worker lifecycle and its current reviewable result."""

    record: JobRecord
    """Frozen request, progress and applicability status."""
    current_revision: str = ""
    """Agent revision observed while reading this job."""
    region_id: str = ""
    """Agent region retained with the job when available."""
    reasoning: Reasoning | None = None
    """Agent explanation retained when this job was requested through the agent API."""
    proposal: TranslationProposal | DimensionProposal | None = None
    """Unapplied current geometry proposal, including same-record metrics."""
    registration: RegistrationResult | None = None
    """Separated candidate placements generated by an automatic registration job."""
    report: ComparisonReport | None = None
    """Directional evidence report for a completed comparison job."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ProposalComparison:
    """Numerical proposal review and optional identical-camera evidence images."""

    inspection: JobInspection
    """Outcome, support and in-sample fit metrics on the original admitted records."""
    before: RenderArtifact | None = None
    """Current geometry on the proposal's frozen evidence."""
    after: RenderArtifact | None = None
    """Proposed geometry using the same records, crop and camera."""
    measurement_note: str = "Before/after metrics reuse the same associated records; they are in-sample diagnostics, not independent accuracy."
    """Explicit interpretation boundary for fitting evidence."""


# %% Complete model revision
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AgentRevision:
    """Geometry plus the source and accepted default placement observed by an agent."""

    geometry_signature: str
    """Existing worker geometry/registration fingerprint."""
    model_source: str
    """Source description used by source-copy operations."""
    accepted_candidate: int | None
    """Acceptance is a decision even when candidate geometry is unchanged."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AgentJobNote:
    """Agent interpretation and reusable evidence identity paired with a worker."""

    reasoning: Reasoning
    """Why the job was requested, distinct from worker measurements."""
    region_id: str = ""
    """Original reusable region, absent for global registration searches."""


@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class ComparisonQuery:
    """Request a bounded directional comparison over retained evidence."""

    expected_revision: str
    """Current geometry and placement decision observed before requesting the job."""
    region_id: str
    """Reusable frozen scan-coordinate crop."""
    reasoning: Reasoning
    """Purpose and assumptions for the comparison."""
    candidate_index: int | None = None
    """Explicit or accepted current placement."""
    geometry_role: str = "visual"
    """Actual surfaces used for residuals and evidence support."""
    body_name: str = ""
    """Optional body scope; empty compares all surfaces in the role."""
    maximum_points: int = 50000
    """Bounded original evidence records."""
    model_samples: int = 20000
    """Independent area-weighted model sampling budget."""
    tolerance_m: float = 0.02
    """Declared support threshold, not a confidence estimate."""


# %% Transport failures
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AgentError:
    """A bounded tool refusal or invalid request, without a backend traceback."""

    message: str
    """Specific correction the caller needs to make."""
    error_type: str
    """Validation, scope or revision error class."""
    current_revision: str
    """Current project revision, so stale callers can inspect again."""


# %% Browser revision notification
@dataclass(kw_only=True, config=ConfigDict(extra="forbid"))
class AgentRevisionState:
    """Cheap model change notification independent of point-cloud and mesh loading."""

    revision: str
    """Current source/geometry/registration decision fingerprint."""
    mode: AgentMode
    """Server-authorized application mode shown to the human workspace."""
