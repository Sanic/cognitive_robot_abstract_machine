"""
Small typed operation interfaces shared by local, CLI and MCP callers.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scan_refiner.agent.models import (
    ProjectQuery,
    ProjectSummary,
    BodyQuery,
    BodySearchResult,
    RegionQuery,
    EvidenceRegion,
    RegionSampleQuery,
    SampleSummary,
    RegionRenderQuery,
    RenderArtifact,
    RegistrationQuery,
    RefinementQuery,
    ChangeSetQuery,
    ChangeSet,
    ChangeSetList,
    ChangeSetDecisionQuery,
    ChangeSetResult,
    ChangeSetHeatmapQuery,
    ComparisonQuery,
    JobQuery,
    JobInspection,
    ProposalQuery,
    ProposalComparison,
    ApplyQuery,
    AgentAction,
    AcceptanceQuery,
    HistoryQuery,
    SourceQuery,
)
from scan_refiner.jobs import JobRecord
from scan_refiner.source_application import SourceReview, SourceResult


# %% InspectionOperations
@dataclass(kw_only=True)
class InspectionOperations(ABC):
    """
    Explicit contract for inspection operations.
    """

    @abstractmethod
    def inspect_project(self, *, request: ProjectQuery) -> ProjectSummary:
        """
        Inspect identities, revision, evidence availability and granted authority.
        """

    @abstractmethod
    def find_bodies(self, *, request: BodyQuery) -> BodySearchResult:
        """
        Search bounded geometry summaries, optionally verifying live source metadata.
        """

    @abstractmethod
    def create_region(self, *, request: RegionQuery) -> EvidenceRegion:
        """
        Retain explicit scan-relative evidence bounds reusable across point resolutions.
        """

    @abstractmethod
    def inspect_region(self, *, request: RegionSampleQuery) -> SampleSummary:
        """
        Sample bounded original evidence and return provenance/artifact references only.
        """

    @abstractmethod
    def render_region(self, *, request: RegionRenderQuery) -> RenderArtifact:
        """
        Render a frozen crop with current geometry or an unapplied registration
        candidate.
        """


# %% ProposalOperations
@dataclass(kw_only=True)
class ProposalOperations(ABC):
    """
    Explicit contract for proposal operations.
    """

    @abstractmethod
    def propose_registration(self, *, request: RegistrationQuery) -> JobRecord:
        """
        Generate unapplied upright placement candidates using the existing worker.
        """

    @abstractmethod
    def propose_refinement(self, *, request: RefinementQuery) -> JobRecord:
        """
        Generate one bounded body proposal without editing project geometry.
        """

    @abstractmethod
    def submit_change_set(self, *, request: ChangeSetQuery) -> ChangeSet:
        """
        Retain explicit body edits for review without editing project history.
        """

    @abstractmethod
    def list_change_sets(self, *, request: ProjectQuery) -> ChangeSetList:
        """
        List recent proposals and their applicability at one revision.
        """

    @abstractmethod
    def compute_change_set_heatmap(
        self, *, request: ChangeSetHeatmapQuery
    ) -> JobRecord:
        """
        Compute before and after distances on one set of scan records.
        """

    @abstractmethod
    def propose_comparison(self, *, request: ComparisonQuery) -> JobRecord:
        """
        Compute existing labelled residuals and model support over a retained region.
        """

    @abstractmethod
    def inspect_job(self, *, request: JobQuery) -> JobInspection:
        """
        Read job progress and only expose results valid for the current model revision.
        """

    @abstractmethod
    def cancel_job(self, *, request: JobQuery) -> JobRecord:
        """
        Cancel only the worker owned by the shared local service.
        """

    @abstractmethod
    def compare_proposal(self, *, request: ProposalQuery) -> ProposalComparison:
        """
        Expose same-record metrics and optional identical-camera before/after renders.
        """


# %% DecisionOperations
@dataclass(kw_only=True)
class DecisionOperations(ABC):
    """
    Explicit contract for decision operations.
    """

    @abstractmethod
    def apply_proposal(self, *, request: ApplyQuery) -> AgentAction:
        """
        Apply one exact current proposal within the server scope and existing Undo
        journal.
        """

    @abstractmethod
    def decide_change_set(self, *, request: ChangeSetDecisionQuery) -> ChangeSetResult:
        """
        Accept all edits as one history action or decline the proposal.
        """

    @abstractmethod
    def accept_registration(self, *, request: AcceptanceQuery) -> AgentAction:
        """
        Accept a current registration only when project-wide alignment authority was
        granted.
        """

    @abstractmethod
    def undo_action(self, *, request: HistoryQuery) -> ProjectSummary:
        """
        Undo an agent action only when no intervening human/project action exists.
        """

    @abstractmethod
    def redo_action(self, *, request: HistoryQuery) -> ProjectSummary:
        """
        Redo only the exact undone agent action without replacing a later history
        branch.
        """


# %% SourceOperations
@dataclass(kw_only=True)
class SourceOperations(ABC):
    """
    Explicit contract for source operations.
    """

    @abstractmethod
    def review_source(self, *, request: SourceQuery) -> SourceReview:
        """
        Prepare a source-copy review of accepted geometry, never unapplied proposals.
        """

    @abstractmethod
    def export_source(self, *, request: SourceQuery) -> SourceResult:
        """
        Publish a validated independent source copy while retaining original source
        files.
        """


# %% Unified operation surface
@dataclass(kw_only=True)
class AgentOperations(
    InspectionOperations, ProposalOperations, DecisionOperations, SourceOperations
):
    """
    Composition of the small operation contracts exposed by one project connection.
    """
