"""Typed HTTP client connecting agents and CLI callers to the shared local workspace."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit
from typing_extensions import TypeVar
import json
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.agent.codec import AgentCodec
from scan_refiner.agent.interfaces import AgentOperations
from scan_refiner.agent.models import AgentOperation, AgentError
from scan_refiner.exceptions import InvalidEvidence
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


# %% Response typing
ResponseType = TypeVar("ResponseType")
"""Concrete response contract selected by each explicit operation."""


# %% Shared HTTP transport
@dataclass(kw_only=True)
class AgentClient(AgentOperations):
    """Use the existing server's worker owner and journal instead of editing project files."""

    base_url: str
    """URL of the running local refiner server."""
    timeout_seconds: float = 240
    """Bounded wait for a tool request, including source validation."""

    def exchange(
        self,
        *,
        operation: AgentOperation,
        request: object,
        response_type: type[ResponseType],
    ) -> ResponseType:
        """Exchange serializer-owned records with the shared revision-checked endpoint."""
        address = urlsplit(self.base_url)
        if (
            address.scheme not in ("http", "https")
            or not address.hostname
            or address.query
            or address.fragment
        ):
            raise InvalidEvidence(
                reason="Supply an HTTP(S) refiner base URL without query or fragment."
            )
        connection = (HTTPSConnection if address.scheme == "https" else HTTPConnection)(
            address.hostname, address.port, timeout=self.timeout_seconds
        )
        payload = AgentCodec().encode_record(record=request)
        connection.request(
            "POST",
            address.path.rstrip("/") + "/api/agent/" + operation.value,
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = response.read()
        connection.close()
        if response.getheader("Content-Type") != "application/json":
            raise InvalidEvidence(
                reason="The target did not return a typed agent response; connect to an agent-enabled refiner server."
            )
        if response.status >= 400:
            failure = DataclassJSONSerializer.from_json(
                data=json.loads(data), clazz=AgentError
            )
            raise InvalidEvidence(reason=failure.message)
        return DataclassJSONSerializer.from_json(
            data=json.loads(data), clazz=response_type
        )

    def read_render_image(self, *, artifact_id: str) -> bytes:
        """Fetch only a project-owned PNG artifact, never an arbitrary filesystem path."""
        address = urlsplit(self.base_url)
        connection = (HTTPSConnection if address.scheme == "https" else HTTPConnection)(
            address.hostname, address.port, timeout=self.timeout_seconds
        )
        connection.request(
            "GET", address.path.rstrip("/") + "/api/agent/images/" + artifact_id
        )
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        if response.status != 200 or response.getheader("Content-Type") != "image/png":
            raise InvalidEvidence(reason="This image artifact is unavailable.")
        return payload

    def inspect_project(self, *, request: ProjectQuery) -> ProjectSummary:
        return self.exchange(
            operation=AgentOperation.PROJECT,
            request=request,
            response_type=ProjectSummary,
        )

    def find_bodies(self, *, request: BodyQuery) -> BodySearchResult:
        return self.exchange(
            operation=AgentOperation.BODIES,
            request=request,
            response_type=BodySearchResult,
        )

    def create_region(self, *, request: RegionQuery) -> EvidenceRegion:
        return self.exchange(
            operation=AgentOperation.REGION,
            request=request,
            response_type=EvidenceRegion,
        )

    def inspect_region(self, *, request: RegionSampleQuery) -> SampleSummary:
        return self.exchange(
            operation=AgentOperation.SAMPLE,
            request=request,
            response_type=SampleSummary,
        )

    def render_region(self, *, request: RegionRenderQuery) -> RenderArtifact:
        return self.exchange(
            operation=AgentOperation.RENDER,
            request=request,
            response_type=RenderArtifact,
        )

    def propose_registration(self, *, request: RegistrationQuery) -> JobRecord:
        return self.exchange(
            operation=AgentOperation.REGISTER, request=request, response_type=JobRecord
        )

    def propose_refinement(self, *, request: RefinementQuery) -> JobRecord:
        return self.exchange(
            operation=AgentOperation.REFINE, request=request, response_type=JobRecord
        )

    def submit_change_set(self, *, request: ChangeSetQuery) -> ChangeSet:
        return self.exchange(
            operation=AgentOperation.SUBMIT_CHANGE_SET,
            request=request,
            response_type=ChangeSet,
        )

    def list_change_sets(self, *, request: ProjectQuery) -> ChangeSetList:
        return self.exchange(
            operation=AgentOperation.LIST_CHANGE_SETS,
            request=request,
            response_type=ChangeSetList,
        )

    def compute_change_set_heatmap(
        self, *, request: ChangeSetHeatmapQuery
    ) -> JobRecord:
        return self.exchange(
            operation=AgentOperation.COMPUTE_CHANGE_SET_HEATMAP,
            request=request,
            response_type=JobRecord,
        )

    def decide_change_set(self, *, request: ChangeSetDecisionQuery) -> ChangeSetResult:
        return self.exchange(
            operation=AgentOperation.DECIDE_CHANGE_SET,
            request=request,
            response_type=ChangeSetResult,
        )

    def propose_comparison(self, *, request: ComparisonQuery) -> JobRecord:
        return self.exchange(
            operation=AgentOperation.HEATMAP, request=request, response_type=JobRecord
        )

    def inspect_job(self, *, request: JobQuery) -> JobInspection:
        return self.exchange(
            operation=AgentOperation.JOB, request=request, response_type=JobInspection
        )

    def cancel_job(self, *, request: JobQuery) -> JobRecord:
        return self.exchange(
            operation=AgentOperation.CANCEL, request=request, response_type=JobRecord
        )

    def compare_proposal(self, *, request: ProposalQuery) -> ProposalComparison:
        return self.exchange(
            operation=AgentOperation.COMPARE,
            request=request,
            response_type=ProposalComparison,
        )

    def apply_proposal(self, *, request: ApplyQuery) -> AgentAction:
        return self.exchange(
            operation=AgentOperation.APPLY, request=request, response_type=AgentAction
        )

    def accept_registration(self, *, request: AcceptanceQuery) -> AgentAction:
        return self.exchange(
            operation=AgentOperation.ACCEPT, request=request, response_type=AgentAction
        )

    def undo_action(self, *, request: HistoryQuery) -> ProjectSummary:
        return self.exchange(
            operation=AgentOperation.UNDO, request=request, response_type=ProjectSummary
        )

    def redo_action(self, *, request: HistoryQuery) -> ProjectSummary:
        return self.exchange(
            operation=AgentOperation.REDO, request=request, response_type=ProjectSummary
        )

    def review_source(self, *, request: SourceQuery) -> SourceReview:
        return self.exchange(
            operation=AgentOperation.SOURCE_REVIEW,
            request=request,
            response_type=SourceReview,
        )

    def export_source(self, *, request: SourceQuery) -> SourceResult:
        return self.exchange(
            operation=AgentOperation.EXPORT, request=request, response_type=SourceResult
        )
