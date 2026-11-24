"""
HTTP resources for service-owned bounded analyses and explicit results.
"""

from __future__ import annotations

# %% Imports
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlsplit
import numpy as np
from typing_extensions import TYPE_CHECKING
from krrood.adapters.json_serializer import DataclassJSONSerializer, to_json
from scan_refiner.jobs import JobFile, JobKind, JobRequest, JobStatus
from scan_refiner.coverage import ComparisonArray

if TYPE_CHECKING:
    from scan_refiner.web_service import InspectionServer


# %% Routes
class JobRoute(StrEnum):
    """
    Computation resources beneath the local service.
    """

    ROOT = "/api/jobs"
    CANCEL = "cancel"
    APPLY = "apply"
    POINTS = "points"
    DISTANCES = "distances"
    PROPOSED_DISTANCES = "proposed-distances"
    REPORT = "report"
    PROPOSAL = "proposal"


# %% Job resource handler
@dataclass(init=False, kw_only=True)
class JobRequestHandler(SimpleHTTPRequestHandler, ABC):
    """
    Serve worker lifecycle resources independently of inspection controls.
    """

    server: InspectionServer
    """
    Owning service, project mutation lock and bounded process manager.
    """

    @abstractmethod
    def send_payload(self, *, payload: bytes, content_type: str) -> None:
        """
        Send a completed response with its declared media type.
        """

    def serve_job_get(self) -> bool:
        """
        Handle job progress and result arrays, returning whether the path belongs here.
        """
        path = urlsplit(self.path).path
        if path == JobRoute.ROOT:
            self.send_payload(
                payload=json.dumps(to_json(self.server.jobs.records)).encode(),
                content_type="application/json",
            )
            return True
        if not path.startswith(JobRoute.ROOT + "/"):
            return False
        parts = path[len(JobRoute.ROOT) + 1 :].split("/")
        if len(parts) > 2 or re.fullmatch("[0-9a-f]{32}", parts[0]) is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        identifier = parts[0]
        directory = self.server.jobs.directory(identifier=identifier)
        if not (directory / JobFile.STATE).exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        record = self.server.jobs.get(identifier=identifier)
        if len(parts) == 1:
            self.send_payload(
                payload=json.dumps(
                    DataclassJSONSerializer.to_json(instance=record)
                ).encode(),
                content_type="application/json",
            )
            return True
        if record.status != JobStatus.SUCCEEDED or record.request.kind not in (
            JobKind.HEATMAP,
            JobKind.CHANGE_SET_HEATMAP,
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ):
            self.send_error(
                HTTPStatus.CONFLICT,
                "Result is unfinished or belongs to an earlier model or registration revision.",
            )
            return True
        if parts[1] == JobRoute.PROPOSAL and record.request.kind in (
            JobKind.REFINEMENT,
            JobKind.DIMENSIONS,
        ):
            self.send_payload(
                payload=(directory / JobFile.PROPOSAL).read_bytes(),
                content_type="application/json",
            )
            return True
        if (
            record.request.kind in (JobKind.REFINEMENT, JobKind.DIMENSIONS)
            and parts[1] != JobRoute.POINTS
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        if parts[1] == JobRoute.REPORT:
            report = directory / JobFile.REPORT
            if not report.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return True
            self.send_payload(
                payload=report.read_bytes(), content_type="application/json"
            )
            return True
        if (
            parts[1] == JobRoute.PROPOSED_DISTANCES
            and record.request.kind != JobKind.CHANGE_SET_HEATMAP
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        if parts[1] not in (
            JobRoute.POINTS,
            JobRoute.DISTANCES,
            JobRoute.PROPOSED_DISTANCES,
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        with np.load(
            self.server.jobs.result_path(identifier=identifier), allow_pickle=False
        ) as arrays:
            payload = (
                arrays[ComparisonArray.RECORDS].tobytes()
                if parts[1] == JobRoute.POINTS
                else arrays[
                    ComparisonArray.PROPOSED_DISTANCES
                    if parts[1] == JobRoute.PROPOSED_DISTANCES
                    else ComparisonArray.DISTANCES
                ]
                .astype("<f4")
                .tobytes()
            )
        self.send_payload(payload=payload, content_type="application/octet-stream")
        return True

    def serve_job_post(self, *, length: int) -> bool:
        """
        Start/cancel work or apply reviewed proposals after origin validation.
        """
        path = urlsplit(self.path).path
        if path != JobRoute.ROOT and not path.startswith(JobRoute.ROOT + "/"):
            return False
        with self.server.action_lock:
            if path == JobRoute.ROOT:
                if any(
                    process.poll() is None
                    for process in self.server.jobs.processes.values()
                ):
                    self.send_error(HTTPStatus.CONFLICT, "A job is already running.")
                    return True
                request = DataclassJSONSerializer.from_json(
                    data=json.loads(self.rfile.read(length)), clazz=JobRequest
                )
                if request.kind == JobKind.CHANGE_SET_HEATMAP:
                    self.send_error(
                        HTTPStatus.BAD_REQUEST,
                        "Request the heatmap through the stored change set.",
                    )
                    return True
                problem = self.server.jobs.request_problem(request=request)
                if problem is not None:
                    self.send_error(HTTPStatus.BAD_REQUEST, problem)
                    return True
                record = self.server.jobs.start(request=request)
            else:
                parts = path[len(JobRoute.ROOT) + 1 :].split("/")
                if (
                    len(parts) != 2
                    or re.fullmatch("[0-9a-f]{32}", parts[0]) is None
                    or parts[1] not in (JobRoute.CANCEL, JobRoute.APPLY)
                ):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return True
                identifier = parts[0]
                if not (
                    self.server.jobs.directory(identifier=identifier) / JobFile.STATE
                ).exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return True
                if parts[1] == JobRoute.CANCEL:
                    record = self.server.jobs.cancel(identifier=identifier)
                else:
                    record = self.server.jobs.get(identifier=identifier)
                    if (
                        record.status != JobStatus.SUCCEEDED
                        or record.request.kind
                        not in (
                            JobKind.REGISTRATION,
                            JobKind.REFINEMENT,
                            JobKind.DIMENSIONS,
                        )
                    ):
                        self.send_error(
                            HTTPStatus.CONFLICT,
                            "Only current successful proposals can be applied.",
                        )
                        return True
                    if record.request.kind in (JobKind.REFINEMENT, JobKind.DIMENSIONS):
                        proposal = self.server.jobs.read_proposal(identifier=identifier)
                        if not proposal.applicable:
                            self.send_error(
                                HTTPStatus.CONFLICT,
                                "This fit is rejected; change its constraints or evidence.",
                            )
                            return True
                    self.server.jobs.apply(identifier=identifier)
                    record = self.server.jobs.get(identifier=identifier)
        self.send_payload(
            payload=json.dumps(
                DataclassJSONSerializer.to_json(instance=record)
            ).encode(),
            content_type="application/json",
        )
        return True
