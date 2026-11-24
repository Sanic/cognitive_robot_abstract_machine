"""Loopback inspection with bounded evidence and reversible view actions."""

from __future__ import annotations

# %% Imports
import gzip
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, urlsplit

import numpy as np
from krrood.adapters.json_serializer import DataclassJSONSerializer, to_json

from scan_refiner.dimensions import BodyDimensionInfo, DimensionFrame
from scan_refiner.frames import CoordinateFrame, OrientedRegion
from scan_refiner.job_service import JobRoute
from scan_refiner.agent.http_service import AgentRequestHandler
from scan_refiner.agent.service import AgentService
from scan_refiner.agent.models import AgentScope
import shutil
from scan_refiner.jobs import JobManager
from scan_refiner.project import RefinementProject
from scan_refiner.history import ActionHistory
from scan_refiner.inspection_state import InspectionState, Measurement, SavedView
from scan_refiner.sections import ModelSection
from scan_refiner.records import ProjectFile, GeometryRole
from scan_refiner.spatial_index import IndexFile, SpatialScanIndex
from scan_refiner.spatial_index import IndexRecord
from scan_refiner.point_records import PointRecordLayout


# %% HTTP vocabulary
class InspectionRoute(StrEnum):
    """Evidence and action resources beneath a project viewer."""

    INFO = "/api/scan-info"
    REGION = "/api/region"
    STATE = "/api/state"
    UNDO = "/api/undo"
    REDO = "/api/redo"
    SECTION = "/api/section"
    ACCEPT = "/api/accept"
    BODY_DIMENSIONS = "/api/body-dimensions"


class QueryField(StrEnum):
    """Regional query arguments expressed in original metric scan coordinates."""

    BOX_FRAME = "frame"
    BODY = "body"
    ROLE = "role"
    LOWER = "lower"
    UPPER = "upper"
    COUNT = "count"
    FRAME = "frame_T_scan"


class QueryFormat(StrEnum):
    """Numeric input forms accepted before parsing coordinate values."""

    NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class ResponseType(StrEnum):
    """Representations returned by the local inspection API."""

    JSON = "application/json"
    RECORDS = "application/octet-stream"


# %% Regional requests
@dataclass(kw_only=True)
class FrameQuery:
    """Optional rigid scan-to-inspection transform in an HTTP query."""

    frame: CoordinateFrame
    """Validated metric inspection frame."""

    @classmethod
    def from_query(cls, *, fields: dict[str, list[str]]) -> FrameQuery | None:
        """Accept an omitted identity frame or a finite proper rigid matrix."""
        if QueryField.FRAME not in fields:
            return cls(frame=CoordinateFrame())
        if len(fields[QueryField.FRAME]) != 1:
            return None
        values = fields[QueryField.FRAME][0].split(",")
        if len(values) != 16 or any(
            len(value) > 64 or re.fullmatch(QueryFormat.NUMBER, value) is None
            for value in values
        ):
            return None
        matrix = np.array(values, dtype=np.float64).reshape(4, 4)
        if not CoordinateFrame.accepts(matrix=matrix):
            return None
        return cls(frame=CoordinateFrame(frame_T_scan=matrix))


@dataclass(kw_only=True)
class RegionRequest:
    """Validated region and browser point budget from an HTTP query."""

    lower: np.ndarray
    """Inclusive inspection-frame lower bounds in metres."""
    upper: np.ndarray
    """Inclusive inspection-frame upper bounds in metres."""
    maximum_count: int
    """Maximum records permitted in this display response."""
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    """Coordinate system in which the request bounds are expressed."""

    @classmethod
    def from_query(
        cls, *, query: str, point_limit: int = 150000
    ) -> RegionRequest | None:
        """Return no request for malformed bounds or a budget outside server limits."""
        fields = parse_qs(query)
        if any(
            key not in fields or len(fields[key]) != 1
            for key in (QueryField.LOWER, QueryField.UPPER, QueryField.COUNT)
        ):
            return None
        count_text = fields[QueryField.COUNT][0]
        if (
            not count_text.isascii()
            or not count_text.isdecimal()
            or len(count_text) > 7
        ):
            return None
        count = int(count_text)
        if not 0 < count <= point_limit:
            return None
        bounds = []
        for key in (QueryField.LOWER, QueryField.UPPER):
            values = fields[key][0].split(",")
            if len(values) != 3 or any(
                len(value) > 64 or re.fullmatch(QueryFormat.NUMBER, value) is None
                for value in values
            ):
                return None
            bounds.append(np.array(values, dtype=np.float64))
        lower, upper = bounds
        if (
            not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower > upper)
        ):
            return None
        frame = FrameQuery.from_query(fields=fields)
        if frame is None:
            return None
        return cls(lower=lower, upper=upper, maximum_count=count, frame=frame.frame)


# %% Inspection actions
@dataclass(kw_only=True)
class AcceptanceAction:
    """Reviewed registration choice applied through project history."""

    candidate_index: int
    """Selected existing placement proposal."""


@dataclass(kw_only=True)
class InspectionAction:
    """Validated inspection replacement with a human-readable history label."""

    inspection: InspectionState
    """Complete view and annotations for the new revision."""
    label: str
    """Action shown in the undo history."""


@dataclass(kw_only=True)
class InspectionSession:
    """Live history and typed annotation templates for the browser."""

    history: ActionHistory
    """Current project revision and reversible decisions."""
    measurement_template: Measurement
    """Typed template for a newly measured endpoint pair."""
    bookmark_template: SavedView
    """Typed template for an annotated view."""
    revision: str
    """Agent revision corresponding to the browser's project state."""


# %% Request handling
@dataclass(kw_only=True)
class IndexInformation:
    """Completed scan identity and binary layout advertised to the browser."""

    index: IndexRecord
    """Coordinate bounds, provenance and overview availability."""
    record_size: int
    """Bytes occupied by one unquantized evidence record."""
    record_offsets: dict[str, int]
    """Byte positions of attributes in the little-endian binary response."""


@dataclass(init=False, kw_only=True)
class InspectionRequestHandler(AgentRequestHandler):
    """Serve project artifacts and query only the project's completed scan index."""

    server: InspectionServer
    """Listener sharing the project action lock across request handlers."""

    def do_GET(self) -> None:
        """Add regional evidence resources to :meth:`SimpleHTTPRequestHandler.do_GET`."""
        if self.serve_agent_get():
            return
        if self.serve_job_get():
            return
        route = urlsplit(self.path)
        if route.path == InspectionRoute.STATE:
            with self.server.action_lock:
                payload = self.inspection_payload()
            self.send_payload(payload=payload, content_type=ResponseType.JSON)
            return
        if route.path == InspectionRoute.BODY_DIMENSIONS:
            query = parse_qs(route.query)
            if (
                len(query.get(QueryField.BODY, [])) != 1
                or len(query.get(QueryField.ROLE, [])) != 1
                or query[QueryField.ROLE][0] not in list(GeometryRole)
                or len(query.get(QueryField.BOX_FRAME, [DimensionFrame.ROOT.value]))
                != 1
                or query.get(QueryField.BOX_FRAME, [DimensionFrame.ROOT.value])[0]
                not in list(DimensionFrame)
            ):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            project = RefinementProject(directory=Path(self.directory))
            info = BodyDimensionInfo.inspect(
                snapshot=project.effective_snapshot,
                body_name=query[QueryField.BODY][0],
                role=query[QueryField.ROLE][0],
                frame_mode=query.get(QueryField.BOX_FRAME, [DimensionFrame.ROOT.value])[
                    0
                ],
            )
            self.send_payload(
                payload=json.dumps(
                    DataclassJSONSerializer.to_json(instance=info)
                ).encode(),
                content_type=ResponseType.JSON,
            )
            return
        if route.path == InspectionRoute.SECTION:
            query = parse_qs(route.query)
            if not all(
                key in query for key in ("plane", "position", "candidate", "role")
            ):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            position = query["position"][0]
            candidate = query["candidate"][0]
            if (
                re.fullmatch(QueryFormat.NUMBER, position) is None
                or (candidate != "none" and not candidate.isdecimal())
                or query["plane"][0] not in ("xy", "xz", "yz")
                or query["role"][0] not in ("visual", "collision", "both", "none")
            ):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            project = RefinementProject(directory=Path(self.directory))
            candidate_index = None if candidate == "none" else int(candidate)
            if not np.isfinite(float(position)) or (
                candidate_index is not None
                and candidate_index >= len(project.registration.candidates)
            ):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            frame = FrameQuery.from_query(fields=query)
            if frame is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            outlines = ModelSection(project=project).build(
                plane=query["plane"][0],
                position=float(position),
                candidate=candidate_index,
                role=query["role"][0],
                frame_T_scan=frame.frame.frame_T_scan,
            )
            self.send_payload(
                payload=json.dumps(to_json(outlines)).encode(),
                content_type=ResponseType.JSON,
            )
            return
        if route.path not in (InspectionRoute.INFO, InspectionRoute.REGION):
            super().do_GET()
            return
        index = SpatialScanIndex(
            directory=Path(self.directory) / ProjectFile.SCAN_INDEX
        )
        if not (index.directory / IndexFile.RECORD).exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "No completed scan index attached to this project.",
            )
            return
        if route.path == InspectionRoute.INFO:
            layout = PointRecordLayout().metadata
            information = IndexInformation(
                index=index.record,
                record_size=layout.record_size,
                record_offsets=layout.record_offsets,
            )
            payload = json.dumps(
                DataclassJSONSerializer.to_json(instance=information)
            ).encode()
            self.send_payload(payload=payload, content_type=ResponseType.JSON)
            return
        request = RegionRequest.from_query(query=route.query)
        if request is None:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "Supply finite ordered XYZ bounds and a point budget of 1–150000.",
            )
            return
        sampled = OrientedRegion(
            lower=request.lower, upper=request.upper, frame=request.frame
        ).sample(index=index, maximum_count=request.maximum_count)
        self.send_payload(
            payload=sampled.records.tobytes(), content_type=ResponseType.RECORDS
        )

    def do_POST(self) -> None:
        """Publish reversible inspection decisions through the loopback API."""
        if self.serve_agent_post():
            return
        if self.serve_source_post():
            return
        route = urlsplit(self.path).path
        if (
            route
            not in (
                InspectionRoute.STATE,
                InspectionRoute.UNDO,
                InspectionRoute.REDO,
                InspectionRoute.ACCEPT,
            )
            and route != JobRoute.ROOT
            and not route.startswith(JobRoute.ROOT + "/")
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        length = self.headers.get("Content-Length", "0")
        if not length.isdecimal() or int(length) > 1000000:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if self.serve_job_post(length=int(length)):
            return
        project = RefinementProject(directory=Path(self.directory))
        with self.server.action_lock:
            if route == InspectionRoute.STATE:
                data = json.loads(self.rfile.read(int(length)))
                action = DataclassJSONSerializer.from_json(
                    data=data, clazz=InspectionAction
                )
                project.update_inspection(
                    inspection=action.inspection, label=action.label
                )
            elif route == InspectionRoute.ACCEPT:
                action = DataclassJSONSerializer.from_json(
                    data=json.loads(self.rfile.read(int(length))),
                    clazz=AcceptanceAction,
                )
                if (
                    not 0
                    <= action.candidate_index
                    < len(project.registration.candidates)
                ):
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                project.accept(candidate_index=action.candidate_index)
            elif route == InspectionRoute.UNDO:
                project.undo()
            else:
                project.redo()
            payload = self.inspection_payload()
        self.send_payload(payload=payload, content_type=ResponseType.JSON)

    def inspection_payload(self) -> bytes:
        """Pair the history and agent revision while the caller holds the action lock."""
        agent = self.server.agent
        session = InspectionSession(
            history=agent.project.history.load(),
            measurement_template=Measurement(),
            bookmark_template=SavedView(),
            revision=agent.revision,
        )
        return json.dumps(DataclassJSONSerializer.to_json(instance=session)).encode()

    def send_payload(self, *, payload: bytes, content_type: str) -> None:
        """Publish a bounded API response with optional standard HTTP compression."""
        compress = "gzip" in self.headers.get("Accept-Encoding", "")
        if compress:
            payload = gzip.compress(payload, compresslevel=3)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Accept-Encoding")
        if compress:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(payload)


@dataclass(kw_only=True, init=False)
class InspectionServer(ThreadingHTTPServer):
    """Serialize project actions while allowing independent scan queries."""

    action_lock: RLock
    """Mutual exclusion for state changes handled by this listener."""

    agent: AgentService
    """Typed operation service sharing this listener's journal and workers."""

    jobs: JobManager
    """Owned process manager for bounded computations."""

    def __init__(
        self,
        address: tuple[str, int],
        handler: partial,
        project: RefinementProject,
        agent_scope: AgentScope | None = None,
        browser_executable: str | None = None,
    ) -> None:
        """Create a server whose mutation lock is shared by all request handlers."""
        self.jobs = JobManager(project=project)
        self.action_lock = RLock()
        self.agent = AgentService(
            project=project,
            jobs=self.jobs,
            scope=agent_scope or AgentScope(),
            browser_executable=browser_executable or shutil.which("chromium-browser"),
        )
        super().__init__(address, handler)

    def server_close(self) -> None:
        """Stop owned computations before releasing :meth:`ThreadingHTTPServer.server_close` resources."""
        self.jobs.close()
        super().server_close()


@dataclass(kw_only=True)
class InspectionService:
    """Configure local inspection without an application-server dependency."""

    project: RefinementProject
    """Project directory whose own index and artifacts are exposed."""

    agent_scope: AgentScope = field(default_factory=AgentScope)
    """Server-owned agent permissions, inspect-only unless explicitly configured."""

    browser_executable: str | None = None
    """Optional existing Chromium executable for agent image artifacts."""

    def create_server(self, *, port: int = 8765) -> InspectionServer:
        """Bind a loopback server; callers own its serving thread and shutdown."""
        handler = partial(
            InspectionRequestHandler, directory=str(self.project.directory)
        )
        return InspectionServer(
            ("127.0.0.1", port),
            handler,
            self.project,
            self.agent_scope,
            self.browser_executable,
        )
