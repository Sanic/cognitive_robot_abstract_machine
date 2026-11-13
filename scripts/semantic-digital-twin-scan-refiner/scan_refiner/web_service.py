"""Loopback inspection server with read-only bounded regional evidence access."""

from __future__ import annotations

# %% Imports
import gzip
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from krrood.adapters.json_serializer import DataclassJSONSerializer

from scan_refiner.project import RefinementProject
from scan_refiner.records import ProjectFile
from scan_refiner.spatial_index import IndexFile, ScanRegion, SpatialScanIndex
from scan_refiner.spatial_index import IndexRecord
from scan_refiner.point_records import PointRecordLayout


# %% HTTP vocabulary
class InspectionRoute(StrEnum):
    """Read-only evidence resources beneath a project viewer."""

    INFO = "/api/scan-info"
    REGION = "/api/region"


class QueryField(StrEnum):
    """Regional query arguments expressed in original metric scan coordinates."""

    LOWER = "lower"
    UPPER = "upper"
    COUNT = "count"


class QueryFormat(StrEnum):
    """Numeric input forms accepted before parsing coordinate values."""

    NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class ResponseType(StrEnum):
    """Representations returned by the local inspection API."""

    JSON = "application/json"
    RECORDS = "application/octet-stream"


# %% Regional requests
@dataclass
class RegionRequest:
    """Validated region and browser point budget from an HTTP query."""

    lower: np.ndarray
    """Inclusive scan-frame lower bounds in metres."""
    upper: np.ndarray
    """Inclusive scan-frame upper bounds in metres."""
    maximum_count: int
    """Maximum records permitted in this display response."""

    @classmethod
    def from_query(cls, query: str, point_limit: int = 150000) -> RegionRequest | None:
        """Return no request for malformed bounds or a budget outside server limits."""
        fields = parse_qs(query)
        if any(key not in fields or len(fields[key]) != 1 for key in QueryField):
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
        return cls(lower, upper, count)


# %% Request handling
@dataclass
class IndexInformation:
    """Completed scan identity and binary layout advertised to the browser."""

    index: IndexRecord
    """Coordinate bounds, provenance and overview availability."""
    record_size: int
    """Bytes occupied by one unquantized evidence record."""
    record_offsets: dict[str, int]
    """Byte positions of attributes in the little-endian binary response."""


@dataclass(init=False)
class InspectionRequestHandler(SimpleHTTPRequestHandler):
    """Serve project artifacts and query only the project's completed scan index."""

    def do_GET(self) -> None:
        """Add regional evidence resources to :meth:`SimpleHTTPRequestHandler.do_GET`."""
        route = urlsplit(self.path)
        if route.path not in (InspectionRoute.INFO, InspectionRoute.REGION):
            super().do_GET()
            return
        index = SpatialScanIndex(Path(self.directory) / ProjectFile.SCAN_INDEX)
        if not (index.directory / IndexFile.RECORD).exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "No completed scan index attached to this project.",
            )
            return
        if route.path == InspectionRoute.INFO:
            layout = PointRecordLayout().dtype
            information = IndexInformation(
                index.record,
                layout.itemsize,
                {
                    str(key): specification[1]
                    for key, specification in layout.fields.items()
                },
            )
            payload = json.dumps(
                DataclassJSONSerializer.to_json(instance=information)
            ).encode()
            self.send_payload(payload, ResponseType.JSON)
            return
        request = RegionRequest.from_query(route.query)
        if request is None:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "Supply finite ordered XYZ bounds and a point budget of 1–150000.",
            )
            return
        sampled = index.sample(
            ScanRegion(request.lower, request.upper), request.maximum_count
        )
        self.send_payload(sampled.records.tobytes(), ResponseType.RECORDS)

    def send_payload(self, payload: bytes, content_type: ResponseType) -> None:
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


@dataclass
class InspectionService:
    """Configure a read-only local viewer without an application-server dependency."""

    project: RefinementProject
    """Project directory whose own index and artifacts are exposed."""

    def create_server(self, port: int = 8765) -> ThreadingHTTPServer:
        """Bind a loopback server; callers own its serving thread and shutdown."""
        handler = partial(
            InspectionRequestHandler, directory=str(self.project.directory)
        )
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
