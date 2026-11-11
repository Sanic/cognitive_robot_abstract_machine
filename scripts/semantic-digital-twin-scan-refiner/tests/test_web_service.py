"""
Verify read-only regional inspection endpoints and bounded binary replies.
"""

from __future__ import annotations

# %% Imports
import threading
import urllib.request
import json
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from typing_extensions import Iterator
from krrood.adapters.json_field import JSONField

from scan_refiner.point_records import PointField, PointRecordLayout
from scan_refiner.project import RefinementProject
from scan_refiner.records import ProjectFile
from scan_refiner.scan_sources import PreviewPointSource
from scan_refiner.spatial_index import SpatialScanIndex
from scan_refiner.web_service import InspectionService, RegionRequest


# %% Request validation
@pytest.mark.parametrize(
    "query",
    [
        "lower=nan,0,0&upper=1,1,1&count=10",
        "lower=2,0,0&upper=1,1,1&count=10",
        "lower=0,0,0&upper=1,1,1&count=99999999",
        "lower=0,0&upper=1,1,1&count=10",
    ],
)
def test_region_request_rejects_invalid_or_excessive_queries(query: str) -> None:
    """
    Untrusted query text cannot create nonfinite bounds or unbounded browser loads.
    """
    assert RegionRequest.from_query(query) is None


# %% Binary regional access
@pytest.fixture
def indexed_server(
    small_project: RefinementProject,
) -> Iterator[ThreadingHTTPServer]:
    """
    Serve a temporary project and close its listener after each HTTP test.
    """
    source = PreviewPointSource(small_project.scan, batch_capacity=3)
    SpatialScanIndex.build(
        small_project.directory / ProjectFile.SCAN_INDEX, source.batches(), "fixture"
    )
    with InspectionService(small_project).create_server(port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
        thread.join(timeout=10)


def test_web_region_reply_contains_original_bounded_records(
    indexed_server: ThreadingHTTPServer,
) -> None:
    """
    HTTP detail access uses the persistent index rather than reopening a raw scan.
    """
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/region?lower=-1,-1,-1&upper=1,1,1&count=3"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = response.read()
        assert response.status == 200
        assert response.headers["Content-Type"] == "application/octet-stream"
    records = np.frombuffer(payload, dtype=PointRecordLayout().dtype)
    assert len(records) == 3
    assert set(records[PointField.SOURCE_INDICES]).issubset(set(range(8)))


def test_browser_record_layout_advertises_plain_attribute_names(
    indexed_server: ThreadingHTTPServer,
) -> None:
    """
    Browser byte offsets are keyed by plain names, without Python enum wrappers.
    """
    with urllib.request.urlopen(
        f"http://127.0.0.1:{indexed_server.server_port}/api/scan-info", timeout=10
    ) as response:
        information = json.load(response)
    assert information["record_offsets"][JSONField.KEYS] == [
        str(key) for key in PointRecordLayout().dtype.names
    ]
    assert information["record_size"] == PointRecordLayout().dtype.itemsize
