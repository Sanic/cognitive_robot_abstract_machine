"""
Verify read-only regional inspection endpoints and bounded binary replies.
"""

from __future__ import annotations

# %% Imports
from time import monotonic, sleep
from scan_refiner.coverage import ComparisonReport
from scan_refiner.jobs import JobRecord, JobStatus
from scan_refiner.refinement import (
    TranslationConstraints,
    TranslationProposal,
    FitOutcome,
)
from scan_refiner.jobs import JobKind
from scan_refiner.web_service import InspectionServer
from scan_refiner.dimensions import (
    BodyDimensionInfo,
    DimensionConstraints,
    DimensionProposal,
)
import threading
import urllib.request
from urllib.error import HTTPError
import json
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from typing_extensions import Iterator
from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.web_service import InspectionSession, InspectionAction

from scan_refiner.point_records import PointField, PointRecordLayout
from scan_refiner.project import RefinementProject
from scan_refiner.records import ProjectFile, RegistrationCandidate, RegistrationResult
from scan_refiner.jobs import JobRequest
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
    assert RegionRequest.from_query(query=query) is None


# %% Binary regional access
@pytest.fixture
def indexed_server(
    small_project: RefinementProject,
) -> Iterator[ThreadingHTTPServer]:
    """
    Serve a temporary project and close its listener after each HTTP test.
    """
    source = PreviewPointSource(preview=small_project.scan, batch_capacity=3)
    SpatialScanIndex.build(
        directory=small_project.directory / ProjectFile.SCAN_INDEX,
        batches=source.batches(),
        source_identity="fixture",
    )
    with InspectionService(project=small_project).create_server(port=0) as server:
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


# %% Reversible inspection API
def test_http_inspection_actions_support_undo_and_redo(
    indexed_server: ThreadingHTTPServer,
) -> None:
    """
    Browser mutations survive independent requests and traverse durable history.
    """
    url = f"http://127.0.0.1:{indexed_server.server_port}"
    with urllib.request.urlopen(url + "/api/state") as response:
        session = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=InspectionSession
        )
    inspection = session.history.current.inspection
    inspection.view.geometry_mode = "collision"
    action = InspectionAction(inspection=inspection, label="Show collision")
    payload = json.dumps(DataclassJSONSerializer.to_json(instance=action)).encode()
    request = urllib.request.Request(
        url + "/api/state", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        changed = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=InspectionSession
        )
    assert changed.history.current.inspection.view.geometry_mode == "collision"
    for route, expected in (("undo", "visual"), ("redo", "collision")):
        with urllib.request.urlopen(
            urllib.request.Request(url + "/api/" + route, data=b"")
        ) as response:
            changed = DataclassJSONSerializer.from_json(
                data=json.load(response), clazz=InspectionSession
            )
        assert changed.history.current.inspection.view.geometry_mode == expected


# %% Metric section API
def test_http_section_contains_body_labelled_outlines(
    indexed_server: ThreadingHTTPServer,
) -> None:
    """
    The browser receives actual section segments rather than a list type marker.
    """
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/section?plane=xy&position=0&candidate=none&role=visual"
    with urllib.request.urlopen(url) as response:
        outlines = json.load(response)
    assert len(outlines) == 1
    assert "segments" in outlines[0]


# %% Oriented request frames
def test_region_http_uses_exact_oriented_box(
    indexed_server: ThreadingHTTPServer,
) -> None:
    """
    A diagonal frame selects the corresponding cube vertices before budgeting.
    """
    diagonal = float(np.sqrt(0.5))
    matrix = np.array(
        [
            [diagonal, -diagonal, 0, 0],
            [diagonal, diagonal, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    transform = ",".join(map(str, matrix.ravel()))
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/region?lower=-0.01,-1,-1&upper=0.01,1,1&count=8&frame_T_scan={transform}"
    with urllib.request.urlopen(url) as response:
        records = np.frombuffer(response.read(), dtype=PointRecordLayout().dtype)
    assert sorted(records[PointField.SOURCE_INDICES].tolist()) == [0, 1, 6, 7]


def test_region_query_rejects_nonrigid_frame() -> None:
    """
    Malformed scale transforms are rejected rather than changing crop units.
    """
    matrix = np.eye(4)
    matrix[0, 0] = 2
    transform = ",".join(map(str, matrix.ravel()))
    assert (
        RegionRequest.from_query(
            query=f"lower=0,0,0&upper=1,1,1&count=8&frame_T_scan={transform}"
        )
        is None
    )


# %% Browser jobs
def test_server_exposes_owned_jobs(indexed_server: ThreadingHTTPServer) -> None:
    """
    A new project has an explicit empty job list rather than a static-file error.
    """
    with urllib.request.urlopen(
        f"http://127.0.0.1:{indexed_server.server_port}/api/jobs"
    ) as response:
        records = json.load(response)
    assert records == []


def test_job_http_reports_unavailable_body_as_bad_request(
    indexed_server: ThreadingHTTPServer, small_project: RefinementProject
) -> None:
    """
    A selectable body without matching surfaces produces a usable HTTP error.
    """
    small_project.update_registration(
        result=RegistrationResult(
            candidates=[
                RegistrationCandidate(
                    scan_T_root=np.eye(4).tolist(),
                    support_fraction=1.0,
                    median_distance=0.0,
                )
            ]
        ),
        role="visual",
    )
    request = JobRequest(candidate_index=0, body_name="missing/body", maximum_count=8)
    payload = json.dumps(DataclassJSONSerializer.to_json(instance=request)).encode()
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/jobs"
    with pytest.raises(HTTPError) as failure:
        urllib.request.urlopen(
            urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
        )
    assert failure.value.code == 400


# %% Report applicability
def test_report_http_requires_current_completed_analysis(
    indexed_server: ThreadingHTTPServer,
    small_project: RefinementProject,
) -> None:
    """
    A downloadable directional report is guarded by the job's evidence revision.
    """
    small_project.update_registration(
        result=RegistrationResult(
            candidates=[
                RegistrationCandidate(
                    scan_T_root=np.eye(4).tolist(),
                    support_fraction=1.0,
                    median_distance=0.0,
                )
            ]
        ),
        role="visual",
    )
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/jobs"
    request = JobRequest(candidate_index=0, maximum_count=8, model_sample_count=100)
    payload = json.dumps(DataclassJSONSerializer.to_json(instance=request)).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
    ) as response:
        job = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=JobRecord
        )
    deadline = monotonic() + 30
    while (
        job.status in (JobStatus.QUEUED, JobStatus.RUNNING) and monotonic() < deadline
    ):
        sleep(0.05)
        with urllib.request.urlopen(f"{url}/{job.identifier}") as response:
            job = DataclassJSONSerializer.from_json(
                data=json.load(response), clazz=JobRecord
            )
    assert job.status == JobStatus.SUCCEEDED
    with urllib.request.urlopen(f"{url}/{job.identifier}/report") as response:
        report = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=ComparisonReport
        )
    assert report.scan.sample_count == request.maximum_count
    assert report.model_sample_count == request.model_sample_count
    registration = small_project.registration
    registration.candidates[0].scan_T_root[0][3] = 1
    small_project.update_registration(result=registration, role="visual")
    with pytest.raises(HTTPError) as failure:
        urllib.request.urlopen(f"{url}/{job.identifier}/report")
    assert failure.value.code == 409


# %% Rejected pose proposals
def test_rejected_translation_cannot_be_applied_through_http(
    indexed_server: InspectionServer, small_project: RefinementProject
) -> None:
    """
    A completed computation with insufficient evidence is a conflict, not a model edit.
    """
    small_project.update_registration(
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
    request = JobRequest(
        kind=JobKind.REFINEMENT.value,
        candidate_index=0,
        body_name="test/box",
        lower=[-1, -1, -1],
        upper=[1, 1, 1],
        maximum_count=8,
        constraints=TranslationConstraints(axes=[False, False, True]),
    )
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/jobs"
    with urllib.request.urlopen(
        urllib.request.Request(
            url,
            data=json.dumps(DataclassJSONSerializer.to_json(instance=request)).encode(),
            headers={"Content-Type": "application/json"},
        )
    ) as response:
        record = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=JobRecord
        )
    indexed_server.jobs.processes[record.identifier].wait(timeout=30)
    with urllib.request.urlopen(f"{url}/{record.identifier}/proposal") as response:
        proposal = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=TranslationProposal
        )
    assert proposal.outcome == FitOutcome.TOO_FEW_POINTS
    before = small_project.current_state
    with pytest.raises(HTTPError) as failure:
        urllib.request.urlopen(
            urllib.request.Request(f"{url}/{record.identifier}/apply", data=b"")
        )
    assert failure.value.code == 409
    assert small_project.current_state == before


# %% Box eligibility and proposal routes
def test_body_dimension_info_reports_actual_box_bounds(
    indexed_server: InspectionServer,
) -> None:
    """
    The UI receives eligibility and real role bounds before starting a dimension fit.
    """
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/body-dimensions?body=test%2Fbox&role=visual"
    with urllib.request.urlopen(url) as response:
        info = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=BodyDimensionInfo
        )
    assert info.problem == ""
    assert info.bounds.lower == [-0.5, -0.5, -0.5]
    assert info.bounds.upper == [0.5, 0.5, 0.5]


def test_dimension_proposal_can_be_reviewed_through_http(
    indexed_server: InspectionServer, small_project: RefinementProject
) -> None:
    """
    A completed dimension computation is reviewable even if evidence rejects its
    application.
    """
    small_project.update_registration(
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
    request = JobRequest(
        kind=JobKind.DIMENSIONS.value,
        candidate_index=0,
        body_name="test/box",
        lower=[-1, -1, -1],
        upper=[1, 1, 1],
        maximum_count=8,
        dimension_constraints=DimensionConstraints(axes=[True, False, False]),
    )
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/jobs"
    with urllib.request.urlopen(
        urllib.request.Request(
            url,
            data=json.dumps(DataclassJSONSerializer.to_json(instance=request)).encode(),
            headers={"Content-Type": "application/json"},
        )
    ) as response:
        record = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=JobRecord
        )
    indexed_server.jobs.processes[record.identifier].wait(timeout=30)
    with urllib.request.urlopen(f"{url}/{record.identifier}/proposal") as response:
        proposal = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=DimensionProposal
        )
    assert proposal.body_name == request.body_name
    assert proposal.outcome == FitOutcome.TOO_FEW_POINTS


# %% Dimension frame selection
def test_dimension_metadata_accepts_explicit_box_axes(
    indexed_server: InspectionServer,
) -> None:
    """
    The geometric-frame metadata route returns actual bounds and their rigid placement.
    """
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/body-dimensions?body=test%2Fbox&role=visual&frame=box"
    with urllib.request.urlopen(url) as response:
        information = DataclassJSONSerializer.from_json(
            data=json.load(response), clazz=BodyDimensionInfo
        )
    assert information.problem == ""
    np.testing.assert_array_equal(information.frame.root_T_box, np.eye(4))
    np.testing.assert_array_equal(information.bounds.dimensions, [1, 1, 1])


@pytest.mark.parametrize("frame_query", ["frame=unknown", "frame=box&frame=root"])
def test_dimension_metadata_rejects_ambiguous_frame(
    indexed_server: InspectionServer, frame_query: str
) -> None:
    """
    Invalid axes must not silently fall back to a different interpretation of the locks.
    """
    url = f"http://127.0.0.1:{indexed_server.server_port}/api/body-dimensions?body=test%2Fbox&role=visual&{frame_query}"
    with pytest.raises(HTTPError) as failure:
        urllib.request.urlopen(url)
    assert failure.value.code == 400
