"""
Exercise bounded spatial access independently of a particular scan container.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import numpy as np
import pytest

from scan_refiner.exceptions import ExistingProject, InvalidEvidence
from scan_refiner.point_records import PointBatch, PointField, PointRecordLayout
from scan_refiner.spatial_index import ScanRegion, SpatialScanIndex


# %% Evidence fixtures
@pytest.fixture
def point_batches() -> list[PointBatch]:
    """
    Provide distinct stations and identities spanning negative and positive cells.
    """
    records = np.zeros(12, dtype=PointRecordLayout().dtype)
    records[PointField.POINTS] = np.column_stack(
        (np.arange(-6, 6) / 4, np.zeros(12), np.zeros(12))
    )
    records[PointField.COLORS] = np.arange(36, dtype=np.uint8).reshape(12, 3)
    records[PointField.SOURCE_INDICES] = np.arange(100, 112)
    records[PointField.STATION_INDICES] = np.arange(12) % 2
    records[PointField.INTENSITY] = np.arange(12) / 3
    return [PointBatch(records[:5]), PointBatch(records[5:])]


# %% Exact regional access
def test_index_query_preserves_boundary_records_and_original_attributes(
    tmp_path: Path, point_batches: list[PointBatch]
) -> None:
    """
    Closed bounds retrieve exact coordinates, colour, intensity and station identities.
    """
    index = SpatialScanIndex.build(
        tmp_path / "index", iter(point_batches), "fixture", cell_width=0.5
    )
    region = ScanRegion(np.array((-0.5, 0, 0)), np.array((0.5, 0, 0)))
    retrieved = np.concatenate([batch.records for batch in index.batches(region)])
    retrieved.sort(order=PointField.SOURCE_INDICES)
    expected = np.concatenate([batch.records for batch in point_batches])[4:9]
    np.testing.assert_array_equal(retrieved, expected)
    reopened = SpatialScanIndex(index.directory)
    assert reopened.record.point_count == 12
    assert reopened.record.source_identity == "fixture"


def test_index_query_batches_respect_memory_budget(
    tmp_path: Path, point_batches: list[PointBatch]
) -> None:
    """
    Even a large region is exposed as bounded batches rather than one combined cloud.
    """
    index = SpatialScanIndex.build(
        tmp_path / "index", iter(point_batches), "fixture", cell_width=10
    )
    batches = list(
        index.batches(ScanRegion(np.full(3, -10), np.full(3, 10)), batch_capacity=2)
    )
    assert sum(len(batch.records) for batch in batches) == 12
    assert max(len(batch.records) for batch in batches) == 2


def test_region_display_sample_is_reproducible_and_bounded(
    tmp_path: Path, point_batches: list[PointBatch]
) -> None:
    """
    A limited browser budget keeps original records with a repeatable selection.
    """
    index = SpatialScanIndex.build(tmp_path / "index", iter(point_batches), "fixture")
    region = ScanRegion(np.full(3, -10), np.full(3, 10))
    sample = index.sample(region, maximum_count=4)
    repeated = index.sample(region, maximum_count=4)
    assert len(sample.records) == 4
    np.testing.assert_array_equal(sample.records, repeated.records)
    assert set(sample.records[PointField.SOURCE_INDICES]).issubset(set(range(100, 112)))


def test_index_initialization_preserves_existing_data(
    tmp_path: Path, point_batches: list[PointBatch]
) -> None:
    """
    An existing index is never replaced by an import attempt.
    """
    with pytest.raises(ExistingProject):
        SpatialScanIndex.build(tmp_path, iter(point_batches), "fixture")


def test_incomplete_index_is_not_queryable(tmp_path: Path) -> None:
    """
    An interrupted import cannot appear to be a complete spatial index.
    """
    with pytest.raises(InvalidEvidence):
        list(SpatialScanIndex(tmp_path).batches(ScanRegion(np.zeros(3), np.ones(3))))


def test_whole_scene_display_reads_cached_overview(
    tmp_path: Path, point_batches: list[PointBatch], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An overview request must not rescan every dense spatial cell.
    """
    index = SpatialScanIndex.build(tmp_path / "index", iter(point_batches), "fixture")
    monkeypatch.setattr(
        index,
        "batches",
        lambda region: pytest.fail("Dense cells were reopened for an overview."),
    )
    sampled = index.sample(ScanRegion(np.full(3, -10), np.full(3, 10)), maximum_count=4)
    assert len(sampled.records) == 4


@pytest.mark.parametrize(
    "lower,upper", [(np.ones(3), np.zeros(3)), (np.full(3, np.nan), np.ones(3))]
)
def test_region_rejects_invalid_bounds(lower: np.ndarray, upper: np.ndarray) -> None:
    """
    Region semantics require finite ordered bounds before index access.
    """
    with pytest.raises(InvalidEvidence):
        ScanRegion(lower, upper)
