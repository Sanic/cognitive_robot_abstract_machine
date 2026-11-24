"""Disk-backed spatial cells with bounded exact and sampled regional access."""

from __future__ import annotations

# %% Imports
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree
from typing_extensions import Iterator

from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood.patterns.path_enumeration import PathEnumeration
from scan_refiner.exceptions import ExistingProject, InvalidEvidence
from scan_refiner.point_records import PointBatch, PointField, PointRecordLayout


# %% Spatial records
class IndexFile(PathEnumeration):
    """Index artifacts published only after all evidence cells have been written."""

    RECORD = "index.json"
    CELLS = "cells.npz"
    DATA = "data"
    STATIONS = "stations.npz"
    OVERVIEW = "overview.bin"


@dataclass(frozen=True, kw_only=True)
class GridCell:
    """Integer coordinates locating a physical cell in the scan frame."""

    x: int
    """Cell coordinate along scan X."""
    y: int
    """Cell coordinate along scan Y."""
    z: int
    """Cell coordinate along scan Z."""


@dataclass(kw_only=True)
class ScanRegion:
    """Inclusive axis-aligned selection in original scan coordinates and metres."""

    lower: np.ndarray
    """Minimum XYZ coordinates, included in the selection."""
    upper: np.ndarray
    """Maximum XYZ coordinates, included in the selection."""

    def __post_init__(self) -> None:
        """Reject nonfinite, wrongly shaped or inverted regions before querying data."""
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise InvalidEvidence(
                reason="Region bounds must each contain three coordinates."
            )
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all():
            raise InvalidEvidence(reason="Region bounds must be finite.")
        if np.any(self.lower > self.upper):
            raise InvalidEvidence(
                reason="Region lower bounds must not exceed upper bounds."
            )

    def contains(self, *, points: np.ndarray) -> np.ndarray:
        """Select boundary points exactly after conservative spatial filtering."""
        return np.all((points >= self.lower) & (points <= self.upper), axis=1)


@dataclass(kw_only=True)
class IndexRecord:
    """Completed spatial evidence identity and declared coordinate interpretation."""

    source_identity: str
    """Source path/hash or another explicitly declared immutable identity."""
    point_count: int
    """Valid original records retained without display downsampling."""
    cell_width: float
    """Physical cell width in metres; this does not quantize point coordinates."""
    lower: list[float]
    """Exact minimum scan coordinates of the retained evidence."""
    upper: list[float]
    """Exact maximum scan coordinates of the retained evidence."""
    schema_version: int = 1
    """Persisted record and binary point-layout version."""
    overview_count: int = 0
    """Original records cached for full-scene display without dense cell reads."""


@dataclass(kw_only=True)
class RecordSample:
    """A bounded, deterministic priority sample retaining complete original records."""

    maximum_count: int
    """Maximum retained records regardless of stream size."""
    records: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=PointRecordLayout().dtype)
    )
    """Original records currently having the smallest generated priorities."""
    priorities: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Random ranking paired with the retained evidence."""
    generator: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0)
    )
    """Repeatable priority sequence, independent of total source size."""

    def add(self, *, records: np.ndarray) -> None:
        """Retain the smallest priorities using at most one input batch plus the budget."""
        combined = np.concatenate((self.records, records))
        priorities = np.concatenate(
            (self.priorities, self.generator.random(len(records)))
        )
        if len(combined) > self.maximum_count:
            selected = np.argpartition(priorities, self.maximum_count - 1)[
                : self.maximum_count
            ]
            combined, priorities = combined[selected], priorities[selected]
        self.records, self.priorities = combined, priorities


# %% Evidence indexing
@dataclass(kw_only=True)
class SpatialScanIndex:
    """A persistent scan index whose queries never need to reopen the scan source."""

    directory: Path
    """Directory containing completed index metadata and binary cell records."""

    @property
    def record(self) -> IndexRecord:
        """Require a completed supported index before exposing evidence."""
        path = self.directory / IndexFile.RECORD
        if not path.exists():
            raise InvalidEvidence(reason="Spatial index is missing or incomplete.")
        record = DataclassJSONSerializer.from_json(
            data=json.loads(path.read_text()), clazz=IndexRecord
        )
        if record.schema_version != 1:
            raise InvalidEvidence(
                reason=f"Unsupported spatial index schema {record.schema_version}."
            )
        return record

    @classmethod
    def build(
        cls,
        *,
        directory: Path,
        batches: Iterator[PointBatch],
        source_identity: str,
        cell_width: float = 1.0,
        maximum_cells: int = 100000,
        progress_interval: float = 15.0,
        overview_count: int = 200000,
    ) -> SpatialScanIndex:
        """Persist streamed records with bounded working memory and atomic completion."""
        if directory.exists():
            raise ExistingProject(directory=directory)
        if (
            not np.isfinite(cell_width)
            or cell_width <= 0
            or maximum_cells <= 0
            or overview_count <= 0
        ):
            raise InvalidEvidence(
                reason="Index cell width and cell budget must be positive."
            )
        (directory / IndexFile.DATA).mkdir(parents=True)
        cells: dict[GridCell, int] = {}
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        count = 0
        overview = RecordSample(maximum_count=overview_count)
        started = last_progress = monotonic()
        for batch in batches:
            records = batch.records
            if records.dtype != PointRecordLayout().dtype:
                raise InvalidEvidence(
                    reason="Point batch layout does not match the spatial index schema."
                )
            if not len(records):
                continue
            overview.add(records=records)
            points = records[PointField.POINTS]
            if not np.isfinite(points).all():
                raise InvalidEvidence(
                    reason="Invalid coordinates must be excluded explicitly by the source reader."
                )
            lower = np.minimum(lower, points.min(axis=0))
            upper = np.maximum(upper, points.max(axis=0))
            coordinates, inverse, counts = np.unique(
                np.floor(points / cell_width).astype(np.int64),
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            order = np.argsort(inverse, kind="stable")
            start = 0
            for coordinate, retained_count in zip(coordinates, counts, strict=True):
                x, y, z = map(int, coordinate)
                cell = GridCell(x=x, y=y, z=z)
                if cell not in cells:
                    if len(cells) >= maximum_cells:
                        raise InvalidEvidence(
                            reason="Spatial cell budget exceeded; use wider cells or a scoped source."
                        )
                    cells[cell] = len(cells)
                with (directory / IndexFile.DATA / f"{cells[cell]:08d}.bin").open(
                    "ab"
                ) as stream:
                    stream.write(
                        records[order[start : start + retained_count]].tobytes()
                    )
                start += retained_count
            count += len(records)
            now = monotonic()
            if now - last_progress >= progress_interval:
                print(
                    f"Indexed {count:,} records in {len(cells):,} cells ({now - started:.0f}s)",
                    flush=True,
                )
                last_progress = now
        if not count:
            raise InvalidEvidence(
                reason="No valid scan records were supplied for indexing."
            )
        coordinates = np.array(
            [[cell.x, cell.y, cell.z] for cell in cells], dtype=np.int64
        )
        np.savez_compressed(directory / IndexFile.CELLS, coordinates=coordinates)
        (directory / IndexFile.OVERVIEW).write_bytes(overview.records.tobytes())
        record = IndexRecord(
            source_identity=source_identity,
            point_count=count,
            cell_width=cell_width,
            lower=lower.tolist(),
            upper=upper.tolist(),
            overview_count=len(overview.records),
        )
        temporary = (directory / IndexFile.RECORD).with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=record), indent=2)
        )
        temporary.replace(directory / IndexFile.RECORD)
        return cls(directory=directory)

    def batches(
        self, *, region: ScanRegion, batch_capacity: int = 250000
    ) -> Iterator[PointBatch]:
        """Yield exact source records from intersecting cells in bounded batches."""
        record = self.record
        if batch_capacity <= 0:
            raise InvalidEvidence(reason="Regional read capacity must be positive.")
        with np.load(self.directory / IndexFile.CELLS, allow_pickle=False) as arrays:
            coordinates = arrays["coordinates"]
        centers = (coordinates + 0.5) * record.cell_width
        tree = cKDTree(centers)
        center = (region.lower + region.upper) / 2
        radius = (
            np.linalg.norm((region.upper - region.lower) / 2)
            + np.sqrt(3) * record.cell_width / 2
        )
        candidates = sorted(tree.query_ball_point(center, radius))
        for identifier in candidates:
            cell_lower = coordinates[identifier] * record.cell_width
            if np.any(cell_lower > region.upper) or np.any(
                cell_lower + record.cell_width < region.lower
            ):
                continue
            path = self.directory / IndexFile.DATA / f"{identifier:08d}.bin"
            records = np.memmap(path, mode="r", dtype=PointRecordLayout().dtype)
            for start in range(0, len(records), batch_capacity):
                chunk = records[start : start + batch_capacity]
                selected = chunk[region.contains(points=chunk[PointField.POINTS])]
                if len(selected):
                    yield PointBatch(records=selected)

    def sample(self, *, region: ScanRegion, maximum_count: int = 50000) -> PointBatch:
        """Choose a deterministic bounded sample while preserving original attributes."""
        if maximum_count <= 0:
            raise InvalidEvidence(reason="Display point budget must be positive.")
        record = self.record
        sampled = RecordSample(maximum_count=maximum_count)
        if (
            record.overview_count >= min(maximum_count, record.point_count)
            and np.all(region.lower <= record.lower)
            and np.all(region.upper >= record.upper)
        ):
            overview = np.memmap(
                self.directory / IndexFile.OVERVIEW,
                mode="r",
                dtype=PointRecordLayout().dtype,
            )
            sampled.add(records=overview)
            return PointBatch(records=sampled.records)
        for batch in self.batches(region=region):
            sampled.add(records=batch.records)
        return PointBatch(records=sampled.records)
