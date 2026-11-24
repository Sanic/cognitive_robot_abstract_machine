"""
Rigid inspection frames and exact boxes over immutable scan evidence.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from itertools import product
import numpy as np
import trimesh
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.point_records import PointBatch, PointField
from scan_refiner.spatial_index import RecordSample, ScanRegion, SpatialScanIndex


# %% Coordinate frames
@dataclass(kw_only=True)
class CoordinateFrame:
    """
    Rigid mapping from original scan coordinates to inspection coordinates.
    """

    frame_T_scan: np.ndarray = field(default_factory=lambda: np.eye(4))
    """
    Homogeneous scan-to-frame transform for column vectors.
    """

    def __post_init__(self) -> None:
        """
        Refuse nonrigid transforms before dimensions or bounds are interpreted.
        """
        if not self.accepts(matrix=self.frame_T_scan):
            raise InvalidEvidence(
                reason="Inspection frames require a finite proper rigid transform."
            )

    @staticmethod
    def accepts(*, matrix: np.ndarray) -> bool:
        """
        Return whether a matrix preserves metric dimensions and orientation.
        """
        return (
            matrix.shape == (4, 4)
            and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1])
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
            and np.isclose(np.linalg.det(matrix[:3, :3]), 1)
        )

    def to_frame(self, *, scan_points: np.ndarray) -> np.ndarray:
        """
        Express measured positions in this frame without modifying their source.
        """
        return trimesh.transform_points(scan_points, self.frame_T_scan)

    def to_scan(self, *, frame_points: np.ndarray) -> np.ndarray:
        """
        Recover original scan coordinates from inspection coordinates.
        """
        return trimesh.transform_points(frame_points, np.linalg.inv(self.frame_T_scan))


# %% Oriented evidence regions
@dataclass(kw_only=True)
class OrientedRegion:
    """
    An axis-aligned box in a declared frame, queried against raw scan records.
    """

    lower: np.ndarray
    """
    Inclusive lower inspection-frame bounds in metres.
    """

    upper: np.ndarray
    """
    Inclusive upper inspection-frame bounds in metres.
    """

    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    """
    Rigid relationship to the stored scan frame.
    """

    boundary_tolerance: float = 1e-9
    """
    Metric allowance for numerical roundoff at transformed boundaries.
    """

    def __post_init__(self) -> None:
        """
        Validate metric bounds before any indexed evidence is read.
        """
        ScanRegion(lower=self.lower, upper=self.upper)
        if not np.isfinite(self.boundary_tolerance) or self.boundary_tolerance < 0:
            raise InvalidEvidence(
                reason="Region boundary tolerance must be finite and nonnegative."
            )

    @property
    def corners(self) -> np.ndarray:
        """
        Return the eight box corners in inspection coordinates.
        """
        return np.array(list(product(*zip(self.lower, self.upper, strict=True))))

    @property
    def scan_bounds(self) -> ScanRegion:
        """
        Conservatively bound the oriented box for scan-cell lookup.
        """
        corners = self.frame.to_scan(frame_points=self.corners)
        return ScanRegion(
            lower=corners.min(axis=0) - self.boundary_tolerance,
            upper=corners.max(axis=0) + self.boundary_tolerance,
        )

    def contains(self, *, scan_points: np.ndarray) -> np.ndarray:
        """
        Select original points against the actual oriented box.
        """
        points = self.frame.to_frame(scan_points=scan_points)
        return np.all(
            (points >= self.lower - self.boundary_tolerance)
            & (points <= self.upper + self.boundary_tolerance),
            axis=1,
        )

    def sample(
        self, *, index: SpatialScanIndex, maximum_count: int = 50000
    ) -> PointBatch:
        """
        Filter before budgeting and retain complete records ordered by identity.
        """
        if maximum_count <= 0:
            raise InvalidEvidence(reason="Display point budget must be positive.")
        record = index.record
        index_corners = np.array(
            list(product(*zip(record.lower, record.upper, strict=True)))
        )
        if self.contains(scan_points=index_corners).all():
            sampled = index.sample(
                region=ScanRegion(
                    lower=np.array(record.lower), upper=np.array(record.upper)
                ),
                maximum_count=maximum_count,
            )
            return sampled
        sample = RecordSample(maximum_count=maximum_count)
        for batch in index.batches(region=self.scan_bounds):
            sample.add(
                records=batch.records[
                    self.contains(scan_points=batch.records[PointField.POINTS])
                ]
            )
        records = np.sort(
            sample.records,
            order=[PointField.STATION_INDICES, PointField.SOURCE_INDICES],
        )
        return PointBatch(records=records)
