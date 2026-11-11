"""
Preview imports and deterministic sampling with source-record provenance.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.records import ArrayField
from scan_refiner.point_records import PointField


# %% Preview evidence
@dataclass
class ScanPreview:
    """
    A bounded preview in scan coordinates, with optional original RGB.
    """

    points: np.ndarray
    """Coordinates converted to metres without registration."""
    colors: np.ndarray
    """
    RGB bytes paired with the retained points.
    """

    source_indices: np.ndarray
    """Existing upstream record identities, or preview positions when none were provided."""
    station_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint32)
    )
    """
    Upstream station identities; zero denotes the single imported preview station.
    """

    intensity: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """
    Original intensity where supplied; NaN denotes unavailable measurements.
    """

    def __post_init__(self) -> None:
        """
        Require finite XYZ records and one paired RGB and identity record per point.
        """
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise InvalidEvidence("Scan points must have shape (count, 3).")
        if not len(self.points) or not np.isfinite(self.points).all():
            raise InvalidEvidence("Scan points must be nonempty and finite.")
        if self.colors.shape != self.points.shape:
            raise InvalidEvidence(
                "Scan RGB must have one three-channel record per point."
            )
        if self.source_indices.shape != (len(self.points),):
            raise InvalidEvidence("Scan record identities must match point count.")
        if not len(self.station_indices):
            self.station_indices = np.zeros(len(self.points), dtype=np.uint32)
        if not len(self.intensity):
            self.intensity = np.full(len(self.points), np.nan)
        if self.station_indices.shape != (
            len(self.points),
        ) or self.intensity.shape != (len(self.points),):
            raise InvalidEvidence(
                "Station and intensity records must match point count."
            )

    @classmethod
    def read(cls, path: Path, scale: float = 1.0) -> ScanPreview:
        """
        Import NPZ or PLY preview evidence with explicit source-to-metre scaling.
        """
        if not np.isfinite(scale) or scale <= 0:
            raise InvalidEvidence("Scan scale must be finite and positive.")
        source_indices = station_indices = intensity = None
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                points = np.array(arrays[ArrayField.POINTS], dtype=np.float64)
                color_key = (
                    ArrayField.COLORS
                    if ArrayField.COLORS in arrays
                    else ArrayField.COLOURS
                )
                colors = (
                    np.array(arrays[color_key])
                    if color_key in arrays
                    else np.full(points.shape, 160, dtype=np.uint8)
                )
                source_indices = arrays.get(PointField.SOURCE_INDICES)
                station_indices = arrays.get(PointField.STATION_INDICES)
                intensity = arrays.get(PointField.INTENSITY)
        elif path.suffix.lower() == ".ply":
            cloud = trimesh.load(path, process=False)
            if not isinstance(cloud, trimesh.points.PointCloud):
                raise InvalidEvidence("Expected a PLY point cloud without mesh faces.")
            points = np.asarray(cloud.vertices, dtype=np.float64)
            colors = np.asarray(cloud.colors[:, :3], dtype=np.uint8)
        else:
            raise InvalidEvidence(
                "Import a bounded NPZ or PLY preview for this first version."
            )
        if colors.dtype != np.uint8:
            raise InvalidEvidence(
                "RGB must be uint8; convert normalized colours explicitly."
            )
        return cls(
            points * scale,
            colors,
            np.arange(len(points), dtype=np.int64)
            if source_indices is None
            else source_indices,
            np.empty(0, dtype=np.uint32)
            if station_indices is None
            else station_indices,
            np.empty(0) if intensity is None else intensity,
        )

    def save(self, path: Path) -> None:
        """
        Preserve imported records independently of the original preview file.
        """
        np.savez_compressed(
            path,
            **{
                ArrayField.POINTS: self.points,
                ArrayField.COLORS: self.colors,
                ArrayField.SOURCE_INDICES: self.source_indices,
                PointField.STATION_INDICES: self.station_indices,
                PointField.INTENSITY: self.intensity,
            },
        )

    def sample_indices(self, spacing: float, maximum_count: int) -> np.ndarray:
        """
        Retain original records from spatial cells, capped deterministically.
        """
        if spacing <= 0 or maximum_count <= 0:
            raise InvalidEvidence("Sampling spacing and point budget must be positive.")
        cells = np.floor((self.points - self.points.min(axis=0)) / spacing).astype(
            np.int64
        )
        _, indices = np.unique(cells, axis=0, return_index=True)
        indices.sort()
        if len(indices) > maximum_count:
            indices = indices[
                np.linspace(0, len(indices) - 1, maximum_count, dtype=int)
            ]
        return indices
