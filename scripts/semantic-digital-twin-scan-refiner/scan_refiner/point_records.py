"""
Portable numeric records shared with the isolated E57 reader process.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


# %% Record layout
class PointField(StrEnum):
    """
    Evidence attributes retained independently of browser sampling.
    """

    POINTS = "points"
    COLORS = "colors"
    SOURCE_INDICES = "source_indices"
    STATION_INDICES = "station_indices"
    INTENSITY = "intensity"


@dataclass
class PointRecordLayout:
    """
    A fixed binary layout preserving precision and source-record identities.
    """

    @property
    def dtype(self) -> np.dtype:
        """
        Use float64 coordinates and intensity without quantizing original evidence.
        """
        return np.dtype(
            [
                (PointField.POINTS, "<f8", 3),
                (PointField.COLORS, "u1", 3),
                (PointField.SOURCE_INDICES, "<u8"),
                (PointField.STATION_INDICES, "<u4"),
                (PointField.INTENSITY, "<f8"),
            ]
        )


@dataclass
class PointBatch:
    """
    Original evidence in source scan coordinates after declared station poses.
    """

    records: np.ndarray
    """
    Structured rows conforming to :class:`PointRecordLayout`.
    """
