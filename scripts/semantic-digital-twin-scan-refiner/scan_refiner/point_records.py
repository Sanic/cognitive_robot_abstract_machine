"""
Portable numeric records shared with the isolated E57 reader process.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


# %% Browser layout contract
@dataclass(kw_only=True)
class PointLayoutMetadata:
    """
    Numeric wire layout shared by indexed and preview-only job results.
    """

    record_size: int
    """
    Byte count occupied by one original point record.
    """

    record_offsets: dict[str, int]
    """
    Little-endian attribute positions keyed by plain field names.
    """


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


@dataclass(kw_only=True)
class PointRecordLayout:
    """
    A fixed binary layout preserving precision and source-record identities.
    """

    @property
    def metadata(self) -> PointLayoutMetadata:
        """
        Describe the wire format without requiring an attached dense index.
        """
        layout = self.dtype
        return PointLayoutMetadata(
            record_size=layout.itemsize,
            record_offsets={
                str(key): specification[1]
                for key, specification in layout.fields.items()
            },
        )

    @property
    def dtype(self) -> np.dtype:
        """
        Use float64 coordinates and intensity without quantizing original evidence.
        """
        return np.dtype(
            [
                (str(PointField.POINTS), "<f8", 3),
                (str(PointField.COLORS), "u1", 3),
                (str(PointField.SOURCE_INDICES), "<u8"),
                (str(PointField.STATION_INDICES), "<u4"),
                (str(PointField.INTENSITY), "<f8"),
            ]
        )


@dataclass(kw_only=True)
class PointBatch:
    """
    Original evidence in source scan coordinates after declared station poses.
    """

    records: np.ndarray
    """
    Structured rows conforming to :class:`PointRecordLayout`.
    """
