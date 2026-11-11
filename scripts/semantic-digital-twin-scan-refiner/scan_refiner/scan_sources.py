"""
Bounded source adapters for existing previews and isolated E57 decoding.
"""

from __future__ import annotations

# %% Imports
import os
import struct
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing_extensions import Iterator

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.point_records import PointBatch, PointField, PointRecordLayout
from scan_refiner.scan import ScanPreview


# %% Source interface
@dataclass
class PointSource(ABC):
    """
    An ordered stream of evidence with original source-record identities.
    """

    @abstractmethod
    def batches(self) -> Iterator[PointBatch]:
        """
        Yield bounded original records without sampling or coordinate quantization.
        """


@dataclass
class PreviewPointSource(PointSource):
    """
    Expose an already loaded preview through the common evidence interface.
    """

    preview: ScanPreview
    """Previously imported evidence in declared metric scan coordinates."""
    batch_capacity: int = 250000
    """
    Maximum rows allocated by the source adapter at a time.
    """

    def batches(self) -> Iterator[PointBatch]:
        if self.batch_capacity <= 0:
            raise InvalidEvidence("Source batch capacity must be positive.")
        for start in range(0, len(self.preview.points), self.batch_capacity):
            stop = start + self.batch_capacity
            records = np.empty(
                len(self.preview.points[start:stop]), dtype=PointRecordLayout().dtype
            )
            records[PointField.POINTS] = self.preview.points[start:stop]
            records[PointField.COLORS] = self.preview.colors[start:stop]
            records[PointField.SOURCE_INDICES] = self.preview.source_indices[start:stop]
            records[PointField.STATION_INDICES] = self.preview.station_indices[
                start:stop
            ]
            records[PointField.INTENSITY] = self.preview.intensity[start:stop]
            yield PointBatch(records)


@dataclass
class E57PointSource(PointSource):
    """
    Decode E57 using an explicitly selected interpreter without changing dependencies.
    """

    source: Path
    """
    Immutable E57 source path shared with the reader subprocess.
    """

    interpreter: Path
    """
    Existing Python environment containing pye57 and NumPy.
    """

    metadata: Path
    """
    Station metadata output inside the new spatial index directory.
    """

    batch_capacity: int = 250000
    """
    Maximum decoded records in a binary protocol batch.
    """

    def batches(self) -> Iterator[PointBatch]:
        if self.batch_capacity <= 0:
            raise InvalidEvidence("E57 source batch capacity must be positive.")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        worker = Path(__file__).parent.parent / "read_e57.py"
        command = [
            str(self.interpreter),
            str(worker),
            str(self.source),
            str(self.metadata),
            "--batch-capacity",
            str(self.batch_capacity),
        ]
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, env=environment
        ) as process:
            try:
                if process.stdout is None:
                    raise InvalidEvidence(
                        "E57 reader did not expose its binary stream."
                    )
                prefix = process.stdout.read(8)
                while prefix:
                    if len(prefix) != 8:
                        raise InvalidEvidence("Truncated E57 batch header.")
                    count = struct.unpack("<Q", prefix)[0]
                    if count > self.batch_capacity:
                        raise InvalidEvidence(
                            "E57 reader exceeded its declared batch capacity."
                        )
                    payload = process.stdout.read(
                        count * PointRecordLayout().dtype.itemsize
                    )
                    if len(payload) != count * PointRecordLayout().dtype.itemsize:
                        raise InvalidEvidence("Truncated E57 evidence batch.")
                    yield PointBatch(
                        np.frombuffer(payload, dtype=PointRecordLayout().dtype)
                    )
                    prefix = process.stdout.read(8)
                if process.wait() != 0:
                    raise InvalidEvidence(
                        "Isolated E57 reader failed; see its stderr diagnostics."
                    )
            finally:
                if process.poll() is None:
                    process.terminate()
