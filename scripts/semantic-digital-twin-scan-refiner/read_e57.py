"""
Stream valid E57 records from an optional isolated reader interpreter.
"""

from __future__ import annotations

# %% Imports
import argparse
import struct
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pye57
from pye57.e57 import convert_spherical_to_cartesian

from scan_refiner.point_records import PointField, PointRecordLayout


# %% E57 vocabulary
class E57Field(StrEnum):
    """
    Coordinate and attribute names defined by the E57 point schema.
    """

    X = "cartesianX"
    Y = "cartesianY"
    Z = "cartesianZ"
    RANGE = "sphericalRange"
    AZIMUTH = "sphericalAzimuth"
    ELEVATION = "sphericalElevation"
    CARTESIAN_INVALID = "cartesianInvalidState"
    SPHERICAL_INVALID = "sphericalInvalidState"
    RED = "colorRed"
    GREEN = "colorGreen"
    BLUE = "colorBlue"
    INTENSITY = "intensity"


class StationField(StrEnum):
    """
    Numeric station metadata retained beside the spatial index.
    """

    TRANSFORMS = "scan_T_station"
    COUNTS = "point_counts"
    FIELDS = "point_fields"


# %% Isolated decoding
@dataclass(kw_only=True)
class E57RecordReader:
    """
    Emit original station/record identities and transformed metric scan coordinates.
    """

    source: Path
    """
    Immutable E57 container; all stations are visited.
    """

    metadata: Path
    """
    Destination for station transforms, source counts and available attributes.
    """

    batch_capacity: int = 250000
    """
    Maximum records decoded at a time, independent of total scan size.
    """

    def stream(self, *, parser: argparse.ArgumentParser) -> None:
        """
        Write length-prefixed batches on stdout and progress only on stderr.
        """
        if self.batch_capacity <= 0:
            parser.error("Batch capacity must be positive.")
        transforms, counts, schemas = [], [], []
        with pye57.E57(str(self.source)) as scan:
            for station_index in range(scan.scan_count):
                header = scan.get_header(station_index)
                transform = np.eye(4)
                transform[:3, :3] = header.rotation_matrix
                transform[:3, 3] = header.translation
                transforms.append(transform)
                counts.append(header.point_count)
                schemas.append(";".join(header.point_fields))
                cartesian = (E57Field.X, E57Field.Y, E57Field.Z)
                spherical = (E57Field.RANGE, E57Field.AZIMUTH, E57Field.ELEVATION)
                coordinates = (
                    cartesian
                    if all(key in header.point_fields for key in cartesian)
                    else spherical
                )
                if not all(key in header.point_fields for key in coordinates):
                    parser.error(
                        f"Station {station_index} has no supported complete coordinates."
                    )
                invalid = (
                    E57Field.CARTESIAN_INVALID
                    if coordinates == cartesian
                    else E57Field.SPHERICAL_INVALID
                )
                attributes = (
                    E57Field.RED,
                    E57Field.GREEN,
                    E57Field.BLUE,
                    E57Field.INTENSITY,
                    invalid,
                )
                fields = list(coordinates) + [
                    key for key in attributes if key in header.point_fields
                ]
                data, buffers = scan.make_buffers(fields, self.batch_capacity)
                reader = header.points.reader(buffers)
                decoded = 0
                count = reader.read()
                while count:
                    points = np.column_stack(
                        [data[key][:count] for key in coordinates]
                    ).astype(np.float64)
                    if coordinates == spherical:
                        points = convert_spherical_to_cartesian(points)
                    points = scan.to_global(points, header.rotation, header.translation)
                    valid = np.isfinite(points).all(axis=1)
                    if invalid in data:
                        valid &= data[invalid][:count] == 0
                    records = np.empty(
                        np.count_nonzero(valid), dtype=PointRecordLayout().dtype
                    )
                    records[PointField.POINTS] = points[valid]
                    records[PointField.SOURCE_INDICES] = np.flatnonzero(valid) + decoded
                    records[PointField.STATION_INDICES] = station_index
                    records[PointField.COLORS] = 160
                    for channel, key in enumerate(
                        (E57Field.RED, E57Field.GREEN, E57Field.BLUE)
                    ):
                        if key in data:
                            values = data[key][:count][valid]
                            if (
                                np.any(values < 0)
                                or np.any(values > 255)
                                or np.any(values != np.floor(values))
                            ):
                                parser.error(
                                    "This evidence layout supports byte RGB only; refusing a lossy colour conversion."
                                )
                            records[PointField.COLORS][:, channel] = values.astype(
                                np.uint8
                            )
                    records[PointField.INTENSITY] = (
                        data[E57Field.INTENSITY][:count][valid]
                        if E57Field.INTENSITY in data
                        else np.nan
                    )
                    sys.stdout.buffer.write(struct.pack("<Q", len(records)))
                    sys.stdout.buffer.write(records.tobytes())
                    sys.stdout.buffer.flush()
                    decoded += count
                    count = reader.read()
                reader.close()
                print(
                    f"Station {station_index}: decoded {decoded:,} source records",
                    file=sys.stderr,
                    flush=True,
                )
        np.savez_compressed(
            self.metadata,
            **{
                StationField.TRANSFORMS: np.array(transforms),
                StationField.COUNTS: np.array(counts, dtype=np.uint64),
                StationField.FIELDS: np.array(schemas),
            },
        )


# %% Standalone entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--batch-capacity", type=int, default=250000)
    arguments = parser.parse_args()
    E57RecordReader(
        source=arguments.source,
        metadata=arguments.metadata,
        batch_capacity=arguments.batch_capacity,
    ).stream(parser=parser)
