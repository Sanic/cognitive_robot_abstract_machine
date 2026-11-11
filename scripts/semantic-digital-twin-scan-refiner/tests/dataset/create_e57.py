"""
Create a small multi-station E57 fixture using the optional reader environment.
"""

from __future__ import annotations

# %% Imports
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pye57


# %% Fixture construction
@dataclass
class StationFixture:
    """
    Two station poses with overlapping local record indices and one invalid point.
    """

    destination: Path
    """New fixture file owned by the invoking test."""

    def write(self) -> None:
        """
        Write transformed stations with byte RGB and scaled intensity evidence.
        """
        points = np.array(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (9, 9, 9)), dtype=np.float64
        )
        data = {
            "cartesianX": points[:, 0],
            "cartesianY": points[:, 1],
            "cartesianZ": points[:, 2],
            "cartesianInvalidState": np.array((0, 0, 0, 1), dtype=np.int8),
            "colorRed": np.array((10, 20, 30, 40), dtype=np.uint8),
            "colorGreen": np.array((50, 60, 70, 80), dtype=np.uint8),
            "colorBlue": np.array((90, 100, 110, 120), dtype=np.uint8),
            "intensity": np.array((0.1, 0.2, 0.3, 0.4)),
        }
        with pye57.E57(str(self.destination), mode="w") as scan:
            scan.write_scan_raw(data, name="origin")
            scan.write_scan_raw(
                data,
                name="rotated",
                rotation=np.array((np.sqrt(0.5), 0, 0, np.sqrt(0.5))),
                translation=np.array((10, 20, 30)),
            )


# %% Fixture command
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    StationFixture(parser.parse_args().destination).write()
