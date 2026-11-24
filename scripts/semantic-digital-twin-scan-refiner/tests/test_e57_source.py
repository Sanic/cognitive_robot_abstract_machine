"""
Optional local integration of streamed multi-station E57 evidence.
"""

from __future__ import annotations

# %% Imports
import os
import subprocess
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest
import sys

from scan_refiner.point_records import PointField
from scan_refiner.scan_sources import E57PointSource
from scan_refiner.project import RefinementProject
from scan_refiner.records import ProjectFile
from scan_refiner.spatial_index import IndexFile, ScanRegion, SpatialScanIndex


# %% Reader environment
class ReaderEnvironment(StrEnum):
    """
    Optional existing reader interpreter without a test dependency installation.
    """

    PYTHON = "SCAN_REFINER_E57_PYTHON"


# %% Container semantics
def test_e57_stream_preserves_station_poses_invalid_masks_and_source_ids(
    tmp_path: Path,
) -> None:
    """
    Separate stations retain identities while coordinates share the E57 scan frame.
    """
    interpreter = os.environ.get(ReaderEnvironment.PYTHON)
    if interpreter is None:
        pytest.skip(
            "Set SCAN_REFINER_E57_PYTHON to an existing pye57 environment for this optional integration."
        )
    source = tmp_path / "stations.e57"
    fixture = Path(__file__).parent / "dataset" / "create_e57.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [interpreter, str(fixture), str(source)], check=True, env=environment
    )
    directory = tmp_path / "index"
    reader = E57PointSource(
        source=source,
        interpreter=Path(interpreter),
        metadata=directory / IndexFile.STATIONS,
        batch_capacity=2,
    )
    index = SpatialScanIndex.build(
        directory=directory, batches=reader.batches(), source_identity="fixture-e57"
    )
    records = np.concatenate(
        [
            batch.records
            for batch in index.batches(
                region=ScanRegion(lower=np.full(3, -100), upper=np.full(3, 100))
            )
        ]
    )
    records.sort(order=[PointField.STATION_INDICES, PointField.SOURCE_INDICES])
    assert len(records) == 6
    np.testing.assert_array_equal(
        records[PointField.SOURCE_INDICES], (0, 1, 2, 0, 1, 2)
    )
    np.testing.assert_array_equal(
        records[PointField.STATION_INDICES], (0, 0, 0, 1, 1, 1)
    )
    np.testing.assert_allclose(
        records[PointField.POINTS],
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (10, 20, 30), (10, 21, 30), (9, 20, 30)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        records[PointField.INTENSITY], (0.1, 0.2, 0.3, 0.1, 0.2, 0.3)
    )
    np.testing.assert_array_equal(
        records[PointField.COLORS][:, 0], (10, 20, 30, 10, 20, 30)
    )
    assert (directory / IndexFile.STATIONS).exists()


def test_command_line_preserves_reader_virtual_environment(
    tmp_path: Path, small_project: RefinementProject
) -> None:
    """
    A virtualenv Python symlink must retain its environment when launched by the CLI.
    """
    interpreter = os.environ.get(ReaderEnvironment.PYTHON)
    if interpreter is None:
        pytest.skip("Set SCAN_REFINER_E57_PYTHON for the optional reader integration.")
    source = tmp_path / "cli-stations.e57"
    fixture = Path(__file__).parent / "dataset" / "create_e57.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [interpreter, str(fixture), str(source)], check=True, env=environment
    )
    entry = Path(__file__).parent.parent / "refine.py"
    subprocess.run(
        [
            sys.executable,
            str(entry),
            "index",
            "--project",
            str(small_project.directory),
            "--e57",
            str(source),
            "--reader-python",
            interpreter,
        ],
        check=True,
        env=environment,
    )
    index = SpatialScanIndex(directory=small_project.directory / ProjectFile.SCAN_INDEX)
    assert index.record.point_count == 6
