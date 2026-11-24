"""
Check that exported regional evidence can be reopened without losing identities.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import numpy as np

from scan_refiner.point_records import PointField
from scan_refiner.scan import ScanPreview
from scan_refiner.scan_sources import PreviewPointSource


# %% Source identity round trip
def test_preview_import_retains_existing_station_and_record_ids(tmp_path: Path) -> None:
    """
    A regional NPZ keeps upstream station identities rather than inventing new ones.
    """
    path = tmp_path / "region.npz"
    np.savez(
        path,
        points=np.array(((1, 2, 3), (4, 5, 6))),
        colors=np.array(((10, 20, 30), (40, 50, 60)), dtype=np.uint8),
        source_indices=np.array((900, 900), dtype=np.uint64),
        station_indices=np.array((1, 2), dtype=np.uint32),
        intensity=np.array((0.2, 0.8)),
    )
    preview = ScanPreview.read(path=path)
    records = np.concatenate(
        [batch.records for batch in PreviewPointSource(preview=preview).batches()]
    )
    np.testing.assert_array_equal(records[PointField.SOURCE_INDICES], (900, 900))
    np.testing.assert_array_equal(records[PointField.STATION_INDICES], (1, 2))
    np.testing.assert_allclose(records[PointField.INTENSITY], (0.2, 0.8))
