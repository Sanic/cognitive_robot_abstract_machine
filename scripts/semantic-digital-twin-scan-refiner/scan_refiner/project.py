"""
Durable baselines and explicitly reviewed registration proposals.
"""

from __future__ import annotations

# %% Imports
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from krrood.adapters.json_serializer import DataclassJSONSerializer

from scan_refiner.exceptions import ExistingProject, InvalidEvidence
from scan_refiner.records import (
    ArrayField,
    GeometryRole,
    ProjectFile,
    ProjectRecord,
    RegistrationResult,
)
from scan_refiner.registration import RegistrationSearch
from scan_refiner.scan import ScanPreview
from scan_refiner.world import WorldSnapshot
from scan_refiner.point_records import PointField


# %% Project lifecycle
@dataclass
class RefinementProject:
    """
    A local project whose saved geometry and scan remain independent of source edits.
    """

    directory: Path
    """Directory containing the immutable baseline and derived inspection artifacts."""

    @staticmethod
    def digest(path: Path) -> str:
        """
        Hash file content using bounded reads without interpreting its format.
        """
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @classmethod
    def create(
        cls,
        directory: Path,
        scan_path: Path,
        snapshot: WorldSnapshot,
        model_source: str,
        scan_scale: float = 1.0,
    ) -> RefinementProject:
        """
        Create a new baseline, refusing every existing destination directory.
        """
        if directory.exists():
            raise ExistingProject(directory)
        scan = ScanPreview.read(scan_path, scan_scale)
        if not snapshot.meshes:
            raise InvalidEvidence("A project needs a world containing geometry.")
        scan_digest = cls.digest(scan_path)
        directory.mkdir(parents=True)
        project = cls(directory)
        snapshot.save(directory / ProjectFile.WORLD)
        scan.save(directory / ProjectFile.SCAN)
        record = ProjectRecord(
            str(scan_path.resolve()),
            scan_digest,
            model_source,
            cls.digest(directory / ProjectFile.WORLD),
            scan_scale,
            len(scan.points),
        )
        project.write_record(record)
        return project

    @property
    def record(self) -> ProjectRecord:
        """
        Read the current project contract and reject unsupported schema revisions.
        """
        data = json.loads((self.directory / ProjectFile.RECORD).read_text())
        record = DataclassJSONSerializer.from_json(data, ProjectRecord)
        if record.schema_version != 1:
            raise InvalidEvidence(
                f"Unsupported project schema {record.schema_version}."
            )
        return record

    def write_record(self, record: ProjectRecord) -> None:
        """
        Publish metadata atomically after the corresponding artifacts exist.
        """
        destination = self.directory / ProjectFile.RECORD
        temporary = destination.with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(record), indent=2)
        )
        temporary.replace(destination)

    @property
    def scan(self) -> ScanPreview:
        """
        Restore the frozen original preview rather than rereading an external file.
        """
        with np.load(self.directory / ProjectFile.SCAN, allow_pickle=False) as arrays:
            return ScanPreview(
                np.array(arrays[ArrayField.POINTS]),
                np.array(arrays[ArrayField.COLORS]),
                np.array(arrays[ArrayField.SOURCE_INDICES]),
                arrays.get(PointField.STATION_INDICES, np.empty(0, dtype=np.uint32)),
                arrays.get(PointField.INTENSITY, np.empty(0)),
            )

    @property
    def snapshot(self) -> WorldSnapshot:
        """
        Restore geometry including original local-origin compensation and world poses.
        """
        return WorldSnapshot.read(self.directory / ProjectFile.WORLD)

    @property
    def registration(self) -> RegistrationResult:
        """
        Read saved proposals, returning an empty set before any registration search.
        """
        path = self.directory / ProjectFile.CANDIDATES
        if not path.exists():
            return RegistrationResult()
        return DataclassJSONSerializer.from_json(
            json.loads(path.read_text()), RegistrationResult
        )

    def align(
        self, search: RegistrationSearch, role: GeometryRole = GeometryRole.VISUAL
    ) -> RegistrationResult:
        """
        Save automatic upright candidates and invalidate any previously accepted
        transform.
        """
        scan = self.scan
        indices = scan.sample_indices(search.cell_width / 3, 20000)
        candidates = search.search(self.snapshot.samples(role), scan.points[indices])
        result = RegistrationResult(candidates, DataclassJSONSerializer.to_json(search))
        destination = self.directory / ProjectFile.CANDIDATES
        temporary = destination.with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(result), indent=2)
        )
        temporary.replace(destination)
        record = self.record
        record.accepted_candidate = None
        record.registration_geometry = str(role)
        self.write_record(record)
        accepted_path = self.directory / ProjectFile.TRANSFORM
        if accepted_path.exists():
            accepted_path.unlink()
        return result

    def accept(self, candidate_index: int) -> None:
        """
        Record a reviewed hypothesis and export its root-to-scan transform.
        """
        candidates = self.registration.candidates
        if not 0 <= candidate_index < len(candidates):
            raise InvalidEvidence("Choose an existing registration candidate index.")
        np.save(
            self.directory / ProjectFile.TRANSFORM,
            candidates[candidate_index].scan_T_root,
        )
        record = self.record
        record.accepted_candidate = candidate_index
        self.write_record(record)
