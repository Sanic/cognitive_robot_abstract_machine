"""
Durable baselines and explicitly reviewed registration proposals.
"""

from __future__ import annotations

# %% Imports
import hashlib
from copy import deepcopy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from krrood.adapters.json_serializer import DataclassJSONSerializer

from scan_refiner.dimensions import BodyDimensions, BodyDimensionInfo, DimensionFrame
from scan_refiner.refinement import BodyTranslation
from scan_refiner.history import ProjectHistory, ProjectState
from scan_refiner.inspection_state import InspectionState, InspectionFrame
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
@dataclass(kw_only=True)
class RefinementProject:
    """
    A local project whose saved geometry and scan remain independent of source edits.
    """

    directory: Path
    """
    Directory containing the immutable baseline and derived inspection artifacts.
    """

    @staticmethod
    def digest(*, path: Path) -> str:
        """
        Hash file content using bounded reads without interpreting its format.
        """
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @classmethod
    def create(
        cls,
        *,
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
            raise ExistingProject(directory=directory)
        scan = ScanPreview.read(path=scan_path, scale=scan_scale)
        if not snapshot.meshes:
            raise InvalidEvidence(reason="A project needs a world containing geometry.")
        scan_digest = cls.digest(path=scan_path)
        directory.mkdir(parents=True)
        project = cls(directory=directory)
        snapshot.save(path=directory / ProjectFile.WORLD)
        scan.save(path=directory / ProjectFile.SCAN)
        record = ProjectRecord(
            scan_source=str(scan_path.resolve()),
            scan_sha256=scan_digest,
            model_source=model_source,
            world_sha256=cls.digest(path=directory / ProjectFile.WORLD),
            scan_scale=scan_scale,
            scan_count=len(scan.points),
        )
        project.write_record(record=record)
        return project

    @property
    def record(self) -> ProjectRecord:
        """
        Read the current project contract and reject unsupported schema revisions.
        """
        if (self.directory / ProjectFile.HISTORY).exists():
            return (
                ProjectHistory(path=self.directory / ProjectFile.HISTORY)
                .load()
                .current.record
            )
        data = json.loads((self.directory / ProjectFile.RECORD).read_text())
        record = DataclassJSONSerializer.from_json(data=data, clazz=ProjectRecord)
        if record.schema_version != 1:
            raise InvalidEvidence(
                reason=f"Unsupported project schema {record.schema_version}."
            )
        return record

    def write_record(self, *, record: ProjectRecord) -> None:
        """
        Publish metadata atomically after the corresponding artifacts exist.
        """
        destination = self.directory / ProjectFile.RECORD
        temporary = destination.with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=record), indent=2)
        )
        temporary.replace(destination)

    @property
    def scan(self) -> ScanPreview:
        """
        Restore the frozen original preview rather than rereading an external file.
        """
        with np.load(self.directory / ProjectFile.SCAN, allow_pickle=False) as arrays:
            return ScanPreview(
                points=np.array(arrays[ArrayField.POINTS]),
                colors=np.array(arrays[ArrayField.COLORS]),
                source_indices=np.array(arrays[ArrayField.SOURCE_INDICES]),
                station_indices=arrays.get(
                    PointField.STATION_INDICES, np.empty(0, dtype=np.uint32)
                ),
                intensity=arrays.get(PointField.INTENSITY, np.empty(0)),
            )

    @property
    def snapshot(self) -> WorldSnapshot:
        """
        Restore geometry including original local-origin compensation and world poses.
        """
        return WorldSnapshot.read(path=self.directory / ProjectFile.WORLD)

    @property
    def effective_snapshot(self) -> WorldSnapshot:
        """
        Restore frozen geometry with the current reviewed body overrides applied.
        """
        snapshot = self.snapshot
        state = self.current_state
        snapshot.apply_dimensions(overrides=state.dimension_overrides)
        snapshot.apply_translations(overrides=state.overrides)
        return snapshot

    def apply_translation(
        self, *, body_name: str, translation: list[float], source_job: str
    ) -> None:
        """
        Append an undoable selected-body displacement without changing attached bodies.
        """
        self.apply_translations(
            edits=[
                BodyTranslation(
                    body_name=body_name, translation=translation, source_job=source_job
                )
            ],
            label="Apply body translation",
        )

    def apply_translations(self, *, edits: list[BodyTranslation], label: str) -> None:
        """
        Apply root-frame displacements together as one reversible revision.
        """
        names = {surface.body_name for surface in self.snapshot.meshes}
        identities = {edit.body_name for edit in edits}
        if not edits or len(identities) != len(edits) or not identities <= names:
            raise InvalidEvidence(
                reason="Supply one translation for each distinct frozen body."
            )
        state = deepcopy(self.current_state)
        current = {item.body_name: item for item in state.overrides}
        for edit in edits:
            edit.validate()
            previous = current.get(edit.body_name)
            translation = (
                list(edit.translation)
                if previous is None
                else (np.array(previous.translation) + edit.translation).tolist()
            )
            current[edit.body_name] = BodyTranslation(
                body_name=edit.body_name,
                translation=translation,
                source_job=edit.source_job,
            )
        state.overrides = list(current.values())
        self.publish(state=state, label=label)

    def apply_dimensions(
        self,
        *,
        body_name: str,
        scale: list[float],
        offset: list[float],
        source_job: str,
        matrix: list[list[float]] | None = None,
    ) -> None:
        """
        Compose a reviewed current-frame box deformation with prior pose and size edits.
        """
        override = BodyDimensions(
            body_name=body_name,
            scale=scale,
            offset=offset,
            source_job=source_job,
            matrix=matrix or [],
        )
        override.validate()
        snapshot = self.effective_snapshot
        surfaces = [
            surface for surface in snapshot.meshes if surface.body_name == body_name
        ]
        if not surfaces:
            raise InvalidEvidence(reason="Choose a body present in the frozen model.")
        info = BodyDimensionInfo.inspect(
            snapshot=snapshot,
            body_name=body_name,
            role=str(surfaces[0].role),
            frame_mode=DimensionFrame.BOX.value,
        )
        if info.problem:
            raise InvalidEvidence(reason=info.problem)
        snapshot.apply_dimensions(overrides=[override])
        after = BodyDimensionInfo.inspect(
            snapshot=snapshot,
            body_name=body_name,
            role=str(surfaces[0].role),
            frame_mode=DimensionFrame.BOX.value,
        )
        if after.problem:
            raise InvalidEvidence(
                reason="The proposed stretch does not preserve compatible box geometry in both roles."
            )
        state = deepcopy(self.current_state)
        translation = next(
            (
                np.array(item.translation)
                for item in state.overrides
                if item.body_name == body_name
            ),
            np.zeros(3),
        )
        previous = next(
            (item for item in state.dimension_overrides if item.body_name == body_name),
            None,
        )
        prior_linear = np.eye(3) if previous is None else previous.linear
        prior_offset = np.zeros(3) if previous is None else np.array(previous.offset)
        proposed_linear = override.linear
        cumulative = proposed_linear @ prior_linear
        if override.matrix or (previous is not None and previous.matrix):
            override.scale = [1.0, 1.0, 1.0]
            override.matrix = cumulative.tolist()
        else:
            override.scale = np.diag(cumulative).tolist()
        override.offset = (
            proposed_linear @ (prior_offset + translation) + offset - translation
        ).tolist()
        state.dimension_overrides = [
            item for item in state.dimension_overrides if item.body_name != body_name
        ] + [override]
        self.publish(state=state, label="Apply body dimensions")

    @property
    def registration(self) -> RegistrationResult:
        """
        Read saved proposals, returning an empty set before any registration search.
        """
        if (self.directory / ProjectFile.HISTORY).exists():
            return (
                ProjectHistory(path=self.directory / ProjectFile.HISTORY)
                .load()
                .current.registration
            )
        path = self.directory / ProjectFile.CANDIDATES
        if not path.exists():
            return RegistrationResult()
        return DataclassJSONSerializer.from_json(
            data=json.loads(path.read_text()), clazz=RegistrationResult
        )

    def align(
        self, *, search: RegistrationSearch, role: GeometryRole = GeometryRole.VISUAL
    ) -> RegistrationResult:
        """
        Save automatic upright candidates and invalidate any previously accepted
        transform.
        """
        scan = self.scan
        indices = scan.sample_indices(
            spacing=search.cell_width / 3, maximum_count=20000
        )
        candidates = search.search(
            model_points=self.effective_snapshot.samples(role=role),
            scan_points=scan.points[indices],
        )
        result = RegistrationResult(
            candidates=candidates,
            settings=DataclassJSONSerializer.to_json(instance=search),
        )
        self.update_registration(result=result, role=str(role))
        return result

    def accept(self, *, candidate_index: int) -> None:
        """
        Record a reviewed hypothesis and export its root-to-scan transform.
        """
        candidates = self.registration.candidates
        if not 0 <= candidate_index < len(candidates):
            raise InvalidEvidence(
                reason="Choose an existing registration candidate index."
            )
        state = deepcopy(self.current_state)
        state.record.accepted_candidate = candidate_index
        self.publish(state=state, label="Accept registration")

    @property
    def history(self) -> ProjectHistory:
        """
        Expose durable revisions, migrating existing projects on their first action.
        """
        path = self.directory / ProjectFile.HISTORY
        if path.exists():
            return ProjectHistory(path=path)
        baseline = ProjectState(record=self.record, registration=self.registration)
        if baseline.registration.candidates:
            baseline.inspection.view.coordinate_frame = InspectionFrame.WORLD.value
        return ProjectHistory(path=path, baseline=baseline)

    @property
    def current_state(self) -> ProjectState:
        """
        Restore the authoritative revision including inspection annotations.
        """
        return self.history.load().current

    def publish(self, *, state: ProjectState, label: str) -> None:
        """
        Append a reversible decision and discard an abandoned redo branch.
        """
        history = self.history.load()
        state.inspection.validate()
        for override in state.overrides:
            override.validate()
        for override in state.dimension_overrides:
            override.validate()
        if (
            state.record,
            state.registration,
            state.inspection,
            state.overrides,
            state.dimension_overrides,
        ) == (
            history.current.record,
            history.current.registration,
            history.current.inspection,
            history.current.overrides,
            history.current.dimension_overrides,
        ):
            return
        history.past.append(history.current)
        state.label = label
        history.current = state
        history.future.clear()
        self.history.save(history=history)
        self.synchronize(state=state)

    def synchronize(self, *, state: ProjectState) -> None:
        """
        Publish compatibility artifacts derived from the journal's current decision.
        """
        self.write_record(record=state.record)
        destination = self.directory / ProjectFile.CANDIDATES
        temporary = destination.with_suffix(".partial")
        temporary.write_text(
            json.dumps(
                DataclassJSONSerializer.to_json(instance=state.registration), indent=2
            )
        )
        temporary.replace(destination)
        accepted = state.record.accepted_candidate
        transform = self.directory / ProjectFile.TRANSFORM
        if accepted is None:
            transform.unlink(missing_ok=True)
            return
        temporary = transform.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.save(stream, state.registration.candidates[accepted].scan_T_root)
        temporary.replace(transform)

    def update_inspection(self, *, inspection: InspectionState, label: str) -> None:
        """
        Save validated inspection settings as an undoable action.
        """
        state = deepcopy(self.current_state)
        if inspection.saved_assemblies != state.inspection.saved_assemblies:
            snapshot = self.snapshot
            frozen = {item.identifier for item in snapshot.assemblies}
            bodies = {mesh.body_name for mesh in snapshot.meshes}
            if any(
                item.identifier in frozen or not set(item.body_names) <= bodies
                for item in inspection.saved_assemblies
            ):
                raise InvalidEvidence(
                    reason="Saved assemblies must use existing bodies and distinct identities."
                )
        state.inspection = inspection
        self.publish(state=state, label=label)

    def update_registration(self, *, result: RegistrationResult, role: str) -> None:
        """
        Replace proposals reversibly and clear the preceding acceptance.
        """
        state = deepcopy(self.current_state)
        if not state.registration.candidates and result.candidates:
            state.inspection.view.coordinate_frame = InspectionFrame.WORLD.value
        state.registration = result
        state.record.accepted_candidate = None
        state.record.registration_geometry = role
        state.inspection.view.selected_candidate = 0 if result.candidates else None
        self.publish(state=state, label="Generate registration candidates")

    def undo(self) -> None:
        """
        Restore the preceding project decision, preserving its redo revision.
        """
        history = self.history.load()
        if not history.past:
            return
        history.future.append(history.current)
        history.current = history.past.pop()
        self.history.save(history=history)
        self.synchronize(state=history.current)

    def redo(self) -> None:
        """
        Reapply the most recently undone decision.
        """
        history = self.history.load()
        if not history.future:
            return
        history.past.append(history.current)
        history.current = history.future.pop()
        self.history.save(history=history)
        self.synchronize(state=history.current)
