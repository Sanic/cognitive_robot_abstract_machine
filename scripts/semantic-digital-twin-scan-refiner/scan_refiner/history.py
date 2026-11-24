"""Atomic project revisions with durable undo and redo."""

from __future__ import annotations

# %% Imports
import json
from dataclasses import dataclass, field
from pathlib import Path

from krrood.adapters.json_serializer import DataclassJSONSerializer

from scan_refiner.dimensions import BodyDimensions
from scan_refiner.refinement import BodyTranslation
from scan_refiner.inspection_state import InspectionState
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.records import ProjectRecord, RegistrationResult


# %% Revision records
@dataclass(kw_only=True)
class ProjectState:
    """Mutable decisions referring to an immutable scan and world baseline."""

    record: ProjectRecord
    """Registration decision and baseline identities."""
    registration: RegistrationResult
    """Candidate placements retained across registration reruns."""
    inspection: InspectionState = field(default_factory=InspectionState)
    """Reversible inspection controls and annotations."""
    overrides: list[BodyTranslation] = field(default_factory=list)
    """Selected-body root-frame displacements relative to the frozen baseline."""
    dimension_overrides: list[BodyDimensions] = field(default_factory=list)
    """Cumulative positive root-axis deformations before separate body translations."""
    label: str = "Initial state"
    """Action that produced this revision."""


@dataclass(kw_only=True)
class ActionHistory:
    """Current revision and the two branches available for traversal."""

    current: ProjectState
    """Authoritative project state."""
    past: list[ProjectState] = field(default_factory=list)
    """Earlier states in chronological order."""
    future: list[ProjectState] = field(default_factory=list)
    """Undone states, with the next redo last."""


@dataclass(kw_only=True)
class ProjectHistory:
    """Persist small metadata snapshots independently of evidence artifacts."""

    path: Path
    """Atomic journal destination."""
    baseline: ProjectState | None = None
    """Migration state before the first journal entry exists."""

    def load(self) -> ActionHistory:
        """Restore revisions or expose the unsaved baseline."""
        if not self.path.exists():
            if self.baseline is None:
                raise InvalidEvidence(
                    reason="A missing journal requires an existing project baseline."
                )
            return ActionHistory(current=self.baseline)
        return DataclassJSONSerializer.from_json(
            data=json.loads(self.path.read_text()), clazz=ActionHistory
        )

    def save(self, *, history: ActionHistory) -> None:
        """Atomically publish a complete undo/redo revision chain."""
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=history), indent=2)
        )
        temporary.replace(self.path)
