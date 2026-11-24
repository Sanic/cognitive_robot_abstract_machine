"""
Exercise durable undo/redo without copying scan or mesh evidence.
"""

from __future__ import annotations

# %% Imports
from copy import deepcopy

import pytest

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.inspection_state import Measurement, SavedView
from scan_refiner.project import RefinementProject
from scan_refiner.records import ProjectFile, RegistrationCandidate, RegistrationResult


# %% Inspection history
def test_inspection_undo_redo_survives_project_reopening(
    small_project: RefinementProject,
) -> None:
    """
    Undo restores the prior view and redo reapplies it after a new process opens it.
    """
    state = small_project.current_state
    changed = deepcopy(state.inspection)
    changed.view.geometry_mode = "collision"
    small_project.update_inspection(inspection=changed, label="Show collision")
    reopened = RefinementProject(directory=small_project.directory)
    assert reopened.current_state.inspection.view.geometry_mode == "collision"
    reopened.undo()
    assert (
        reopened.current_state.inspection.view.geometry_mode
        == state.inspection.view.geometry_mode
    )
    reopened.redo()
    assert reopened.current_state.inspection.view.geometry_mode == "collision"


def test_new_action_after_undo_clears_redo_branch(
    small_project: RefinementProject,
) -> None:
    """
    A new decision retains earlier undo state while replacing the abandoned redo path.
    """
    changed = deepcopy(small_project.current_state.inspection)
    changed.view.geometry_mode = "collision"
    small_project.update_inspection(inspection=changed, label="Show collision")
    small_project.undo()
    changed = deepcopy(small_project.current_state.inspection)
    changed.view.point_size = 3.0
    small_project.update_inspection(inspection=changed, label="Point size")
    assert small_project.history.load().future == []
    small_project.undo()
    assert small_project.current_state.inspection.view.point_size == 2.0


def test_history_preserves_immutable_geometry_and_scan(
    small_project: RefinementProject,
) -> None:
    """
    View history modifies metadata without rewriting the frozen evidence files.
    """
    paths = [
        small_project.directory / ProjectFile.SCAN,
        small_project.directory / ProjectFile.WORLD,
    ]
    before = [path.read_bytes() for path in paths]
    changed = deepcopy(small_project.current_state.inspection)
    changed.view.mesh_opacity = 0.5
    small_project.update_inspection(inspection=changed, label="Opacity")
    small_project.undo()
    assert [path.read_bytes() for path in paths] == before


def test_measurement_and_bookmark_are_undoable(
    small_project: RefinementProject,
) -> None:
    """
    Annotations and saved views participate in the same project history.
    """
    changed = deepcopy(small_project.current_state.inspection)
    measurement = Measurement(label="Width", start=[0, 0, 0], end=[3, 4, 0])
    changed.measurements.append(measurement)
    changed.bookmarks.append(
        SavedView(label="Countertop", note="Inspect rim", view=deepcopy(changed.view))
    )
    small_project.update_inspection(inspection=changed, label="Save inspection")
    assert small_project.current_state.inspection.measurements[0].distance == 5.0
    small_project.undo()
    assert small_project.current_state.inspection.measurements == []
    assert small_project.current_state.inspection.bookmarks == []
    small_project.redo()
    assert small_project.current_state.inspection.bookmarks[0].note == "Inspect rim"


def test_registration_acceptance_can_be_undone(
    small_project: RefinementProject,
) -> None:
    """
    Undoing acceptance restores the previous decision and removes the derived transform.
    """
    candidate = RegistrationCandidate(
        scan_T_root=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        support_fraction=1.0,
        median_distance=0.0,
    )
    small_project.update_registration(
        result=RegistrationResult(candidates=[candidate]), role="visual"
    )
    small_project.accept(candidate_index=0)
    assert (small_project.directory / ProjectFile.TRANSFORM).exists()
    small_project.undo()
    assert small_project.record.accepted_candidate is None
    assert not (small_project.directory / ProjectFile.TRANSFORM).exists()
    small_project.redo()
    assert small_project.record.accepted_candidate == 0


def test_replacing_registration_retains_previous_candidates(
    small_project: RefinementProject,
) -> None:
    """
    A rerun is a reversible revision rather than destructive candidate replacement.
    """
    original = RegistrationResult(
        candidates=[
            RegistrationCandidate(
                scan_T_root=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                support_fraction=1.0,
                median_distance=0.0,
            )
        ]
    )
    replacement = RegistrationResult(
        candidates=[
            RegistrationCandidate(
                scan_T_root=[[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
                support_fraction=0.9,
                median_distance=0.01,
            )
        ]
    )
    small_project.update_registration(result=original, role="visual")
    small_project.accept(candidate_index=0)
    small_project.update_registration(result=replacement, role="collision")
    small_project.undo()
    assert small_project.registration == original
    assert small_project.record.accepted_candidate == 0
    assert small_project.record.registration_geometry == "visual"


def test_invalid_view_does_not_create_a_history_entry(
    small_project: RefinementProject,
) -> None:
    """
    Malformed region and camera settings fail before publishing a revision.
    """
    changed = deepcopy(small_project.current_state.inspection)
    changed.view.point_budget = 0
    with pytest.raises(InvalidEvidence):
        small_project.update_inspection(inspection=changed, label="Invalid budget")
    assert small_project.history.load().past == []


# %% Camera persistence
def test_camera_round_trips_with_saved_view(small_project: RefinementProject) -> None:
    """
    Camera state retains eye coordinates independently of rendering dictionaries.
    """
    changed = deepcopy(small_project.current_state.inspection)
    changed.view.camera.eye = [1.0, 2.0, 3.0]
    small_project.update_inspection(inspection=changed, label="Move camera")
    reopened = RefinementProject(directory=small_project.directory)
    assert reopened.current_state.inspection.view.camera.eye == [1.0, 2.0, 3.0]


# %% Journal authority
def test_journal_wins_over_stale_compatibility_metadata(
    small_project: RefinementProject,
) -> None:
    """
    A stale legacy record cannot override an atomically published revision.
    """
    candidate = RegistrationCandidate(
        scan_T_root=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        support_fraction=1.0,
        median_distance=0.0,
    )
    small_project.update_registration(
        result=RegistrationResult(candidates=[candidate]), role="visual"
    )
    stale = small_project.record
    small_project.accept(candidate_index=0)
    small_project.write_record(record=stale)
    assert (
        RefinementProject(directory=small_project.directory).record.accepted_candidate
        == 0
    )


# %% Inspection-frame persistence
def test_crop_frame_survives_axis_changes_and_undo(
    small_project: RefinementProject,
) -> None:
    """
    Changing display axes does not reinterpret the loaded query's frozen frame.
    """
    before = deepcopy(small_project.current_state.inspection)
    before.view.region_frame_T_scan = [
        [0, -1, 0, 4],
        [1, 0, 0, -2],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    before.view.region_lower = [0, 0, 0]
    before.view.region_upper = [1, 2, 3]
    small_project.update_inspection(inspection=before, label="Load oriented crop")
    changed = deepcopy(before)
    changed.view.coordinate_frame = "world"
    changed.view.frame_yaw = 30
    small_project.update_inspection(inspection=changed, label="Orient view")
    assert (
        small_project.current_state.inspection.view.region_frame_T_scan
        == before.view.region_frame_T_scan
    )
    small_project.undo()
    assert small_project.current_state.inspection == before
