"""
Verify browser-frame transforms agree with root-to-scan placement conventions.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess
import json
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.inspection_state import InspectionState


# %% Rigid display frames
def test_browser_frame_round_trip_and_manual_yaw(javascript_runtime: Path) -> None:
    """
    World axes remove scanner pose while manual yaw preserves metric dimensions.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/coordinate_frame.cjs"),
            str(root / "scan_refiner/web/coordinate_frame.js"),
        ],
        check=True,
    )


# %% Browser serialization
def test_browser_matrix_encoding_round_trips(
    javascript_runtime: Path, tmp_path: Path
) -> None:
    """
    Nested matrix rows retain the collection shape accepted by the project serializer.
    """
    record = InspectionState()
    record.view.region_frame_T_scan = [
        [1, 0, 0, 4],
        [0, 1, 0, -2],
        [0, 0, 1, 136],
        [0, 0, 0, 1],
    ]
    source = tmp_path / "state.json"
    source.write_text(json.dumps(DataclassJSONSerializer.to_json(instance=record)))
    root = Path(__file__).parents[1]
    response = subprocess.check_output(
        [
            javascript_runtime,
            str(root / "tests/web/encode_inspection.cjs"),
            str(root / "scan_refiner/web"),
            str(source),
        ],
        text=True,
    )
    restored = DataclassJSONSerializer.from_json(
        data=json.loads(response), clazz=InspectionState
    )
    assert restored == record


# %% Atomic view changes
def test_view_changes_do_not_interleave_during_restoration(
    javascript_runtime: Path,
) -> None:
    """
    A second mutation cannot change the state being captured by an active action.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/inspection_action.cjs"),
            str(root / "scan_refiner/web/inspection_tools.js"),
        ],
        check=True,
    )


# %% Requested frame choice
def test_requested_frame_is_captured_before_controls_refresh(
    javascript_runtime: Path,
) -> None:
    """
    Orienting uses the selected axes even when busy-state refresh resets fields.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/frame_choice.cjs"),
            str(root / "scan_refiner/web/workspace.js"),
        ],
        check=True,
    )


# %% Result applicability
def test_workspace_hides_stale_comparison_statistics(javascript_runtime: Path) -> None:
    """
    Older result statistics cannot be shown as current after placement changes.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/stale_summary.cjs"),
            str(root / "scan_refiner/web/jobs.js"),
        ],
        check=True,
    )


# %% Hidden-stage rendering
def test_queued_resize_skips_hidden_stage(javascript_runtime: Path) -> None:
    """
    Stage navigation cannot poison the renderer queue by hiding a pending resize.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/hidden_resize.cjs"),
            str(root / "scan_refiner/web/plot_buffers.js"),
        ],
        check=True,
    )
