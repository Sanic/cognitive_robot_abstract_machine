"""
Exercise actual browser controls with Plotly's encoded coordinate buffers.
"""

from __future__ import annotations

# %% Imports
import subprocess
from pathlib import Path

from scan_refiner.inspection import InspectionViewer
from scan_refiner.project import RefinementProject


# %% Coordinate preservation
def test_geometry_update_preserves_the_live_camera_after_an_orbit(
    small_project: RefinementProject, tmp_path: Path, javascript_runtime: Path
) -> None:
    """
    Review can redraw geometry while Plotly's layout still holds an older camera.
    """
    root = Path(__file__).parents[1]
    figure_path = tmp_path / "figure.json"
    figure_path.write_text(
        InspectionViewer(project=small_project).build_scene().figure.to_json()
    )
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/viewer_controls.cjs"),
            str(root / "scan_refiner/web"),
            str(figure_path),
            "live-camera",
        ],
        check=True,
    )


def test_alignment_switch_preserves_encoded_mesh_vertices(
    small_project: RefinementProject, tmp_path: Path, javascript_runtime: Path
) -> None:
    """
    Changing candidates retains every vertex from the original Plotly wire format.
    """
    figure_path = tmp_path / "figure.json"
    figure_path.write_text(
        InspectionViewer(project=small_project).build_scene().figure.to_json()
    )
    script = Path(__file__).parent / "web" / "viewer_controls.cjs"
    assets = Path(__file__).parent.parent / "scan_refiner" / "web"
    subprocess.run(
        [
            str(javascript_runtime),
            str(script),
            str(assets),
            str(figure_path),
            "coordinates",
        ],
        check=True,
    )


def test_geometry_switches_do_not_overlap_renderer_updates(
    small_project: RefinementProject, tmp_path: Path, javascript_runtime: Path
) -> None:
    """
    Rapid geometry choices reach the last requested state one renderer update at a time.
    """
    figure_path = tmp_path / "figure.json"
    figure_path.write_text(
        InspectionViewer(project=small_project).build_scene().figure.to_json()
    )
    script = Path(__file__).parent / "web" / "viewer_controls.cjs"
    assets = Path(__file__).parent.parent / "scan_refiner" / "web"
    subprocess.run(
        [
            str(javascript_runtime),
            str(script),
            str(assets),
            str(figure_path),
            "geometry",
        ],
        check=True,
    )
