"""
Heavy viewer work has visible, nested and reversible interaction locks.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess
from scan_refiner.inspection import InspectionViewer
from scan_refiner.project import RefinementProject


# %% Loading feedback
def test_activity_waits_for_paint_and_keeps_outer_operation_visible(
    javascript_runtime: Path,
) -> None:
    """
    Nested operations do not clear Undo feedback or unlock controls prematurely.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/viewer_activity.cjs"),
            str(root / "scan_refiner/web/plot_buffers.js"),
        ],
        check=True,
    )


# %% Deferred startup
def test_startup_status_precedes_the_inert_figure(
    small_project: RefinementProject,
) -> None:
    """
    The loading panel is present before embedded scripts can initialize WebGL.
    """
    page = InspectionViewer(project=small_project).write().read_text()
    assert (
        page.index('id="viewer-busy"')
        < page.index('<template id="viewer-figure">')
        < page.index("Plotly.newPlot(")
    )
    assert "globalThis.prepareViewerFigure=" in page
    assert "await globalThis.prepareViewerFigure?.()" in page
