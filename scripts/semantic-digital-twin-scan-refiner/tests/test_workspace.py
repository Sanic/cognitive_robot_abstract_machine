"""
Check the staged shell and independent inspection/evaluation controls.
"""

from __future__ import annotations

# %% Imports
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from html.parser import HTMLParser
from scan_refiner.inspection import InspectionViewer
from scan_refiner.project import RefinementProject
from scan_refiner.point_records import PointRecordLayout


# %% HTML structure
@dataclass(kw_only=True)
class WorkspaceElements(HTMLParser):
    """
    Collect actual element identities and enabled stage navigation.
    """

    identifiers: list[str] = field(default_factory=list)
    """
    Element identities present in the generated document.
    """

    stages: set[str] = field(default_factory=set)
    """
    Navigation targets exposed as implemented stages.
    """

    def __post_init__(self) -> None:
        """
        Initialize the parser's tokenization state.
        """
        super().__init__()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """
        Collect controls in addition to :meth:`HTMLParser.handle_starttag`.
        """
        attributes = dict(attrs)
        if "id" in attributes:
            self.identifiers.append(attributes["id"])
        if "data-stage" in attributes:
            self.stages.add(attributes["data-stage"])


# %% Shared shell
def test_workspace_has_single_scene_and_explicit_available_stages(
    small_project: RefinementProject,
) -> None:
    """
    Navigation reuses one scene and does not pretend future stages are available.
    """
    page = InspectionViewer(project=small_project).write().read_text()
    elements = WorkspaceElements()
    elements.feed(page)
    assert elements.stages == {"prepare", "align", "inspect", "refine"}
    assert elements.identifiers.count("scene") == 1
    assert len(elements.identifiers) == len(set(elements.identifiers))
    assert "__STYLES__" not in page


def test_workspace_separates_evaluation_budget_from_display_budget(
    small_project: RefinementProject,
) -> None:
    """
    Display quality and analysis settings are distinct user choices.
    """
    page = InspectionViewer(project=small_project).write().read_text()
    elements = WorkspaceElements()
    elements.feed(page)
    assert {
        "coordinate-frame",
        "frame-yaw",
        "draw-box",
        "point-budget",
        "comparison-budget",
        "start-comparison",
        "start-registration",
        "jobs-drawer",
    } <= set(elements.identifiers)


# %% Preview-only comparisons
def test_workspace_can_decode_job_records_without_scan_index(
    small_project: RefinementProject,
) -> None:
    """
    A preview-only project still advertises the binary layout of analysis results.
    """
    scene = InspectionViewer(project=small_project).build_scene()
    assert scene.state.point_layout == PointRecordLayout().metadata


# %% Directional results workspace
def test_results_have_dedicated_inspector_and_sampling_controls(
    small_project: RefinementProject,
) -> None:
    """
    Results remain separate from crop controls and retain explicit sample budgets.
    """
    page = InspectionViewer(project=small_project).write().read_text()
    elements = WorkspaceElements()
    elements.feed(page)
    assert {
        "comparison-report",
        "report-body-list",
        "report-download",
        "comparison-model-budget",
    } <= set(elements.identifiers)
    assert 'data-inspector-panel="results"' in page


# %% Inspector task navigation
def test_switching_inspector_resets_previous_scroll_position(
    javascript_runtime: str,
) -> None:
    """
    Opening results from a long region form must show the report heading.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            javascript_runtime,
            str(root / "tests/web/inspector_scroll.cjs"),
            str(root / "scan_refiner/web/workspace.js"),
        ],
        check=True,
    )
