"""
Check physical distance interpretation and portable viewer controls.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import trimesh

from scan_refiner.inspection import SurfaceComparison, InspectionViewer
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh, WorldSnapshot


# %% Browser dependencies
@dataclass(kw_only=True)
class ScriptSources(HTMLParser):
    """
    Collect external script dependencies without matching bundled source strings.
    """

    external_sources: list[str] = field(default_factory=list)
    """
    URLs actually requested by script elements in the generated page.
    """

    def __post_init__(self) -> None:
        """
        Initialize the HTML parser's tokenization state.
        """
        super().__init__()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """
        Record script source attributes beyond :meth:`HTMLParser.handle_starttag`.
        """
        if tag != "script":
            return
        self.external_sources.extend(
            value for key, value in attrs if key == "src" and value is not None
        )


# %% Distance interpretation
def test_comparison_uses_triangle_surfaces_and_transform_direction() -> None:
    """
    A point above a triangle interior is measured to the face, not its vertices.
    """
    mesh = trimesh.creation.box()
    scan_T_root = np.eye(4)
    scan_T_root[:3, 3] = (4, -2, 1)
    scan_points = np.array([[4, -2, 1.7], [4.1, -2, 1.5]])
    distances = SurfaceComparison().scan_to_model(
        scan_points=scan_points, root_mesh=mesh, scan_T_root=scan_T_root
    )
    np.testing.assert_allclose(distances, (0.2, 0.0), atol=1e-12)


# %% Viewer generation
def test_viewer_is_self_contained_and_marks_provisional_registration(
    tmp_path: Path,
) -> None:
    """
    Review does not require internet access or imply a candidate was accepted.
    """
    mesh = trimesh.creation.box()
    source = tmp_path / "scan.npz"
    np.savez(source, points=mesh.vertices)
    project = RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=source,
        snapshot=WorldSnapshot(
            meshes=[BodyMesh(body_name="room/box", role=GeometryRole.VISUAL, mesh=mesh)]
        ),
        model_source="test",
    )
    path = InspectionViewer(project=project).write()
    html = path.read_text()
    scripts = ScriptSources()
    scripts.feed(html)
    assert scripts.external_sources == []
    assert "No registration candidates" in html
    assert "camera" in html
    assert project.record.accepted_candidate is None
