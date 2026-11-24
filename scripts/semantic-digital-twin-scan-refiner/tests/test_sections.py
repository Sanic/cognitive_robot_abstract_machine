"""
Validate metric section outlines against a known frozen box.
"""

from __future__ import annotations

# %% Imports
import numpy as np
from scan_refiner.project import RefinementProject
from scan_refiner.sections import ModelSection


# %% Cross sections
def test_xy_section_lies_on_requested_scan_plane(
    small_project: RefinementProject,
) -> None:
    """
    A central slice of the fixture cube has four sides at its exact bounds.
    """
    section = ModelSection(project=small_project).build(
        plane="xy", position=0.0, candidate=None, role="visual"
    )
    vertices = np.array(section[0].segments).reshape(-1, 3)
    assert np.unique(vertices[:, 2]).tolist() == [0.0]
    assert vertices.min(axis=0).tolist() == [-0.5, -0.5, 0.0]
    assert vertices.max(axis=0).tolist() == [0.5, 0.5, 0.0]


# %% Inspection-frame sections
def test_section_plane_uses_selected_inspection_frame(
    small_project: RefinementProject,
) -> None:
    """
    A translated frame changes section coordinates without moving the baseline.
    """
    matrix = np.eye(4)
    matrix[2, 3] = 10
    outlines = ModelSection(project=small_project).build(
        plane="xy", position=10.0, candidate=None, role="visual", frame_T_scan=matrix
    )
    vertices = np.array(outlines[0].segments).reshape(-1, 3)
    assert np.unique(vertices[:, 2]).tolist() == [10.0]
