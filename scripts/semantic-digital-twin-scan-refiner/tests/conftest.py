"""
Isolate scan-refiner tests from ORM generation and large scan inputs.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
import trimesh
from pathlib import Path

from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh, WorldSnapshot

from test.orm_interface_build import ORM_BUILD_OPTION, OrmBuild


# %% Test configuration
def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Accept the workspace's explicit no-build choice without loading world fixtures.
    """
    parser.addoption(ORM_BUILD_OPTION, choices=[OrmBuild.NEVER], default=OrmBuild.NEVER)


# %% Synthetic evidence
@pytest.fixture
def asymmetric_points() -> np.ndarray:
    """
    Return separated surfaces with different heights and a reproducible sampling.
    """
    generator = np.random.default_rng(7)
    points = generator.uniform((0, 0, 0), (4, 3, 2), size=(1800, 3))
    points[:600, 0] = 0
    points[600:1200, 1] = 0
    points[1200:, 2] = 0
    return points


@pytest.fixture
def small_project(tmp_path: Path) -> RefinementProject:
    """
    Provide an isolated geometry baseline and byte-colour preview for inspection tests.
    """
    mesh = trimesh.creation.box()
    snapshot = WorldSnapshot([BodyMesh("test/box", GeometryRole.VISUAL, mesh)])
    source = tmp_path / "scan.npz"
    np.savez(
        source,
        points=mesh.vertices,
        colors=np.full(mesh.vertices.shape, 128, dtype=np.uint8),
    )
    return RefinementProject.create(tmp_path / "project", source, snapshot, "fixture")
