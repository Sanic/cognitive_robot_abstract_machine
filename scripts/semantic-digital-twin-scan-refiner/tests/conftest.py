"""
Isolate scan-refiner tests from ORM generation and large scan inputs.
"""

from __future__ import annotations

# %% Imports
import os
import shutil
from enum import StrEnum
import numpy as np
import pytest
import trimesh
from pathlib import Path

from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh, WorldSnapshot

from test.orm_interface_build import ORM_BUILD_OPTION, OrmBuild


# %% Test configuration
class BrowserTestEnvironment(StrEnum):
    """
    Optional existing JavaScript executable for testing browser controls.
    """

    NODE = "SCAN_REFINER_NODE"


@pytest.fixture
def javascript_runtime() -> Path:
    """
    Use an available Node runtime without installing browser-test dependencies.
    """
    executable = os.environ.get(BrowserTestEnvironment.NODE) or shutil.which("node")
    if executable is None:
        pytest.skip(
            "An existing Node runtime is required for browser-control unit tests."
        )
    return Path(executable)


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
    snapshot = WorldSnapshot(
        meshes=[BodyMesh(body_name="test/box", role=GeometryRole.VISUAL, mesh=mesh)]
    )
    source = tmp_path / "scan.npz"
    np.savez(
        source,
        points=mesh.vertices,
        colors=np.full(mesh.vertices.shape, 128, dtype=np.uint8),
    )
    return RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=source,
        snapshot=snapshot,
        model_source="fixture",
    )
