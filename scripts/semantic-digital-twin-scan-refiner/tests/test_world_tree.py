"""
Exercise the outliner snapshot with both supported world sources.
"""

# %% Imports
from pathlib import Path
import subprocess

from scan_refiner.project import RefinementProject
from semantic_digital_twin.adapters.urdf import URDFParser
from scan_refiner.world import WorldSnapshot


# %% Imported topology
def test_urdf_snapshot_round_trips_geometry_free_root(tmp_path: Path) -> None:
    """
    Keep the fixture's joints and its geometry-free link in the archive.
    """
    urdf = Path(__file__).parent / "dataset" / "source_world.urdf"
    snapshot = WorldSnapshot.from_world(
        world=URDFParser.from_file(str(urdf)).create_world()
    )
    path = tmp_path / "world.npz"
    snapshot.save(path=path)
    loaded = WorldSnapshot.read(path=path)

    parents = {node.name: node.parent_name for node in loaded.kinematic_nodes}
    assert parents == {
        "test/root": None,
        "test/box": "test/root",
        "test/door": "test/root",
        "test/handle": "test/box",
    }
    assert "test/root" not in {item.body_name for item in loaded.meshes}


def test_python_kitchen_provider_preserves_connections() -> None:
    """
    Capture every joint of the directly modelled kitchen.
    """
    from_provider = WorldSnapshot.from_provider(
        identifier="semantic_digital_twin.predefined_maps.kitchen_environment:KitchenEnvironment"
    )
    nodes = {node.name: node.parent_name for node in from_provider.kinematic_nodes}
    assert {name for name, parent in nodes.items() if parent is None} == {"root"}
    assert {item.body_name for item in from_provider.meshes} <= nodes.keys()
    assert all(parent is None or parent in nodes for parent in nodes.values())
    assert any(parent and nodes[parent] for parent in nodes.values())


def test_snapshot_without_parent_links_remains_readable(
    small_project: RefinementProject,
) -> None:
    """
    An archive with only mesh evidence still exposes its original geometry.
    """
    assert small_project.snapshot.kinematic_nodes == []
    assert [mesh.body_name for mesh in small_project.snapshot.meshes] == ["test/box"]


# %% Browser selection
def test_branch_scope_uses_only_descendant_geometry(javascript_runtime: Path) -> None:
    """
    A geometry-free parent still identifies its descendant surfaces exactly once.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/world_tree.cjs"),
            str(root / "scan_refiner/web/world_tree.js"),
        ],
        check=True,
    )
