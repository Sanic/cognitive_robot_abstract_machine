"""
Assemblies retain exact geometric members without reopening world sources.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess

import numpy as np
import pytest
import trimesh
from scan_refiner.assemblies import AssemblyMembership
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.world import BodyMesh
from scan_refiner.world import WorldSnapshot
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    KitchenIsland,
)


# %% Declared membership
def test_semantic_assembly_follows_its_declared_branch(tmp_path: Path) -> None:
    """
    The cabinet and its fixed child share the assembly; the neighbouring door does not.
    """
    source = Path(__file__).parent / "dataset/source_world.urdf"
    world = URDFParser.from_file(file_path=str(source)).create_world()
    root = world.get_body_by_name(PrefixedName.from_string("test/box"))
    with world.modify_world():
        world.add_semantic_annotation(KitchenIsland(root=root))
    snapshot = WorldSnapshot.from_world(world=world)
    path = tmp_path / "snapshot.npz"
    snapshot.save(path=path)
    (assembly,) = WorldSnapshot.read(path=path).assemblies
    assert assembly.label == "Kitchen island"
    assert assembly.body_names == ["test/box", "test/handle"]


# %% Saved groups
def test_saved_assembly_undo_redo_without_changing_baseline(tmp_path: Path) -> None:
    """
    Only existing members can be saved and inspection history restores the group.
    """
    names = ["test/left", "test/right"]
    snapshot = WorldSnapshot(
        meshes=[
            BodyMesh(
                body_name=name, role=GeometryRole.VISUAL, mesh=trimesh.creation.box()
            )
            for name in names
        ]
    )
    scan = tmp_path / "scan.npz"
    np.savez(scan, points=snapshot.meshes[0].mesh.vertices)
    project = RefinementProject.create(
        directory=tmp_path / "project",
        scan_path=scan,
        snapshot=snapshot,
        model_source="fixture",
    )
    digest = project.record.world_sha256
    inspection = project.current_state.inspection
    assembly = AssemblyMembership(
        identifier="saved:workcell", label="Workcell", body_names=names
    )
    inspection.saved_assemblies.append(assembly)
    inspection.view.selected_assembly = assembly.identifier
    project.update_inspection(inspection=inspection, label="Save workcell")
    assert project.current_state.inspection.saved_assemblies == [assembly]
    project.undo()
    assert project.current_state.inspection.saved_assemblies == []
    project.redo()
    assert (
        project.current_state.inspection.view.selected_assembly == assembly.identifier
    )
    assert project.snapshot.assemblies == []
    assert project.record.world_sha256 == digest
    invalid = project.current_state.inspection
    invalid.saved_assemblies[0].body_names.append("test/unknown")
    with pytest.raises(InvalidEvidence):
        project.update_inspection(inspection=invalid, label="Save invalid membership")
    assert project.current_state.inspection.saved_assemblies == [assembly]


# %% Browser handoff
def test_browser_copies_assembly_members_and_their_combined_crop(
    javascript_runtime: Path,
) -> None:
    """
    An assembly handoff retains its member names and a direct metric region query.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/assembly_context.cjs"),
            str(root / "scan_refiner/web"),
        ],
        check=True,
    )
