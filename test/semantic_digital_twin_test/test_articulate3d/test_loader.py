import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from semantic_digital_twin.adapters.articulate3d_dataset.exceptions import (
    Articulate3DSceneNotFoundError,
)
from semantic_digital_twin.adapters.articulate3d_dataset.loader import (
    Articulate3DDatasetLoader,
)
from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DAnnotationFile,
    Articulate3DArticulation,
    Articulate3DPart,
    Articulate3DScene,
)
from semantic_digital_twin.adapters.scannetpp_dataset.loader import (
    ScanNetPPDatasetLoader,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)

# %% fixtures


@pytest.fixture
def resources() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture
def scannetpp_dataset_loader() -> ScanNetPPDatasetLoader:
    return ScanNetPPDatasetLoader(
        directory=Path(__file__).parent.parent / "test_scannetpp" / "resources"
    )


@pytest.fixture
def cabinet(resources, scannetpp_dataset_loader) -> Articulate3DScene:
    """
    A cabinet with a door, a handle turning on the door, and a drawer annotated twice,
    whose second annotation names a base that is no part.

    A flap turns on the drawer, with a knob on the flap, and a faceless hinge sits on
    the door. A lock is articulated without being a part. The handle lists the cabinet
    and then the door as its bases; a faceless lid lists the knob and then the door,
    neither of them its parent in the part hierarchy.
    """
    loader = Articulate3DDatasetLoader(
        directory=resources, scannetpp_dataset_loader=scannetpp_dataset_loader
    )
    return loader.load("table_and_cup")


@pytest.fixture
def cabinet_world(cabinet) -> World:
    return cabinet.create_world()


@pytest.fixture(scope="module")
def kitchen() -> Articulate3DScene:
    """
    A real ScanNet++ kitchen with its Articulate3D annotations, read from a local copy
    of both datasets.
    """
    datasets = Path(
        "/media/itsme/93a4dee6-6389-4190-bde7-497ed57c6fc2/home/me/datasets"
    )
    if not datasets.is_dir():
        pytest.skip("ScanNet++ and Articulate3D are not downloaded on this machine")
    loader = Articulate3DDatasetLoader(
        directory=datasets / "articulate3d",
        scannetpp_dataset_loader=ScanNetPPDatasetLoader(
            directory=datasets / "scannet++_articulate3d_compatible"
        ),
    )
    return loader.load("f9f95681fd")


def articulation_of(scene: Articulate3DScene, part_id: int) -> Articulate3DArticulation:
    """
    :return: The part's last annotated articulation, the one that moves it.
    """
    *_, articulation = [
        articulation
        for articulation in scene.articulations
        if articulation.part_id == part_id
    ]
    return articulation


# %% articulated parts


def test_rotating_part_turns_about_its_axis_through_its_origin(cabinet, cabinet_world):
    door = cabinet_world.get_body_by_name("door.1.1.2_1")
    articulation = articulation_of(cabinet, part_id=2)
    angle = articulation.upper_limit

    door.parent_connection.position = angle

    scan_T_door = np.eye(4)
    scan_T_door[:3, :3] = Rotation.from_rotvec(angle * articulation.axis).as_matrix()
    scan_T_door[:3, 3] = articulation.origin - scan_T_door[:3, :3] @ articulation.origin
    np.testing.assert_allclose(
        cabinet_world.compute_forward_kinematics_np(cabinet_world.root, door),
        scan_T_door,
        atol=1e-12,
    )


def test_sliding_part_moves_along_its_axis(cabinet, cabinet_world):
    drawer = cabinet_world.get_body_by_name("drawer.1.1.2_2")
    articulation = articulation_of(cabinet, part_id=4)
    distance = articulation.upper_limit

    drawer.parent_connection.position = distance

    np.testing.assert_allclose(
        cabinet_world.compute_forward_kinematics_np(cabinet_world.root, drawer)[:3, 3],
        distance * articulation.axis,
    )


def test_articulated_part_moves_relative_to_its_base_part(cabinet_world):
    door = cabinet_world.get_body_by_name("door.1.1.2_1")

    assert isinstance(door.parent_connection, RevoluteConnection)
    assert door.parent_connection.parent is cabinet_world.get_body_by_name(
        "cabinet.1.1"
    )


def test_articulated_part_can_move_on_another_articulated_part(cabinet_world):
    handle = cabinet_world.get_body_by_name("handle.1.1.2_1.3_1")

    assert isinstance(handle.parent_connection, RevoluteConnection)
    assert handle.parent_connection.parent is cabinet_world.get_body_by_name(
        "door.1.1.2_1"
    )


def test_articulated_part_whose_base_is_no_part_moves_relative_to_the_root(
    cabinet_world,
):
    drawer = cabinet_world.get_body_by_name("drawer.1.1.2_2")

    assert drawer.parent_connection.parent is cabinet_world.root


def test_articulated_part_is_limited_to_its_range(cabinet, cabinet_world):
    door = cabinet_world.get_body_by_name("door.1.1.2_1")
    articulation = articulation_of(cabinet, part_id=2)
    limits = door.parent_connection.dof.limits

    assert (limits.lower.position, limits.upper.position) == (
        articulation.lower_limit,
        articulation.upper_limit,
    )


def test_last_articulation_of_a_part_moves_it(cabinet_world):
    drawer = cabinet_world.get_body_by_name("drawer.1.1.2_2")

    assert isinstance(drawer.parent_connection, PrismaticConnection)


def test_articulation_of_a_part_without_annotation_is_left_out(cabinet_world):
    assert {
        str(connection.child.name)
        for connection in cabinet_world.connections
        if isinstance(connection, ActiveConnection1DOF)
    } == {
        "door.1.1.2_1",
        "handle.1.1.2_1.3_1",
        "drawer.1.1.2_2",
        "flap.1.1.2_2.3_1",
        "lid.1.1.2_3",
    }


# %% several bases


def test_of_several_bases_the_hierarchy_parent_is_the_base(cabinet_world):
    handle = cabinet_world.get_body_by_name("handle.1.1.2_1.3_1")

    assert handle.parent_connection.parent is cabinet_world.get_body_by_name(
        "door.1.1.2_1"
    )


def test_of_several_bases_none_the_hierarchy_parent_the_first_is_the_base(
    cabinet_world,
):
    lid = cabinet_world.get_body_by_name("lid.1.1.2_3")

    assert lid.parent_connection.parent is cabinet_world.get_body_by_name(
        "knob.1.1.2_2.3_1.4_1"
    )


# %% fixed parts


def test_part_without_articulation_is_fixed_to_the_part_above_it(cabinet_world):
    knob = cabinet_world.get_body_by_name("knob.1.1.2_2.3_1.4_1")

    assert isinstance(knob.parent_connection, FixedConnection)
    assert knob.parent_connection.parent is cabinet_world.get_body_by_name(
        "flap.1.1.2_2.3_1"
    )


def test_part_without_articulation_moves_with_the_part_above_it(cabinet, cabinet_world):
    door = cabinet_world.get_body_by_name("door.1.1.2_1")
    hinge = cabinet_world.get_body_by_name("hinge.1.1.2_1.3_2")

    door.parent_connection.position = articulation_of(cabinet, part_id=2).upper_limit

    np.testing.assert_allclose(
        cabinet_world.compute_forward_kinematics_np(cabinet_world.root, hinge),
        cabinet_world.compute_forward_kinematics_np(cabinet_world.root, door),
    )


def test_part_at_the_top_of_the_hierarchy_is_fixed_to_the_root(cabinet_world):
    cabinet_body = cabinet_world.get_body_by_name("cabinet.1.1")

    assert isinstance(cabinet_body.parent_connection, FixedConnection)
    assert cabinet_body.parent_connection.parent is cabinet_world.root


def test_faces_no_part_covers_stay_with_the_scan(cabinet, cabinet_world):
    [surface] = cabinet_world.root.visual.shapes
    scan_mesh = cabinet.scannetpp_scene.mesh

    np.testing.assert_array_equal(
        np.unique(surface.mesh.vertices, axis=0),
        np.unique(scan_mesh.vertices[scan_mesh.faces[4]], axis=0),
    )


def test_create_world_describes_each_part_body_by_its_part_name(cabinet, cabinet_world):
    assert {
        (str(annotation.root.name), annotation.description)
        for annotation in cabinet_world.get_semantic_annotations_by_type(
            NaturalLanguageDescription
        )
    } == {(part.label, part.part_name) for part in cabinet.parts}


def test_create_world_records_no_change_of_a_body_s_shapes(cabinet):
    """
    The scan is cut before it is in the world: a change of a body's shapes is recorded
    as the whole body serialized before and after it.
    """
    world = cabinet.create_world()

    assert not any(
        isinstance(modification, AttributeUpdateModification)
        for block in world.get_world_model_manager().model_modification_blocks
        for modification in block.modifications
    )


# %% loading


def test_load_of_a_scene_without_annotations_raises(tmp_path):
    loader = Articulate3DDatasetLoader(
        directory=tmp_path,
        scannetpp_dataset_loader=ScanNetPPDatasetLoader(directory=tmp_path),
    )

    with pytest.raises(Articulate3DSceneNotFoundError):
        loader.load("missing_scene")


def test_load_ignores_the_face_count_the_annotations_record(
    tmp_path, resources, scannetpp_dataset_loader, cabinet
):
    for annotation_file in Articulate3DAnnotationFile:
        shutil.copy(
            annotation_file.path(resources, "table_and_cup"),
            annotation_file.path(tmp_path, "table_and_cup"),
        )
    parts_path = Articulate3DAnnotationFile.PARTS.path(tmp_path, "table_and_cup")
    parts = json.loads(parts_path.read_text())
    parts["data"]["stats"]["totalFaces"] -= 1
    parts_path.write_text(json.dumps(parts))
    loader = Articulate3DDatasetLoader(
        directory=tmp_path, scannetpp_dataset_loader=scannetpp_dataset_loader
    )

    scene = loader.load("table_and_cup")

    for part, expected in zip(scene.parts, cabinet.parts, strict=True):
        np.testing.assert_array_equal(part.face_indices, expected.face_indices)


def test_scene_ids_lists_every_scene_a_directory_annotates(tmp_path, resources):
    for scene_id in ["b_scene", "a_scene"]:
        shutil.copy(
            Articulate3DAnnotationFile.PARTS.path(resources, "table_and_cup"),
            Articulate3DAnnotationFile.PARTS.path(tmp_path, scene_id),
        )

    assert Articulate3DAnnotationFile.PARTS.scene_ids(tmp_path) == [
        "a_scene",
        "b_scene",
    ]


def test_parts_are_read_from_their_file_alone(resources, cabinet):
    parts = Articulate3DPart.all_in_file(
        Articulate3DAnnotationFile.PARTS.path(resources, "table_and_cup")
    )

    assert [part.label for part in parts] == [part.label for part in cabinet.parts]
    for part, expected in zip(parts, cabinet.parts, strict=True):
        np.testing.assert_array_equal(part.face_indices, expected.face_indices)


# %% real scene


def test_every_articulated_kitchen_part_gets_a_movable_connection(kitchen):
    world = kitchen.create_world()
    part_ids = {part.part_id for part in kitchen.parts}

    assert len(
        [
            connection
            for connection in world.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
    ) == len(
        {
            articulation.part_id
            for articulation in kitchen.articulations
            if articulation.part_id in part_ids
        }
    )
