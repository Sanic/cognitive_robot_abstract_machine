"""
Geometric regression tests for the predefined kitchen environment.
"""

from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
    WindowWallPart,
)
from semantic_digital_twin.adapters.world_mesh_exporter import WorldMeshExtractor
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Cooktop,
    CounterTop,
    KitchenIsland,
    WallPanel,
)
from semantic_digital_twin.world import World

# %% Countertop membership


@pytest.fixture
def kitchen_world() -> World:
    """
    Build the predefined kitchen with its modelled semantic relationships.
    """
    return KitchenEnvironment().create_world()


@pytest.mark.parametrize(
    "body_name",
    [
        "module_1_cabinet",
        "module_1_face_plate",
        "module_1_door",
        "module_1_handle",
        "dishwasher",
        "dishwasher_door",
        "dishwasher_handle",
    ],
)
def test_countertop_context_includes_all_under_counter_parts(
    kitchen_world: World, body_name: str
) -> None:
    """
    All under-counter modules and their nested parts belong to the worktop context.
    """
    snapshot = WorldMeshExtractor().extract(kitchen_world)
    countertop = next(
        annotation
        for annotation in snapshot.semantic_annotations
        if annotation.name == "counter_top"
    )
    body = next(body for body in snapshot.body_meshes if body.name == body_name)

    assert countertop.identifier in body.context_semantic_annotation_identifiers


def test_module_1_face_plate_belongs_to_its_cabinet(kitchen_world: World) -> None:
    """
    The fixed panel shares the cabinet's context along with its door and handle.
    """
    cabinet = kitchen_world.get_semantic_annotation_by_name("module_1_cabinet")
    panel = kitchen_world.get_semantic_annotation_by_name("module_1_face_plate")

    assert panel.root.parent_kinematic_structure_entity is cabinet.root


def test_oven_tower_is_a_cabinet(kitchen_world: World) -> None:
    """
    The housing for the oven and drawers carries the general cabinet class.
    """
    tower = kitchen_world.get_semantic_annotation_by_name("oven_tower")

    assert type(tower) is Cabinet


def test_oven_tower_contains_its_oven_and_drawers(kitchen_world: World) -> None:
    """
    The cabinet retains its mounted drawers and contained oven.
    """
    tower = kitchen_world.get_semantic_annotation_by_name("oven_tower")

    assert tower.drawers == [
        kitchen_world.get_semantic_annotation_by_name("oven_side_drawer_left"),
        kitchen_world.get_semantic_annotation_by_name("oven_side_drawer_right"),
        kitchen_world.get_semantic_annotation_by_name("oven_center_drawer"),
    ]
    assert tower.objects == [
        kitchen_world.get_semantic_annotation_by_name(name)
        for name in [
            "oven_base_facing",
            "oven_top_plate",
            "oven_top_gap_facing",
            "oven",
        ]
    ]


# %% Wall-mounted strips


@pytest.mark.parametrize(
    "wall_part", [WindowWallPart.CHANNEL, WindowWallPart.LOWER_STRIP]
)
def test_wall_strips_are_attached_to_the_north_wall(
    kitchen_world: World, wall_part: WindowWallPart
) -> None:
    """
    Both fixed strips move with the wall they are mounted on.
    """
    wall = kitchen_world.get_semantic_annotation_by_name(WindowWallPart.WALL)
    panel = kitchen_world.get_semantic_annotation_by_name(wall_part)

    assert panel.root.parent_kinematic_structure_entity is wall.root


@pytest.mark.parametrize(
    "wall_part", [WindowWallPart.CHANNEL, WindowWallPart.LOWER_STRIP]
)
def test_wall_strips_export_wall_context_with_direct_panel_labels(
    kitchen_world: World, wall_part: WindowWallPart
) -> None:
    """
    Wall membership is contextual while each strip retains its own panel class.
    """
    snapshot = WorldMeshExtractor().extract(kitchen_world)
    annotations = {
        annotation.identifier: annotation
        for annotation in snapshot.semantic_annotations
    }
    wall = next(
        annotation
        for annotation in annotations.values()
        if annotation.name == WindowWallPart.WALL
    )
    panel = next(body for body in snapshot.body_meshes if body.name == wall_part)

    assert panel.context_semantic_annotation_identifiers == (wall.identifier,)
    assert [
        annotations[identifier].type_name
        for identifier in panel.direct_semantic_annotation_identifiers
    ] == [WallPanel.__name__]


# %% Cooking table cabinets


@pytest.mark.parametrize("side", ["left", "right"])
def test_cooking_table_modules_are_cabinets(kitchen_world: World, side: str) -> None:
    """
    Each drawer housing carries the general cabinet class.
    """
    cabinet = kitchen_world.get_semantic_annotation_by_name(f"cooking_mod_{side}")

    assert type(cabinet) is Cabinet


@pytest.mark.parametrize("side", ["left", "right"])
def test_cooking_table_cabinets_keep_their_drawers(
    kitchen_world: World, side: str
) -> None:
    """
    The table contains each cabinet with its drawer and nested handle.
    """
    table = kitchen_world.get_semantic_annotation_by_name("cooking_table")
    cabinet = kitchen_world.get_semantic_annotation_by_name(f"cooking_mod_{side}")
    drawer = kitchen_world.get_semantic_annotation_by_name(f"cooking_drawer_{side}")
    handle = kitchen_world.get_semantic_annotation_by_name(
        f"cooking_drawer_handle_{side}"
    )

    assert cabinet.root.parent_kinematic_structure_entity is table.root
    assert cabinet.drawers == [drawer]
    assert drawer.handles == [handle]


# %% mesh bounds


def visual_world_bounds(world: World, body_name: str) -> np.ndarray:
    """
    Return a body's visual mesh bounds in the world root frame.
    """
    body = world.get_body_by_name(body_name)
    mesh = body.visual.combined_mesh.copy()
    mesh.apply_transform(world.compute_forward_kinematics_np(world.root, body))
    return mesh.bounds


# %% handle placement


def test_module_1_has_fixed_face_plate_above_door() -> None:
    """
    Module 1 should have a 14.3 cm face plate separated from its door.
    """
    world = KitchenEnvironment().create_world()
    face_plate_bounds = visual_world_bounds(world, "module_1_face_plate")
    door_bounds = visual_world_bounds(world, "module_1_door")

    assert np.isclose(np.ptp(face_plate_bounds[:, 2]), 0.143)
    assert np.isclose(face_plate_bounds[0, 2] - door_bounds[1, 2], 0.005)


@pytest.mark.parametrize("body_name", ["module_1_face_plate", "module_1_door"])
def test_module_1_front_piece_is_59_5_centimeters_wide(body_name: str) -> None:
    """
    Each Module 1 front piece should be narrower than the cabinet carcass.
    """
    world = KitchenEnvironment().create_world()
    body = world.get_body_by_name(body_name)

    assert np.isclose(body.visual.combined_mesh.extents[1], 0.595)


def test_module_1_handle_top_matches_scan_inset() -> None:
    """
    The Module 1 handle should be inset from the shortened door's top edge.
    """
    world = KitchenEnvironment().create_world()
    handle_bounds = visual_world_bounds(world, "module_1_handle")
    door_bounds = visual_world_bounds(world, "module_1_door")

    assert np.isclose(door_bounds[1, 2] - handle_bounds[1, 2], 0.044)


@pytest.mark.parametrize(
    ("handle_name", "door_name"),
    [
        ("dishwasher_handle", "dishwasher_door"),
        ("oven_handle", "oven_door"),
        ("module_1_handle", "module_1_door"),
        ("oven_cabinet_handle", "oven_cabinet_door"),
    ],
)
def test_horizontal_door_handle_fits_within_door_width(
    handle_name: str, door_name: str
) -> None:
    """
    A full-width horizontal handle should not extend beyond either door edge.
    """
    world = KitchenEnvironment().create_world()
    handle_bounds = visual_world_bounds(world, handle_name)
    door_bounds = visual_world_bounds(world, door_name)

    assert handle_bounds[0, 0] >= door_bounds[0, 0]
    assert handle_bounds[1, 0] <= door_bounds[1, 0]


def test_fridge_door_handle_is_near_free_edge() -> None:
    """
    The vertical fridge handle should be inset from the edge opposite its hinge.
    """
    world = KitchenEnvironment().create_world()
    handle_bounds = visual_world_bounds(world, "fridge_door_handle")
    door_bounds = visual_world_bounds(world, "fridge_door")

    handle_center_x = handle_bounds[:, 0].mean()
    assert np.isclose(door_bounds[1, 0] - handle_center_x, 0.041)


# %% the kitchen island


def test_the_sideboard_is_a_kitchen_island_built_from_its_corpus_under_its_worktop() -> (
    None
):
    """
    The sideboard is the kitchen island: its corpus is the cabinet built into it and its
    worktop is the island's counter top, as in the other modelled kitchens.
    """
    world = KitchenEnvironment().create_world()
    [island] = world.get_semantic_annotations_by_type(KitchenIsland)

    assert island.cabinets == [
        world.get_semantic_annotation_by_name("sideboard_cabinet")
    ]
    assert island.counter_top is world.get_semantic_annotation_by_name("sideboard")
    assert isinstance(island.counter_top, CounterTop)


def test_the_island_worktop_still_carries_its_cooktop() -> None:
    """
    Naming the worktop a counter top must not lose what stands on it.
    """
    world = KitchenEnvironment().create_world()
    worktop = world.get_semantic_annotation_by_name("sideboard")

    assert worktop.get_objects_of_type(Cooktop) == [
        world.get_semantic_annotation_by_name("sideboard_cooktop")
    ]
