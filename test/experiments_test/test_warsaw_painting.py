"""
Painting a world's bodies so that one can be told from another.

A viewer showing a world in the colours it was scanned in shows a grey room. What
matters here is that painting says something true: bodies of one class look alike,
bodies nothing named look like nothing, and a world nobody asked to paint is left as it
was.
"""

from __future__ import annotations

import numpy as np
import trimesh
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

import pytest

from experiments.warsaw.exceptions import TooManyToPaintError
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.painting import (
    UNNAMED,
    UNNAMED_GREY,
    Coloring,
    PaintColor,
    paint,
    spread_colours,
)

# %% a world with two of a kind and one of nothing


def world_with_two_drawers_and_an_unnamed_body() -> World:
    """
    Build two drawers, a handle, and one body nothing names.
    """
    world = World.create_with_root_body("root")
    bodies = {
        name: Body(
            name=PrefixedName(name),
            visual=ShapeCollection([Box(scale=Scale(0.5, 0.5, 0.2))]),
        )
        for name in ("drawer_one", "drawer_two", "handle_one", "nobody_named_me")
    }
    with world.modify_world():
        for body in bodies.values():
            world.add_connection(FixedConnection(parent=world.root, child=body))
        handle = Handle(name=PrefixedName("a_handle"), root=bodies["handle_one"])
        for name in ("drawer_one", "drawer_two"):
            world.add_semantic_annotation_recursively(
                Drawer(name=PrefixedName(f"a_{name}"), root=bodies[name])
            )
        world.add_semantic_annotation_recursively(handle)
    return world


def colour_of(world: World, body_name: str) -> Color:
    """
    :param world: The painted world.
    :param body_name: The body to look at.
    :return: What its first shape is painted.
    """
    [body] = [one for one in world.bodies if str(one.name) == body_name]
    return list(body.visual)[0].color


# %% what painting says


def test_bodies_of_one_class_are_painted_alike():
    """
    The point of painting by class: what a body is can be read off its colour.
    """
    world = world_with_two_drawers_and_an_unnamed_body()

    paint(world, Coloring.BY_CLASS)

    assert colour_of(world, "drawer_one") == colour_of(world, "drawer_two")
    assert colour_of(world, "handle_one") != colour_of(world, "drawer_one")


def test_bodies_of_one_class_are_painted_apart_when_painted_by_body():
    """
    Two drawers standing side by side are one shape unless they differ in colour.
    """
    world = world_with_two_drawers_and_an_unnamed_body()

    paint(world, Coloring.BY_BODY)

    assert colour_of(world, "drawer_one") != colour_of(world, "drawer_two")


def test_a_body_nothing_named_is_painted_as_one_nothing_named():
    """
    Giving it a colour of its own would show it as just another class, when what it is
    is a body the world never named.
    """
    world = world_with_two_drawers_and_an_unnamed_body()

    painted = paint(world, Coloring.BY_CLASS)

    assert colour_of(world, "nobody_named_me") == UNNAMED_GREY
    assert painted[UNNAMED] == UNNAMED_GREY


def test_a_world_nobody_asked_to_paint_is_left_as_it_was():
    """
    A world already carrying the colours someone wants keeps them.
    """
    world = world_with_two_drawers_and_an_unnamed_body()
    before = colour_of(world, "drawer_one")

    assert paint(world, Coloring.UNCHANGED) == {}
    assert colour_of(world, "drawer_one") == before


def test_what_was_painted_is_reported_so_it_can_be_said_afterwards():
    """
    The colours mean nothing to a viewer without a key, and the key is only knowable
    here.
    """
    world = world_with_two_drawers_and_an_unnamed_body()

    painted = paint(world, Coloring.BY_CLASS)

    assert set(painted) == {"Drawer", "Handle", UNNAMED}


def test_a_scanned_mesh_built_before_painting_is_painted_too(tmp_path):
    """
    A scanned body keeps the mesh it built the first time it was looked at, colours of
    the scan included, so painting that leaves the mesh alone leaves the scan showing.
    """
    scan = trimesh.creation.box(extents=(0.5, 0.5, 0.2))
    scan.visual.vertex_colors = np.random.default_rng(0).integers(
        0, 255, size=(len(scan.vertices), 4), dtype=np.uint8
    )
    world = World.create_with_root_body("root")
    shape = Mesh.from_trimesh(mesh=scan, directory=tmp_path)
    body = Body(name=PrefixedName("scanned_drawer"), visual=ShapeCollection([shape]))
    with world.modify_world():
        world.add_connection(FixedConnection(parent=world.root, child=body))
        world.add_semantic_annotation_recursively(
            Drawer(name=PrefixedName("a_scanned_drawer"), root=body)
        )
    # A run's world has built its meshes long before it is painted.
    shape.mesh.vertices

    painted = paint(world, Coloring.BY_CLASS)

    np.testing.assert_array_equal(
        np.unique(shape.mesh.visual.face_colors, axis=0),
        [trimesh.visual.color.to_rgba(painted["Drawer"].to_rgba())],
    )


# %% colours that can be told apart


def test_every_colour_of_a_palette_differs_from_every_other():
    """
    A palette repeating itself would show two classes as one.
    """
    colours = spread_colours(24)

    assert len({(one.R, one.G, one.B) for one in colours}) == 24


def test_a_palette_of_nothing_is_empty_rather_than_an_error():
    """
    A world with no bodies is a world with nothing to tell apart.
    """
    assert spread_colours(0) == []


# %% colours a model is told the names of


def test_no_colour_a_model_is_told_is_named_like_a_class_of_the_ontology():
    """
    A body said to be painted "tomato" was taken for a tomato.
    """
    load_annotation_modules()
    class_names = {name.lower() for name in annotation_classes(SemanticAnnotation)}

    assert [color.word for color in PaintColor if color.word in class_names] == []


def test_the_colours_objects_are_painted_apart_in_all_differ():
    colors = PaintColor.distinct(len(PaintColor))

    assert len({color.color.to_rgb() for color in colors}) == len(colors)


def test_more_objects_than_colours_are_not_painted_apart():
    with pytest.raises(TooManyToPaintError):
        PaintColor.distinct(len(PaintColor) + 1)


def test_a_group_of_bodies_named_at_once_fits_in_the_colours():
    """
    Classifying names that many bodies in one picture, each in its own colour.
    """
    assert PipelineSettings().group_size <= len(PaintColor)
