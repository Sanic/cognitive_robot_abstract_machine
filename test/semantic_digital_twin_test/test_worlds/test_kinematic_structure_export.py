"""
Writing a world's kinematic structure out as JSON.

The tree is what a reader is shown of a world it cannot open, so what it says about the
structure has to be what the world holds: every entity under the one it hangs from, and
the connection each was attached by when that is asked for.
"""

from __future__ import annotations

import json

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def two_body_world() -> World:
    """
    :return: A world of one body hanging from another by a fixed connection.
    """
    world = World()
    root = Body(name=PrefixedName("root", "krrood_test"))
    child = Body(name=PrefixedName("child", "krrood_test"))
    with world.modify_world():
        world.add_body(root)
        world.add_body(child)
        world.add_connection(
            FixedConnection(root, child, HomogeneousTransformationMatrix())
        )
    return world


def test_the_tree_holds_every_entity_under_the_one_it_hangs_from(
    two_body_world, tmp_path
):
    """
    The export starts at the root and walks down, so a child has to appear under it.
    """
    written = tmp_path / "tree.json"
    two_body_world.export_kinematic_structure_tree_to_json(written)

    tree = json.loads(written.read_text())
    assert tree["name"] == two_body_world.root.name.name
    assert [child["name"] for child in tree["children"]] == ["child"]


def test_each_entity_names_the_connection_it_was_attached_by(two_body_world, tmp_path):
    """
    What kind of connection holds a body is the part of the structure that says how it
    can move.
    """
    written = tmp_path / "tree.json"
    two_body_world.export_kinematic_structure_tree_to_json(written)

    tree = json.loads(written.read_text())
    assert tree["parent_connection"] is None
    [child] = tree["children"]
    assert child["parent_connection"] == FixedConnection.__name__


def test_the_connections_are_left_out_when_they_are_not_asked_for(
    two_body_world, tmp_path
):
    """
    The structure alone is asked for when what holds a body is beside the point.
    """
    written = tmp_path / "tree.json"
    two_body_world.export_kinematic_structure_tree_to_json(
        written, include_connections=False
    )

    tree = json.loads(written.read_text())
    assert "parent_connection" not in tree
    assert all("parent_connection" not in child for child in tree["children"])


def test_an_empty_world_has_no_structure_to_export(tmp_path):
    """
    A world with no entities has no root to start at.
    """
    with pytest.raises(ValueError):
        World().export_kinematic_structure_tree_to_json(tmp_path / "tree.json")
