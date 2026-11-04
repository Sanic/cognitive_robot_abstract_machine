"""
Parts that may move: the kind of joint their class moves on, and where the joint sits.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.exceptions import MovableJointKindError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Blinds,
    Switch,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
    RevoluteConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% a part on a fixed connection


def spawned(annotation_type):
    """
    :param annotation_type: A class of parts that may move.
    :return: One part of it, fixed to the root of a world of its own.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        return annotation_type.create_with_new_body_in_world(
            name=annotation_type.__name__.lower(),
            world=world,
            scale=Scale(0.1, 0.2, 0.3),
        )


def mounted(part, connection_specification):
    """
    :return: The connection the part is mounted on.
    """
    with part._world.modify_world():
        return part.mount_on_movable_joint(connection_specification)


# %% the kind of joint


def test_a_part_fixed_to_its_parent_has_no_movable_joint():
    assert spawned(Blinds).movable_joint is None


def test_a_part_is_mounted_on_the_kind_of_joint_its_class_moves_on():
    blinds = spawned(Blinds)

    connection = mounted(blinds, PrismaticConnectionSpecification(axis=Vector3.Z()))

    assert (blinds.movable_joint, type(connection)) == (
        connection,
        blinds.movable_joint_type,
    )


def test_a_part_is_not_mounted_on_another_kind_of_joint():
    with pytest.raises(MovableJointKindError):
        mounted(spawned(Blinds), RevoluteConnectionSpecification(axis=Vector3.Z()))


@pytest.mark.parametrize(
    "connection_specification",
    [
        RevoluteConnectionSpecification(axis=Vector3.Z()),
        PrismaticConnectionSpecification(axis=Vector3.Z()),
    ],
    ids=["revolute", "prismatic"],
)
def test_a_part_moving_on_several_kinds_is_mounted_on_any_of_them(
    connection_specification,
):
    switch = spawned(Switch)

    connection = mounted(switch, connection_specification)

    assert isinstance(connection, connection_specification.connection_type)


# %% where the joint sits


def test_the_joint_sits_at_the_parts_origin():
    switch = spawned(Switch)

    assert np.array_equal(
        switch.calculate_self_T_movable_joint(Vector3.Z()).to_np(), np.eye(4)
    )
