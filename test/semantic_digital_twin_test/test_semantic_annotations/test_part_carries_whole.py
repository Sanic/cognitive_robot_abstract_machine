"""
Parts that carry their whole: what a thing is part of differs from what it hangs from.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.exceptions import WholeWithoutParentError
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    BaseStation,
    Cable,
    Telephone,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% annotations on bodies of their own


def spawned(world, annotation_type):
    """
    :param world: The world to spawn into.
    :param annotation_type: The class of the annotation.
    :return: One annotation of it, fixed to the world's root.
    """
    with world.modify_world():
        return annotation_type.create_with_new_body_in_world(
            name=annotation_type.__name__.lower(),
            world=world,
            scale=Scale(0.1, 0.2, 0.3),
        )


def added(whole, part):
    """
    Add a part to a whole.
    """
    with whole._world.modify_world():
        whole.add(part)


@pytest.fixture
def world():
    """
    :return: A world of nothing but its root.
    """
    return World.create_with_root_body("root")


# %% mounting


def test_a_part_that_carries_its_whole_takes_the_wholes_place(world):
    telephone, cable = spawned(world, Telephone), spawned(world, Cable)

    added(telephone, cable)

    assert (
        cable.root.parent_kinematic_structure_entity,
        telephone.root.parent_kinematic_structure_entity,
        telephone.cables,
    ) == (world.root, cable.root, [cable])


def test_a_part_that_does_not_carry_its_whole_hangs_from_it(world):
    base, cable = spawned(world, BaseStation), spawned(world, Cable)

    added(base, cable)

    assert cable.root.parent_kinematic_structure_entity == base.root


def test_a_part_cannot_carry_a_whole_that_hangs_from_nothing(world):
    telephone = Telephone(root=world.root)
    with world.modify_world():
        world.add_semantic_annotation(telephone)
    cable = spawned(world, Cable)

    with pytest.raises(WholeWithoutParentError):
        added(telephone, cable)
