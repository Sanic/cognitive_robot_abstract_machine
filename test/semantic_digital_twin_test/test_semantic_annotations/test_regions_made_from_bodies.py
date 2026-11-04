"""
Building the region a region-rooted annotation takes out of a body that was measured.

A reconstruction gives every object a body, but some annotations are rooted on a region
instead: an aperture is a hole rather than a thing. Deriving the region from the body
bridges the two, and it belongs to every region-rooted annotation rather than to the one
that happened to need it first.
"""

from __future__ import annotations

import numpy as np

from semantic_digital_twin.semantic_annotations.mixins import HasRootRegion
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Level,
    Wall,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Region

# %% a fixture of one placed body

WINDOW_SCALE = Scale(0.2, 1.0, 1.0)
"""
The size of the body every test here derives its region from.
"""

WINDOW_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 1.5, 1.0)
"""
Where that body stands, away from the origin, which is the only place the placement of a
derived region can be told apart from no placement at all.
"""


def world_with_a_placed_body() -> tuple[World, Wall]:
    """
    :return: A world, and a wall body standing away from its origin.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        body = Wall.create_with_new_body_in_world(
            name="window",
            scale=WINDOW_SCALE,
            world=world,
            world_root_T_self=WINDOW_POSE,
        )
    return world, body


# %% every region-rooted annotation can do it


def test_a_region_rooted_annotation_other_than_an_aperture_can_be_built_from_a_body():
    """
    The bridge is the mixin's, not the aperture's: a run composing its own region-rooted
    class gets it without the ontology having anticipated that class.
    """
    assert hasattr(
        HasRootRegion, Aperture.create_with_new_region_in_world_from_body.__name__
    )

    world, body = world_with_a_placed_body()
    with world.modify_world():
        level = Level.create_with_new_region_in_world_from_body(
            name="storey", world=world, body=body.root
        )
    assert isinstance(level.root, Region)


# %% the region stands where the body stood


def test_a_region_built_from_a_body_stands_where_that_body_stands():
    """
    A region put at the origin instead would cut its hole out of whatever happens to be
    there, so the body's own pose is what a derived region is placed at.
    """
    world, body = world_with_a_placed_body()
    with world.modify_world():
        aperture = Aperture.create_with_new_region_in_world_from_body(
            name="hole", world=world, body=body.root
        )
    world.update_forward_kinematics()
    assert np.allclose(
        aperture.root.global_transform.to_np(), body.root.global_transform.to_np()
    )
