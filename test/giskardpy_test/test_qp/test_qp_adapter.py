import numpy as np
import pytest
from giskardpy.qp.adapters.qp_adapter import DofLimits
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture()
def prismatic_bot(cylinder_bot_world):
    world = World()
    with world.modify_world():
        map = Body(name=PrefixedName("map"))
        robot = Body(name=PrefixedName("robot"))
        dof = DegreeOfFreedom(
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=-1, velocity=-1, acceleration=None, jerk=None
                ),
                upper=DerivativeMap(
                    position=1, velocity=1, acceleration=None, jerk=None
                ),
            ),
            has_hardware_interface=True,
        )
        world.add_degree_of_freedom(dof)
        map_C_robot = PrismaticConnection(
            parent=map, child=robot, dof_id=dof.id, axis=Vector3.Z()
        )
        world.add_connection(map_C_robot)
    MinimalRobot.from_world(world)
    return world


def test_DofLimits(prismatic_bot):
    target_frequency = 20
    prediction_horizon = 10
    expected_jerk_limit = 1 / target_frequency
    limits = DofLimits.create(
        prismatic_bot.active_degrees_of_freedom,
        config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        limits.lower_bounds.evaluate(),
        np.array([-1.0] * 8 + [-expected_jerk_limit] * 10),
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.upper_bounds.evaluate(),
        np.array([1.0] * 8 + [expected_jerk_limit] * 10),
        rtol=1.0e-4,
    )
