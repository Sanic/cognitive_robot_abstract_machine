import os

import pytest

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World


@pytest.fixture(scope="function")
def jeroen_cup_world_fixture() -> World:
    """
    A world holding the single body a cup mesh file describes.
    """
    stl_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "stl",
    )
    world = STLParser(os.path.join(stl_dir, "jeroen_cup.stl")).create_world()
    world.root.name = PrefixedName("root")
    return world
