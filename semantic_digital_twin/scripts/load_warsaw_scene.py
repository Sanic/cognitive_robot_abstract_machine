import os
import time

import numpy as np
from PIL import Image

from semantic_digital_twin.adapters.mesh import OBJParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    TransformGeometry,
    CenterLocalGeometryAndPreserveWorldPose,
)
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.geometry import Color

dir_path = "/home/itsme/work/cram_ws/src/cognitive_robot_abstract_machine/semantic_digital_twin/resources/warsaw_data/objects/"
files = [f for f in os.listdir(dir_path) if f.endswith(".obj")]

world = World()
root = Body(name=PrefixedName("root_body"))
with world.modify_world():
    world.add_body(root)
for file in files:
    obj_world = OBJParser(os.path.join(dir_path, file)).parse()
    with world.modify_world():
        color = Color(1, 0, 0)
        obj_world.bodies[0].collision[0].color = color
        world.merge_world(obj_world)

pipeline = Pipeline(
    steps=[
        TransformGeometry(
            TransformationMatrix.from_xyz_rpy(roll=np.pi / 2, pitch=0, yaw=0)
        ),
        CenterLocalGeometryAndPreserveWorldPose(),
    ]
)
world = pipeline.apply(world)

from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
import rclpy

rclpy.init()

node = rclpy.create_node("semantic_digital_twin")

viz = VizMarkerPublisher(world=world, node=node)

time.sleep(5)
