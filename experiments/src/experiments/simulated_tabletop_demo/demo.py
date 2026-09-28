"""
Ask RoboKudo for colored tabletop objects in a served simulated world.

Run ``robokudo.scripts.main _ae semdt_raytracer_tabletop_query_demo _headless`` in one
process, then run this module in another. The plan serves its world over ROS, and
RoboKudo fetches it before collecting a frame for the query.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

from typing_extensions import ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import VisualizationBackend
from coraplex.demonstrations import RobotDemonstration
from coraplex.exceptions import NothingDetected, UnidentifiedDetections
from coraplex.perception import Detection, RoboKudoPerception
from coraplex.plans.factories import code
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.robots.tiago import Tiago, TiagoJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale

# %% Scene geometry

TABLE_NAME = "demo_table"
"""
The support surface shared by both objects.
"""

CUP_NAME = "demo_cup"
"""
Name of the blue box annotated as a cup in the served world.
"""

MILK_NAME = "demo_milk"
"""
Name of the red box annotated as milk in the served world.
"""

TABLE_HEIGHT = 0.76
"""
Height of the upper face of the table in the world frame.
"""

# %% Demonstration


@dataclass
class SimulatedTabletopDemonstration(RobotDemonstration):
    """
    Serve a Tiago scene and query the robot-mounted ray-traced camera.
    """

    ros_node_name: ClassVar[str] = "simulated_tabletop_demo"
    """
    ROS name for the world owner and plan.
    """

    default_visualization_backend: VisualizationBackend = VisualizationBackend.NONE
    """
    Run the small scene without a separate visualization process.
    """

    world_fetcher: FetchWorldServer | None = field(init=False, default=None)
    """
    Serves the current scene to RoboKudo.
    """

    world_synchronizer: WorldSynchronizer | None = field(init=False, default=None)
    """
    Publishes later model and state changes from the owner world.
    """

    def build_simulated_world(self) -> World:
        """
        Create the robot with its camera raised and aimed at the tabletop.
        """
        world = WorldSpecification(
            world_parser=None,
            robots=[RobotSpecification(semantic_annotation_type=self.used_robot)],
        ).to_domain_object()
        world.get_connection_by_name(TiagoJoint.TORSO_LIFT).position = 0.2
        world.get_connection_by_name(TiagoJoint.HEAD_2).position = -0.35
        return world

    def acquire_world(self) -> World:
        """
        Start serving and publishing the simulated world.
        """
        world = super().acquire_world()
        self.world_synchronizer = WorldSynchronizer(_world=world, node=self.ros_node)
        self.world_fetcher = FetchWorldServer(node=self.ros_node, world=world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        """
        Return whether the requested object already exists.
        """
        return world.is_kinematic_structure_entity_in_world_by_name(MILK_NAME)

    def populate_scene(self, world: World) -> None:
        """
        Place a table and two differently colored boxes in camera view.
        """
        table = BodySpecification.box(
            TABLE_NAME,
            scale=Scale(0.9, 0.8, 0.08),
            color=Color(0.55, 0.5, 0.4, 1.0),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.3, z=TABLE_HEIGHT - 0.04
            ),
        ).spawn(world)
        Cup.get_annotation_specification(
            CUP_NAME,
            BodySpecification.box(
                CUP_NAME,
                scale=Scale(0.1, 0.1, 0.14),
                color=Color(0.1, 0.2, 0.9, 1.0),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.05, y=0.22, z=0.11
                ),
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=table)
        Milk.get_annotation_specification(
            MILK_NAME,
            BodySpecification.box(
                MILK_NAME,
                scale=Scale(0.1, 0.1, 0.18),
                color=Color(0.9, 0.1, 0.1, 1.0),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.05, y=-0.22, z=0.13
                ),
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=table)

    def build_context(self, world: World) -> Context:
        """
        Bind the plan to its robot, world, and ROS node.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Query all tabletop objects and update the red object's world pose.
        """
        return code(lambda: self.query_and_update_world(context.world), context=context)

    def query_and_update_world(self, world: World) -> None:
        """
        Apply the pose of the uniquely red result to the demo's milk object.
        """
        objects = RoboKudoPerception(ros_node=self.ros_node).query_objects()
        red_objects = [
            detected_object
            for detected_object in objects
            if "red" in detected_object.color and detected_object.pose
        ]
        if not red_objects:
            raise NothingDetected(Milk)
        if len(red_objects) > 1:
            raise UnidentifiedDetections(Milk, len(red_objects))

        pose_stamped = red_objects[0].pose[0]
        pose = pose_stamped.pose
        Detection(
            semantic_annotation=Milk,
            pose=Pose.from_xyz_quaternion(
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                reference_frame=world.get_kinematic_structure_entity_by_name(
                    pose_stamped.header.frame_id
                ),
            ),
        ).apply_to(world, trust_orientation=False)

    def tear_down(self) -> None:
        """
        Release the world service and subscription with the ROS session.
        """
        if self.world_fetcher is not None:
            self.world_fetcher.close()
            self.world_fetcher = None
        if self.world_synchronizer is not None:
            self.world_synchronizer.close()
            self.world_synchronizer = None
        super().tear_down()


def main() -> None:
    """
    Run the plan and print RoboKudo's returned objects and poses.
    """
    result_logger = logging.getLogger("coraplex.perception")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = result_logger.level
    previous_propagation = result_logger.propagate
    result_logger.addHandler(handler)
    result_logger.setLevel(logging.INFO)
    result_logger.propagate = False
    try:
        SimulatedTabletopDemonstration(used_robot=Tiago).run()
    finally:
        result_logger.removeHandler(handler)
        result_logger.setLevel(previous_level)
        result_logger.propagate = previous_propagation


if __name__ == "__main__":
    main()
