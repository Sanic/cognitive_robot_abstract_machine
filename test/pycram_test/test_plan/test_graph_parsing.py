from pycram.plans.executables import ModelChangeExecutable
from pycram.motion_executor import simulated_robot
from pycram.plans.executables import GiskardExecutable, Executable
from pycram.plans.attachment_nodes import ModelChangeNode
from pycram.plans.executables import (
    ConditionExecutable,
)
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans.actions.core.pick_up import ReachAction, PickUpAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from pycram.robot_plans.motions.gripper import MoveToolCenterPointMotion
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import SetGripperAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.position_descriptions import (
    VerticalSemanticDirection,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from pycram.utils import split_list_by_type


def test_parse_simple_action(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)

    plan.notify()

    executable = plan.parse()

    assert len(executable.execution_list) == 3
    assert type(executable.execution_list[0]) == ConditionExecutable
    assert type(executable.execution_list[1]) == GiskardExecutable


def test_merge_motions(immutable_model_world, rclpy_node):
    world, view, context = immutable_model_world

    VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()

    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(2, 1.5, 0.7, 0, 0, 0)
    )

    plan = execute_single(
        ReachAction(
            Pose.from_xyz_rpy(2, 1.5, 0.7, reference_frame=world.root),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
            world.get_body_by_name("milk.stl"),
        ),
        context=context,
    )

    plan.notify()

    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert len(executable.motion_mappings) == 2
    assert executable.pre_condition_node
    assert executable.post_condition_node

    with simulated_robot:
        executable.execute()


def test_parse_pick_up(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(
        PickUpAction(
            world.get_body_by_name("milk.stl"),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
        ),
        context=context,
    )

    plan.notify()

    # plan.plan.plot()

    executable = plan.parse()

    assert len(executable.execution_list) == 3
    assert type(executable.execution_list[0]) == GiskardExecutable
    assert type(executable.execution_list[1]) == ModelChangeExecutable
    assert type(executable.execution_list[2]) == GiskardExecutable


def test_parse_complex_plan(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=world.get_body_by_name("milk.stl"),
                arm=Arms.LEFT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()
    assert len(exec.execution_list) == 2


def test_parsing_two_actions_into_one_exec(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=world.get_body_by_name("milk.stl"),
                arm=Arms.LEFT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()

    assert type(exec) == GiskardExecutable
    assert len(exec.motion_mappings) == 3


def test_parse_pick_place(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            PickUpAction(
                world.get_body_by_name("milk.stl"),
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
            PlaceAction(
                world.get_body_by_name("milk.stl"),
                Pose(reference_frame=world.root),
                Arms.RIGHT,
            ),
        ],
        context=context,
    )

    plan.notify()

    # plan.plan.plot()

    executable = plan.parse()

    assert len(executable.execution_list) == 2
    assert len(executable.execution_list[0].execution_list) == 3
    assert len(executable.execution_list[1].execution_list) == 3


def test_parse_transport_plan(immutable_model_world):
    world, view, context = immutable_model_world
    plan = execute_single(
        TransportAction(
            world.get_body_by_name("milk.stl"),
            Pose(reference_frame=world.root),
            Arms.RIGHT,
        ),
        context=context,
    )

    plan.notify()
    exec = plan.parse()

    pass


def test_split_by_type():

    split_list = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        ModelChangeNode(body=None, new_parent=None),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
    ]

    splitted_list = split_list_by_type(split_list, ModelChangeNode)

    assert len(splitted_list) == 3
    assert len(splitted_list[0]) == 1
    assert len(splitted_list[1]) == 1
    assert len(splitted_list[2]) == 1
