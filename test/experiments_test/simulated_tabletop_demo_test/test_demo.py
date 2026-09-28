"""
The simulated tabletop world shares the robot's moving camera.
"""

import logging
from importlib.resources import files
from pathlib import Path
from time import sleep

import numpy as np
import pytest
from rclpy.action import get_action_names_and_types

from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.perception import Detection, ROBOKUDO_QUERY_ACTION_NAME
from coraplex.ros import create_action_client
from coraplex.testing import StandaloneProcess
from experiments.simulated_tabletop_demo.demo import (
    CUP_NAME,
    MILK_NAME,
    TABLE_NAME,
    TABLE_HEIGHT,
    SimulatedTabletopDemonstration,
    main,
)
import robokudo.world as rk_world
from robokudo.annotators.semdt_segmented_objects import SegmentedObjectAnnotator
from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    RuntimeRobotWorldSource,
    SemDTRayTracerCameraConfig,
    SemDTRGBMode,
)
from robokudo.io.semdt_camera_context import RayTracingContext
from robokudo.io.semdt_raytracer_camera_interface import SemDTRayTracerCameraInterface
from robokudo.io.semdt_raytracer_renderer import SemDTRayTracerRenderer
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

# %% Scene and camera


def test_demo_has_two_distinct_objects_and_a_robot_mounted_camera():
    """
    The robot's default camera follows its drive beside two typed objects.
    """
    demonstration = SimulatedTabletopDemonstration(used_robot=Tiago)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)

    assert world.get_body_by_name(TABLE_NAME) is not None
    assert world.get_semantic_annotations_by_type(Cup)[0].root.name.name == CUP_NAME
    assert world.get_semantic_annotations_by_type(Milk)[0].root.name.name == MILK_NAME

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    camera = robot.get_default_camera()
    initial_pose = camera.root.global_transform.to_np().copy()
    forward = initial_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert initial_pose[2, 3] > TABLE_HEIGHT + 0.4
    assert forward[2] < -0.2
    table_x_at_height = (
        initial_pose[0, 3]
        + ((TABLE_HEIGHT - initial_pose[2, 3]) / forward[2]) * forward[0]
    )
    assert 0.85 < table_x_at_height < 1.75
    robot.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.1, reference_frame=robot.root.parent_connection.parent
    )

    assert np.isclose(
        camera.root.global_transform.to_np()[0, 3] - initial_pose[0, 3], 0.1
    )


def test_robot_camera_renders_both_tabletop_objects():
    """
    A frame from the robot camera sees both objects beyond robot geometry.
    """
    demonstration = SimulatedTabletopDemonstration(used_robot=Tiago)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    camera = robot.get_default_camera()
    camera.camera_model = FieldOfViewCameraModel(
        view=camera.field_of_view,
        image_resolution=CameraResolution(width=128, height=128),
    )
    renderer = SemDTRayTracerRenderer(
        rgb_mode=SemDTRGBMode.SEMANTIC,
        logger=logging.getLogger(__name__),
    )

    frame = renderer.render(RayTracingContext(world=world, camera=camera))
    visible = set(np.unique(frame.segmentation))

    assert world.get_body_by_name(CUP_NAME).index in visible
    assert world.get_body_by_name(MILK_NAME).index in visible


def test_rendered_segments_produce_separate_object_poses(monkeypatch):
    """
    Segment depth places the requested object near its world pose.
    """
    demonstration = SimulatedTabletopDemonstration(used_robot=Tiago)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    camera = world.get_semantic_annotations_by_type(Tiago)[0].get_default_camera()
    camera.camera_model = FieldOfViewCameraModel(
        view=camera.field_of_view,
        image_resolution=CameraResolution(width=128, height=128),
    )
    monkeypatch.setattr(rk_world, "world_instance", lambda: world)
    monkeypatch.setattr(
        rk_world, "init_world_entity_tracker_from_world", lambda world: None
    )
    cas = CAS()
    SemDTRayTracerCameraInterface(
        SemDTRayTracerCameraConfig(source=RuntimeRobotWorldSource())
    ).set_data(cas)
    annotator = SegmentedObjectAnnotator()
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)

    annotator.update()

    hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
    assert len(hypotheses) >= 2
    assert all(
        any(isinstance(a, PoseAnnotation) for a in h.annotations) for h in hypotheses
    )
    red_hypothesis = max(
        hypotheses,
        key=lambda hypothesis: float(
            np.mean(
                cas.color_image[
                    hypothesis.roi.roi.pos.y : hypothesis.roi.roi.pos.y
                    + hypothesis.roi.roi.height,
                    hypothesis.roi.roi.pos.x : hypothesis.roi.roi.pos.x
                    + hypothesis.roi.roi.width,
                ][hypothesis.roi.mask > 0][:, 2]
            )
        ),
    )
    pose = next(a for a in red_hypothesis.annotations if isinstance(a, PoseAnnotation))
    observed_position = cas.camera_observation.world_T_camera.to_np() @ np.array(
        [*pose.translation, 1.0]
    )
    expected_position = world.get_body_by_name(MILK_NAME).global_transform.to_np()[:, 3]
    assert np.linalg.norm(observed_position[:3] - expected_position[:3]) < 0.2


# %% Plan to RoboKudo process


def test_module_main_prints_query_result(monkeypatch, capsys):
    """
    The demo entry point sends the plan's RoboKudo result to stdout.
    """

    def report_result(self):
        logging.getLogger("coraplex.perception").info(
            "RoboKudo object: unclassified color=red"
        )

    monkeypatch.setattr(SimulatedTabletopDemonstration, "run", report_result)

    main()

    assert "RoboKudo object: unclassified color=red" in capsys.readouterr().out


WORLD_UPDATE_DELIVERY_SECONDS = 0.3
"""
Time for the ROS subscriber to apply a published state update.
"""


def test_plan_applies_updated_detection_from_separate_ray_tracing_process(monkeypatch):
    """
    A plan receives a later object state through RoboKudo's synced renderer.
    """
    pytest.importorskip("robokudo_msgs")
    demonstration = SimulatedTabletopDemonstration(used_robot=Tiago)
    applied = []
    original_apply = Detection.apply_to

    def record_detection(self, world, trust_orientation=True):
        """
        Record and apply the actual detection returned to the plan.
        """
        applied.append(self)
        return original_apply(self, world, trust_orientation)

    monkeypatch.setattr(Detection, "apply_to", record_detection)
    with RobotDemonstrationRosSession.start("tabletop_test_probe") as probe:

        def is_serving_queries():
            """
            Report when the child process advertises the query action.
            """
            return any(
                name.lstrip("/") == ROBOKUDO_QUERY_ACTION_NAME
                for name, _ in get_action_names_and_types(probe.node)
            )

        launcher = files("robokudo").joinpath("scripts/main.py")
        with StandaloneProcess(
            launcher_path=Path(str(launcher)),
            is_ready=is_serving_queries,
            arguments=["_ae", "semdt_raytracer_tabletop_query_demo", "_headless"],
        ):
            world = demonstration.acquire_world()
            try:
                demonstration.populate_scene(world)
                context = demonstration.build_context(world)
                with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
                    demonstration.build_plan(context).perform()

                    milk = world.get_body_by_name(MILK_NAME)
                    milk.parent_connection.origin = (
                        HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=-0.05,
                            y=-0.07,
                            z=0.13,
                            reference_frame=world.get_body_by_name(TABLE_NAME),
                        )
                    )
                    sleep(WORLD_UPDATE_DELIVERY_SECONDS)
                    demonstration.build_plan(context).perform()
            finally:
                demonstration.tear_down()
                demonstration.stop_visualization()

    assert len(applied) == 2
    assert all(detection.semantic_annotation is Milk for detection in applied)
    first_position = (
        world.transform(applied[0].pose, world.root).to_homogeneous_matrix().to_np()
    )
    second_position = (
        world.transform(applied[1].pose, world.root).to_homogeneous_matrix().to_np()
    )
    assert second_position[1, 3] > first_position[1, 3] + 0.08


def test_tabletop_query_returns_both_untyped_colored_objects_and_plan_uses_red(
    monkeypatch, caplog
):
    """
    RGB-D clustering reports colors; the demo selects red to update milk.
    """
    pytest.importorskip("robokudo_msgs")
    from robokudo_msgs.action import Query

    demonstration = SimulatedTabletopDemonstration(used_robot=Tiago)
    applied = []
    original_apply = Detection.apply_to

    def record_detection(self, world, trust_orientation=True):
        """
        Observe the plan's application of a returned detection.
        """
        applied.append(self)
        return original_apply(self, world, trust_orientation)

    monkeypatch.setattr(Detection, "apply_to", record_detection)
    with RobotDemonstrationRosSession.start("tabletop_cluster_probe") as probe:

        def is_serving_queries():
            """
            Wait until the alternate AE advertises its query action.
            """
            return any(
                name.lstrip("/") == ROBOKUDO_QUERY_ACTION_NAME
                for name, _ in get_action_names_and_types(probe.node)
            )

        launcher = files("robokudo").joinpath("scripts/main.py")
        with StandaloneProcess(
            launcher_path=Path(str(launcher)),
            is_ready=is_serving_queries,
            arguments=["_ae", "semdt_raytracer_tabletop_query_demo", "_headless"],
        ):
            world = demonstration.acquire_world()
            try:
                demonstration.populate_scene(world)
                client = create_action_client(
                    ROBOKUDO_QUERY_ACTION_NAME, Query, probe.node
                )
                assert client.wait_for_server(timeout_sec=10)
                result = client.send_goal(Query.Goal()).result
                assert len(result.res) == 2
                assert all(not designator.type for designator in result.res)
                assert all(designator.pose for designator in result.res)
                assert {
                    color for designator in result.res for color in designator.color
                } >= {"red", "blue"}

                expected_milk_position = world.get_body_by_name(
                    MILK_NAME
                ).global_transform.to_np()[:3, 3]
                context = demonstration.build_context(world)
                with caplog.at_level(logging.INFO, logger="coraplex.perception"):
                    with ExecutionEnvironment(execution_type=ExecutionType.SIMULATED):
                        demonstration.build_plan(context).perform()
            finally:
                demonstration.tear_down()
                demonstration.stop_visualization()

    assert len(applied) == 1
    assert applied[0].semantic_annotation is Milk
    assert "unclassified color=red" in caplog.text
    assert "unclassified color=blue" in caplog.text
    applied_position = (
        world.transform(applied[0].pose, world.root)
        .to_homogeneous_matrix()
        .to_np()[:3, 3]
    )
    assert np.linalg.norm(applied_position - expected_milk_position) < 0.2
