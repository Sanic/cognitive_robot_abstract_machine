from unittest.mock import Mock

import numpy as np
import pytest

import robokudo.world as rk_world
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    RuntimeRobotWorldSource,
    SemDTRayTracerCameraConfig,
    WorldDescriptorSource,
)
from robokudo.descriptors.worlds.world_semdt_raytracer_cylinders import WorldDescriptor
from robokudo.io.semdt_camera_context import (
    RayTracingContext,
    SemDTCameraContextResolver,
)
from robokudo.exceptions import (
    CameraAnnotationMissing,
    CameraPoseOverrideUnavailable,
    RobotAnnotationAmbiguous,
    RobotAnnotationMissing,
)
from semantic_digital_twin.datastructures.camera_model import (
    FieldOfViewCameraModel,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Camera, RobotCamera
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% Semantic camera selection


def test_resolver_selects_named_descriptor_camera():
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    source = WorldDescriptorSource(camera_name=camera.name)

    selected_camera = SemDTCameraContextResolver._select_descriptor_camera(
        world, source
    )

    assert selected_camera is camera


def test_resolver_rejects_unknown_camera_name():
    world = WorldDescriptor().world
    source = WorldDescriptorSource(camera_name=PrefixedName(name="unknown_camera"))

    with pytest.raises(CameraAnnotationMissing):
        SemDTCameraContextResolver._select_descriptor_camera(world, source)


def test_resolver_rejects_world_without_camera_annotation():
    """
    A ray-traced world must declare the semantic camera it renders from.
    """
    world = World()
    with pytest.raises(CameraAnnotationMissing):
        SemDTCameraContextResolver._select_descriptor_camera(
            world, WorldDescriptorSource()
        )


# %% Runtime robot camera selection


def test_resolver_uses_runtime_robots_default_camera(monkeypatch):
    world = Mock(spec=World)
    robot = Mock(spec=AbstractRobot)
    camera = Mock(spec=RobotCamera)
    robot.get_default_camera.return_value = camera
    world.get_semantic_annotations_by_type.return_value = [robot]
    monkeypatch.setattr(rk_world, "world_instance", lambda: world)
    monkeypatch.setattr(rk_world, "init_world_entity_tracker_from_world", Mock())
    resolver = SemDTCameraContextResolver(
        SemDTRayTracerCameraConfig(source=RuntimeRobotWorldSource())
    )

    context = resolver.resolve()

    assert isinstance(context, RayTracingContext)
    assert context.world is world
    assert context.camera is camera
    robot.get_default_camera.assert_called_once_with()


def test_resolver_retries_runtime_lookup_after_world_becomes_available(monkeypatch):
    unavailable_world = Mock(spec=World)
    unavailable_world.get_semantic_annotations_by_type.return_value = []
    available_world = Mock(spec=World)
    robot = Mock(spec=AbstractRobot)
    camera = Mock(spec=RobotCamera)
    robot.get_default_camera.return_value = camera
    available_world.get_semantic_annotations_by_type.return_value = [robot]
    runtime_worlds = iter((unavailable_world, available_world))
    monkeypatch.setattr(rk_world, "world_instance", lambda: next(runtime_worlds))
    monkeypatch.setattr(rk_world, "init_world_entity_tracker_from_world", Mock())
    resolver = SemDTCameraContextResolver(
        SemDTRayTracerCameraConfig(source=RuntimeRobotWorldSource())
    )

    with pytest.raises(RobotAnnotationMissing):
        resolver.resolve()

    context = resolver.resolve()
    assert context.world is available_world
    assert context.camera is camera


def test_resolver_selects_named_robot_default_camera():
    world = Mock(spec=World)
    selected_robot = Mock(spec=AbstractRobot)
    selected_robot.name = PrefixedName(name="selected_robot")
    other_robot = Mock(spec=AbstractRobot)
    other_robot.name = PrefixedName(name="other_robot")
    camera = Mock(spec=RobotCamera)
    selected_robot.get_default_camera.return_value = camera
    world.get_semantic_annotations_by_type.return_value = [
        other_robot,
        selected_robot,
    ]
    source = RuntimeRobotWorldSource(robot_name=selected_robot.name)
    selected_camera = SemDTCameraContextResolver._select_runtime_robot_camera(
        world, source
    )

    assert selected_camera is camera
    other_robot.get_default_camera.assert_not_called()


def test_resolver_rejects_ambiguous_runtime_robot_selection():
    world = Mock(spec=World)
    first_robot = Mock(spec=AbstractRobot)
    first_robot.name = PrefixedName(name="first_robot")
    second_robot = Mock(spec=AbstractRobot)
    second_robot.name = PrefixedName(name="second_robot")
    world.get_semantic_annotations_by_type.return_value = [first_robot, second_robot]
    source = RuntimeRobotWorldSource()
    with pytest.raises(RobotAnnotationAmbiguous):
        SemDTCameraContextResolver._select_runtime_robot_camera(world, source)


# %% Camera pose


def test_resolver_preserves_camera_pose_without_override():
    """
    Loading a camera without an override keeps its descriptor pose.
    """
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    descriptor_pose = camera.root.global_transform.to_np().copy()
    SemDTCameraContextResolver._apply_camera_pose(world, camera, camera_pose=None)

    np.testing.assert_allclose(camera.root.global_transform.to_np(), descriptor_pose)


def test_resolver_applies_camera_pose_override_once():
    """
    The first camera load applies the override without pinning later motion.
    """
    camera_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.8,
        y=0.2,
        z=1.4,
        roll=-2.0,
        yaw=-0.4,
    )
    descriptor = WorldDescriptor()
    world = descriptor.world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    config = SemDTRayTracerCameraConfig(
        source=WorldDescriptorSource(camera_pose=camera_pose)
    )
    module_loader = Mock()
    module_loader.load_world_descriptor.return_value = descriptor
    resolver = SemDTCameraContextResolver(config, module_loader=module_loader)

    context = resolver.resolve()

    assert context.camera is camera
    np.testing.assert_allclose(
        camera.root.parent_connection.origin.to_np(), camera_pose.to_np()
    )

    later_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.7,
        y=0.3,
        z=1.2,
        reference_frame=camera.root.parent_connection.parent,
        child_frame=camera.root,
    )
    with world.modify_world():
        camera.root.parent_connection.origin = later_pose

    assert resolver.resolve().camera is camera
    np.testing.assert_allclose(
        camera.root.parent_connection.origin.to_np(), later_pose.to_np()
    )


def test_resolver_rejects_pose_override_for_world_root_camera():
    """
    A camera without a mutable parent connection cannot be repositioned.
    """
    world = World()
    camera_body = Body(name=PrefixedName(name="root_camera"))
    camera = Camera(
        name=PrefixedName(name="root_camera"),
        root=camera_body,
        forward_facing_axis=Vector3.Z(),
        camera_model=FieldOfViewCameraModel(view=FieldOfView()),
    )
    with world.modify_world():
        world.add_body(camera_body)
        world.add_semantic_annotation(camera)
    with pytest.raises(CameraPoseOverrideUnavailable):
        SemDTCameraContextResolver._apply_camera_pose(
            world,
            camera,
            camera_pose=HomogeneousTransformationMatrix(),
        )
