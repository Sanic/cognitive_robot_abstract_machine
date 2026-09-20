import numpy as np
import pytest

from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.descriptors.worlds.world_semdt_raytracer_cylinders import WorldDescriptor
from robokudo.io.semdt_raytracer_camera_interface import (
    SemDTRayTracerCameraInterface,
)
from robokudo.exceptions import (
    CameraAnnotationMissing,
    CameraPoseOverrideUnavailable,
    CameraResolutionUnavailable,
)
from semantic_digital_twin.datastructures.camera_model import (
    FieldOfViewCameraModel,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% Semantic camera selection


def test_interface_uses_supplied_world_and_selects_its_camera():
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    config = SemDTRayTracerCameraConfig(
        world=world,
        camera_name=camera.name.name,
    )
    interface = SemDTRayTracerCameraInterface(config)

    loaded_world = interface._load_runtime_world()
    selected_camera = interface._select_camera(loaded_world)

    assert loaded_world is world
    assert selected_camera is camera


def test_interface_rejects_unknown_camera_name():
    world = WorldDescriptor().world
    config = SemDTRayTracerCameraConfig(
        world=world,
        camera_name="unknown_camera",
    )
    interface = SemDTRayTracerCameraInterface(config)

    with pytest.raises(CameraAnnotationMissing):
        interface._select_camera(world)


def test_interface_rejects_world_without_camera_annotation():
    """
    A ray-traced world must declare the semantic camera it renders from.
    """
    world = World()
    config = SemDTRayTracerCameraConfig(world=world)
    interface = SemDTRayTracerCameraInterface(config)

    with pytest.raises(CameraAnnotationMissing):
        interface._select_camera(world)


# %% Camera pose


def test_interface_preserves_camera_pose_without_override():
    """
    Loading a camera without an override keeps its descriptor pose.
    """
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    descriptor_pose = camera.root.global_transform.to_np().copy()
    interface = SemDTRayTracerCameraInterface(SemDTRayTracerCameraConfig(world=world))

    loaded_camera = interface._load_camera(world)

    assert loaded_camera is camera
    np.testing.assert_allclose(camera.root.global_transform.to_np(), descriptor_pose)


def test_interface_applies_camera_pose_override_once():
    """
    The first camera load applies the override without pinning later motion.
    """
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    camera_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=-0.8,
        y=0.2,
        z=1.4,
        roll=-2.0,
        yaw=-0.4,
    )
    config = SemDTRayTracerCameraConfig(world=world, camera_pose=camera_pose)
    interface = SemDTRayTracerCameraInterface(config)

    loaded_camera = interface._load_camera(world)

    assert loaded_camera is camera
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

    assert interface._load_camera(world) is camera
    np.testing.assert_allclose(
        camera.root.parent_connection.origin.to_np(), later_pose.to_np()
    )


def test_interface_rejects_pose_override_for_world_root_camera():
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
    config = SemDTRayTracerCameraConfig(
        world=world,
        camera_pose=HomogeneousTransformationMatrix(),
    )
    interface = SemDTRayTracerCameraInterface(config)

    with pytest.raises(CameraPoseOverrideUnavailable):
        interface._load_camera(world)


# %% Effective rendering model


def test_interface_requires_resolution_for_field_of_view_camera():
    """
    A ray-traced semantic camera must define a complete pinhole model.
    """
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    camera.camera_model = FieldOfViewCameraModel(view=camera.field_of_view)
    interface = SemDTRayTracerCameraInterface(SemDTRayTracerCameraConfig(world=world))

    with pytest.raises(CameraResolutionUnavailable):
        interface.set_data(CAS())
