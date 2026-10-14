import numpy as np
import pytest

from robokudo.cas import CAS
from robokudo.exceptions import CameraObservationMissing, CameraPoseMissing
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def test_cas_exposes_camera_observation_without_copying_camera():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(
            horizontal_angle=np.radians(90.0),
            vertical_angle=np.radians(60.0),
        ),
    )
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=root,
        forward_facing_axis=Vector3.X(),
        camera_model=model,
    )
    world_T_camera = HomogeneousTransformationMatrix(
        reference_frame=root,
        child_frame=root,
    )
    observation = CameraObservation(
        camera=camera,
        effective_camera_model=model,
        world_T_camera=world_T_camera,
        timestamp_nanoseconds=123,
    )
    cas = CAS()

    cas.camera_observation = observation

    assert cas.camera_observation is observation
    assert cas.require_camera_observation() is observation
    assert cas.camera_observation.camera is camera
    assert cas.camera_observation.resolution == model.resolution
    assert cas.camera_observation.field_of_view == model.field_of_view
    assert cas.camera_observation.world_T_camera_or_raise() is world_T_camera
    assert np.array_equal(
        cas.camera_observation.camera_T_world, world_T_camera.inverse()
    )


def test_cas_requires_camera_observation():
    with pytest.raises(CameraObservationMissing):
        CAS().require_camera_observation()


def test_camera_observation_requires_pose_for_spatial_access():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(),
    )
    observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=root,
            forward_facing_axis=Vector3.X(),
            camera_model=model,
        ),
        effective_camera_model=model,
        world_T_camera=None,
        timestamp_nanoseconds=123,
    )

    with pytest.raises(CameraPoseMissing):
        observation.world_T_camera_or_raise()
    with pytest.raises(CameraPoseMissing):
        observation.camera_T_world


def test_camera_observation_returns_copy_with_pose():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(),
    )
    observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=root,
            forward_facing_axis=Vector3.X(),
            camera_model=model,
        ),
        effective_camera_model=model,
        world_T_camera=None,
        timestamp_nanoseconds=123,
    )
    world_T_camera = HomogeneousTransformationMatrix(
        reference_frame=root,
        child_frame=root,
    )

    updated_observation = observation.with_world_T_camera(world_T_camera)

    assert observation.world_T_camera is None
    assert updated_observation.world_T_camera is world_T_camera
    assert updated_observation.camera is observation.camera
    assert updated_observation.effective_camera_model is model
    assert (
        updated_observation.timestamp_nanoseconds == observation.timestamp_nanoseconds
    )
