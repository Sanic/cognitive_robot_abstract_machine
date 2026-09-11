import numpy as np

from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2KinectV1
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobotPart,
    Camera,
    RobotCamera,
    Sensor,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def test_camera_can_be_added_without_a_robot():
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

    with world.modify_world():
        world.add_semantic_annotation(camera)

    assert world.get_semantic_annotations_by_type(Camera) == [camera]
    assert not isinstance(camera, AbstractRobotPart)
    assert camera.resolution == model.resolution
    assert camera.field_of_view == model.field_of_view


def test_robot_camera_preserves_robot_part_and_sensor_types():
    assert issubclass(PR2KinectV1, RobotCamera)
    assert issubclass(PR2KinectV1, Camera)
    assert issubclass(PR2KinectV1, Sensor)
    assert issubclass(PR2KinectV1, AbstractRobotPart)
