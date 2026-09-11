import pytest

from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.descriptors.worlds.world_semdt_raytracer_cylinders import WorldDescriptor
from robokudo.io.semdt_raytracer_camera_interface import (
    SemDTRayTracerCameraInterface,
)
from robokudo.exceptions import CameraAnnotationMissing
from semantic_digital_twin.robots.robot_parts import Camera


def test_interface_uses_supplied_world_and_selects_its_camera():
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    config = SemDTRayTracerCameraConfig(
        world=world,
        camera_name=camera.name.name,
    )
    interface = SemDTRayTracerCameraInterface(config)

    loaded_world = interface._load_runtime_world()
    selected_camera = interface._select_or_create_camera(loaded_world)

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
        interface._select_or_create_camera(world)
