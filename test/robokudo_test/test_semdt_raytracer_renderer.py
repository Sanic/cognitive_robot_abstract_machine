import logging

import numpy as np

from robokudo.descriptors.camera_configs.config_semdt_raytracer import SemDTRGBMode
from robokudo.descriptors.worlds.world_semdt_raytracer_cylinders import WorldDescriptor
from robokudo.io.semdt_camera_context import RayTracingContext
from robokudo.io.semdt_raytracer_renderer import SemDTRayTracerRenderer
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.robots.robot_parts import Camera

# %% Renderer contracts


def test_renderer_derives_effective_pinhole_model_from_field_of_view():
    """
    Rendering derives centered intrinsics from an FOV model and its resolution.
    """
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    resolution = CameraResolution(width=32, height=24)
    camera.camera_model = FieldOfViewCameraModel(
        view=camera.field_of_view,
        image_resolution=resolution,
    )
    renderer = SemDTRayTracerRenderer(
        rgb_mode=SemDTRGBMode.SEMANTIC,
        logger=logging.getLogger("semdt-renderer-test"),
    )

    frame = renderer.render(RayTracingContext(world=world, camera=camera))

    assert frame.camera_model.resolution is resolution
    assert np.isclose(
        frame.camera_model.field_of_view.horizontal_angle,
        camera.field_of_view.horizontal_angle,
    )
    assert np.isclose(
        frame.camera_model.field_of_view.vertical_angle,
        camera.field_of_view.vertical_angle,
    )
    assert frame.color_bgr.shape[:2] == frame.depth_mm.shape
    assert frame.depth_mm.shape == frame.segmentation.shape


def test_renderer_converts_projective_depth_to_millimeters():
    """
    Misses become zero and valid meter depths become uint16 millimeters.
    """
    depth_m = np.array([[-1.0, 0.0, 1.2346, 100.0]], dtype=np.float32)

    depth_mm = SemDTRayTracerRenderer._depth_m_to_mm(depth_m)

    np.testing.assert_array_equal(
        depth_mm,
        np.array([[0, 0, 1235, 65535]], dtype=np.uint16),
    )
