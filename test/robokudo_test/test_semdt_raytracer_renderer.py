import logging

import numpy as np
import pytest

from robokudo.descriptors.camera_configs.config_semdt_raytracer import SemDTRGBMode
from robokudo.descriptors.worlds.world_semdt_raytracer_cylinders import WorldDescriptor
from robokudo.exceptions import CameraResolutionUnavailable
from robokudo.io.semdt_camera_context import RayTracingContext
from robokudo.io.semdt_raytracer_renderer import SemDTRayTracerRenderer
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.robots.robot_parts import Camera

# %% Renderer contracts


def test_renderer_requires_complete_pinhole_camera_model():
    """
    Rendering rejects a camera whose image resolution is unknown.
    """
    world = WorldDescriptor().world
    [camera] = world.get_semantic_annotations_by_type(Camera)
    camera.camera_model = FieldOfViewCameraModel(view=camera.field_of_view)
    renderer = SemDTRayTracerRenderer(
        rgb_mode=SemDTRGBMode.SEMANTIC,
        logger=logging.getLogger("semdt-renderer-test"),
    )

    with pytest.raises(CameraResolutionUnavailable):
        renderer.render(RayTracingContext(world=world, camera=camera))


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
