import numpy as np
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
    WorldDescriptorSource,
)
from robokudo.io.semdt_raytracer_camera_interface import (
    SemDTRayTracerCameraInterface,
)
from semantic_digital_twin.datastructures.camera_model import FieldOfViewCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

# %% Interface integration


def test_interface_publishes_aligned_semantic_rgbd_frame():
    """
    The complete interface publishes images and matching semantic camera state.
    """
    interface = SemDTRayTracerCameraInterface(
        SemDTRayTracerCameraConfig(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    cas = CAS()

    interface.set_data(cas)

    assert cas.color_image.shape[:2] == cas.depth_image.shape
    assert cas.get(CASViews.OBJECT_IMAGE).shape == cas.depth_image.shape
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())
    context = interface.context_resolver.resolve()
    assert cas.camera_observation.camera is context.camera
    assert cas.ground_truth_world_ref is context.world


def test_interface_publishes_effective_model_for_field_of_view_camera():
    """
    An FOV camera produces a calibrated effective observation model.
    """
    interface = SemDTRayTracerCameraInterface(
        SemDTRayTracerCameraConfig(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    context = interface.context_resolver.resolve()
    native_field_of_view = context.camera.field_of_view
    resolution = CameraResolution(width=32, height=24)
    context.camera.camera_model = FieldOfViewCameraModel(
        view=native_field_of_view,
        image_resolution=resolution,
    )
    cas = CAS()

    interface.set_data(cas)

    assert cas.camera_observation.camera is context.camera
    assert cas.camera_observation.effective_camera_model.resolution is resolution
    assert np.isclose(
        cas.camera_observation.field_of_view.horizontal_angle,
        native_field_of_view.horizontal_angle,
    )
    assert np.isclose(
        cas.camera_observation.field_of_view.vertical_angle,
        native_field_of_view.vertical_angle,
    )
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())
