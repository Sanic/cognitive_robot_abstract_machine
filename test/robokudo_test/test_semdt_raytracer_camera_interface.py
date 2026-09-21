from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
    WorldDescriptorSource,
)
from robokudo.io.semdt_raytracer_camera_interface import (
    SemDTRayTracerCameraInterface,
)

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
    assert cas.camera_info.width == cas.camera_observation.resolution.width
    assert cas.camera_info.height == cas.camera_observation.resolution.height
    context = interface.context_resolver.resolve()
    assert cas.camera_observation.camera is context.camera
    assert cas.ground_truth_world_ref is context.world
