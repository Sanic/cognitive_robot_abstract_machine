"""
Optical camera metadata and row-first image publication.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import Mock

import numpy as np
import pytest

from robokudo.annotators.static_camera_transform import StaticCameraTransformAnnotator
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.cas_view_codecs import CameraObservationCodec
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from robokudo.cas import CAS, CASViews
from robokudo.types.camera import CameraObservation
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
    WorldDescriptorSource,
)
from robokudo.io.sensor_world import SensorWorldProjector
from robokudo.io.semdt_raytracer_camera_interface import SemDTRayTracerCameraInterface
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.camera_rendering import RenderedRGBDFrame
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


# %% Rectangular rendered observation
@pytest.fixture
def optical_interface() -> SemDTRayTracerCameraInterface:
    """
    Provide a simulated camera with an explicit optical offset.
    """
    interface = SemDTRayTracerCameraInterface(
        SemDTRayTracerCameraConfig(
            source=WorldDescriptorSource(
                descriptor_name="world_semdt_raytracer_cylinders"
            )
        )
    )
    context = interface.context_resolver.resolve()
    context.camera.root_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.1, y=0.2, z=0.3, yaw=np.pi / 2
    )
    return interface


@pytest.fixture
def rectangular_frame(
    optical_interface: SemDTRayTracerCameraInterface,
) -> RenderedRGBDFrame:
    """
    Provide distinct values at every pixel of a width-first rendered frame.
    """
    camera = optical_interface.context_resolver.resolve().camera
    resolution = CameraResolution(width=4, height=3)
    camera_model = PinholeCameraModel.from_field_of_view(
        resolution=resolution, field_of_view=camera.field_of_view
    )
    pixels = np.arange(resolution.width * resolution.height).reshape(
        resolution.width, resolution.height
    )
    return RenderedRGBDFrame(
        color_bgr=np.repeat(pixels[:, :, None], 3, axis=2).astype(np.uint8),
        depth_in_millimeters=pixels.astype(np.uint16),
        segmentation=pixels.astype(np.int32),
        object_color_map={},
        camera_model=camera_model,
    )


# %% Pixel and pose publication
@pytest.mark.parametrize(
    "view", [CASViews.COLOR_IMAGE, CASViews.DEPTH_IMAGE, CASViews.OBJECT_IMAGE]
)
def test_interface_publishes_row_first_pixels(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
    view: str,
) -> None:
    """
    Image publication preserves pixel values while placing height before width.
    """
    context = optical_interface.context_resolver.resolve()
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    images = {
        CASViews.COLOR_IMAGE: rectangular_frame.color_bgr,
        CASViews.DEPTH_IMAGE: rectangular_frame.depth_in_millimeters,
        CASViews.OBJECT_IMAGE: rectangular_frame.segmentation,
    }
    np.testing.assert_array_equal(cas.get(view), images[view].swapaxes(0, 1))


def test_interface_publishes_rendered_optical_pose(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The published camera pose includes the optical offset used for rendering.
    """
    context = optical_interface.context_resolver.resolve()
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    np.testing.assert_allclose(
        cas.camera_observation.world_T_camera.to_np(),
        context.camera.root.global_transform.to_np()
        @ context.camera.root_T_optical.to_np(),
    )


# %% Recorded optical pose


def test_sensor_projection_preserves_optical_attachment(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Projecting a rendered observation preserves both physical-root and optical poses.
    """
    context = optical_interface.context_resolver.resolve()
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    world = SensorWorldProjector(cas.camera_observation).project()
    camera = world.get_semantic_annotation_by_id(context.camera.id)
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        context.camera.root.global_transform.to_np(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        camera.world_T_optical.to_np(),
        cas.camera_observation.world_T_camera.to_np(),
        atol=1e-12,
    )


# %% Static optical pose


def test_static_optical_pose_binds_physical_root_pose(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A static optical pose binds the physical camera root through its optical offset.
    """
    context = optical_interface.context_resolver.resolve()
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    descriptor = StaticCameraTransformAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, pitch=0.1, yaw=0.2
    )
    expected_world_T_root = (
        descriptor.parameters.world_T_camera @ context.camera.root_T_optical.inverse()
    )
    bind_pose = Mock(return_value=expected_world_T_root)
    monkeypatch.setattr(CameraInterface, "bind_world_T_camera", bind_pose)
    annotator = StaticCameraTransformAnnotator(descriptor=descriptor)
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)
    annotator.update()
    np.testing.assert_allclose(
        bind_pose.call_args.kwargs["world_T_camera"].to_np(),
        expected_world_T_root.to_np(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cas.camera_observation.world_T_camera.to_np(),
        descriptor.parameters.world_T_camera.to_np(),
        atol=1e-12,
    )


def test_replayed_optical_pose_updates_physical_root(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replaying an optical pose updates the physical root through its optical attachment.
    """
    context = optical_interface.context_resolver.resolve()
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    world = SensorWorldProjector(cas.camera_observation).project()
    camera = world.get_semantic_annotation_by_id(context.camera.id)
    world_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, roll=0.2, reference_frame=world.root
    )
    observation = CameraObservation(
        camera=camera,
        effective_camera_model=rectangular_frame.camera_model,
        world_T_camera=world_T_optical,
        timestamp_nanoseconds=0,
    )
    SensorWorldProjector.apply_pose(world, observation)
    np.testing.assert_allclose(
        camera.world_T_optical.to_np(), world_T_optical.to_np(), atol=1e-12
    )


# %% Viewing-direction observations


def test_viewing_direction_observation_preserves_projected_root_pose(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An optical observation of a viewing-direction camera preserves its physical root.
    """
    context = optical_interface.context_resolver.resolve()
    context.camera.root_T_optical = None
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    world = SensorWorldProjector(cas.camera_observation).project()
    camera = world.get_semantic_annotation_by_id(context.camera.id)
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        context.camera.root.global_transform.to_np(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        camera.world_T_optical.to_np(),
        cas.camera_observation.world_T_camera.to_np(),
        atol=1e-12,
    )


def test_viewing_direction_observation_roundtrip_preserves_root_pose(
    optical_interface: SemDTRayTracerCameraInterface,
    rectangular_frame: RenderedRGBDFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Recorded viewing-direction images retain the optical offset needed for replay.
    """
    context = optical_interface.context_resolver.resolve()
    context.camera.root_T_optical = None
    monkeypatch.setattr(
        context.world.ray_tracer, "render", Mock(return_value=rectangular_frame)
    )
    cas = CAS()
    optical_interface.set_data(cas)
    codec = CameraObservationCodec()
    restored = codec.decode_with_tracker(
        codec.encode(cas.camera_observation),
        WorldEntityWithIDKwargsTracker.from_world(context.world),
    )
    world = SensorWorldProjector(restored).project()
    camera = world.get_semantic_annotation_by_id(context.camera.id)
    np.testing.assert_allclose(
        camera.root.global_transform.to_np(),
        context.camera.root.global_transform.to_np(),
        atol=1e-12,
    )
