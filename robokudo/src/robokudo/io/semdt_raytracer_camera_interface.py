"""
Simulated RGB-D camera interface backed by the SemDT ray tracer.
"""

from __future__ import annotations

import time

import open3d as o3d
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.io.camera_interface import CameraInterface, ROSCameraInterface
from robokudo.io.semdt_camera_context import (
    RayTracingContext,
    SemDTCameraContextResolver,
)
from robokudo.io.semdt_raytracer_renderer import (
    RenderedRGBDFrame,
    SemDTRayTracerRenderer,
)
from robokudo.types.camera import CameraObservation
from robokudo.utils.type_conversion import (
    o3d_camera_intrinsics_from_ros_camera_info,
)
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel

# %% Semantic Digital Twin ray-tracer interface


class SemDTRayTracerCameraInterface(CameraInterface):
    """
    Render RGB-D camera data from a Semantic Digital Twin world.

    The interface orchestrates context resolution, rendering, and publication of one
    coherent semantic camera observation into the CAS.
    """

    def __init__(self, camera_config: SemDTRayTracerCameraConfig) -> None:
        """
        Initialize the camera interface.

        :param camera_config: Ray-tracer camera configuration descriptor.
        """
        super().__init__(camera_config)
        self.context_resolver = SemDTCameraContextResolver(camera_config)
        """
        Resolver for the configured world and semantic camera.
        """
        self.renderer = SemDTRayTracerRenderer(
            rgb_mode=camera_config.rgb_mode,
            logger=self.rk_logger,
        )
        """
        Renderer that produces aligned image values from a resolved context.
        """

    def has_new_data(self) -> bool:
        """
        Report whether rendered camera data is available.

        :return: Always ``True`` because simulated frames are rendered on demand.
        """
        return True

    def set_data(self, cas: CAS) -> None:
        """
        Render a simulated RGB-D frame and write it into the CAS.

        :param cas: CAS that receives rendered camera data and frame metadata.
        """
        context = self.context_resolver.resolve()
        frame = self.renderer.render(context)
        camera_model = frame.camera_model

        camera_info = self.camera_info_from_camera_model(
            camera_model=camera_model,
            frame_id=context.camera.root.name.name,
        )
        camera_intrinsic = o3d_camera_intrinsics_from_ros_camera_info(camera_info)
        timestamp_ns = time.time_ns()
        camera_info.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
        camera_info.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
        self._write_frame_to_cas(
            cas=cas,
            context=context,
            frame=frame,
            camera_model=camera_model,
            camera_info=camera_info,
            camera_intrinsic=camera_intrinsic,
            timestamp_ns=timestamp_ns,
        )

    def _write_frame_to_cas(
        self,
        cas: CAS,
        context: RayTracingContext,
        frame: RenderedRGBDFrame,
        camera_model: PinholeCameraModel,
        camera_info: CameraInfo,
        camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
        timestamp_ns: int,
    ) -> None:
        """
        Publish one rendered frame and its semantic camera state.

        :param cas: CAS receiving the frame.
        :param context: World and camera used to render the frame.
        :param frame: Aligned rendered image products.
        :param camera_model: Effective pinhole model used for rendering.
        :param camera_info: ROS compatibility calibration for the frame.
        :param camera_intrinsic: Open3D compatibility calibration for the frame.
        :param timestamp_ns: Frame timestamp in nanoseconds since the epoch.
        """
        world = context.world
        camera = context.camera
        world_T_camera = camera.root.global_transform

        cas.set(CASViews.COLOR_IMAGE, frame.color_bgr)
        cas.set(CASViews.DEPTH_IMAGE, frame.depth_mm)
        cas.set(CASViews.CAMERA_INFO, camera_info)
        cas.set(CASViews.CAMERA_INTRINSIC, camera_intrinsic)
        cas.set(CASViews.COLOR2DEPTH_RATIO, self.camera_config.color2depth_ratio)
        cas.set(CASViews.OBJECT_IMAGE, frame.segmentation)
        cas.set(CASViews.OBJECT_COLOR_MAP, frame.object_color_map)
        cas.set_ref(CASViews.GROUND_TRUTH_WORLD_REFERENCE, world)
        cas.camera_observation = CameraObservation(
            camera=camera,
            camera_model=camera_model,
            world_T_camera=world_T_camera,
            timestamp_nanoseconds=timestamp_ns,
        )
        cas.camera_to_world_transform = world_T_camera
        cas.data_timestamp = timestamp_ns
        ROSCameraInterface.store_legacy_camera_to_world_transform_from_cas(cas)
