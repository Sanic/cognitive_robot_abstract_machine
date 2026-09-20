"""
Simulated RGB-D camera interface backed by SemDT RayTracer.
"""

from __future__ import annotations

import io
import time
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from PIL import Image as PILImage, UnidentifiedImageError
from pyglet.canvas.xlib import NoSuchDisplayException
from sensor_msgs.msg import CameraInfo

import robokudo.world as rk_world
from robokudo.cas import CASViews, CAS
from robokudo.exceptions import (
    CameraAnnotationAmbiguous,
    CameraAnnotationMissing,
    CameraPoseOverrideUnavailable,
    CameraResolutionUnavailable,
)
from robokudo.io.camera_interface import CameraInterface, ROSCameraInterface
from robokudo.types.camera import CameraObservation
from robokudo.utils.module_loader import ModuleLoader
from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortionModel,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.robots.robot_parts import Camera, RobotCamera
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% Semantic Digital Twin ray-tracer interface


class SemDTRayTracerCameraInterface(CameraInterface):
    """
    Render RGB-D camera data from a Semantic Digital Twin world.

    The interface renders from a semantic camera in either a supplied runtime world or
    a configured world descriptor. It writes the rendered images and the camera state
    used for the frame into the CAS.

    .. note::
        The configured camera pose places the camera root. The camera's semantic
        forward axis determines the rendered viewing direction.
    """

    def __init__(self, camera_config):
        """
        Initialize the camera interface.

        :param camera_config: RayTracer camera configuration descriptor.
        """
        super().__init__(camera_config)
        self.module_loader = ModuleLoader()
        """
        Loader used to import configured SemDT world descriptors.
        """
        self._world: World | None = None
        """
        Shared runtime world after its first resolution.
        """
        self._camera: Camera | None = None
        """
        Selected semantic camera after its first resolution.
        """

    def has_new_data(self) -> bool:
        """
        Report whether rendered camera data is available.

        :return: Always ``True`` because simulated frames are rendered on demand.
        """
        # Simulated camera can always render a fresh frame.
        return True

    def set_data(self, cas: CAS) -> None:
        """
        Render a simulated RGB-D frame and write it into the CAS.

        :param cas: CAS that receives rendered camera data and frame metadata.
        """
        world = self._load_runtime_world()
        camera = self._load_camera(world)
        camera_model = camera.camera_model
        if not isinstance(camera_model, PinholeCameraModel):
            raise CameraResolutionUnavailable(camera_name=camera.name.name)
        resolution = camera_model.resolution
        field_of_view = camera_model.field_of_view
        minimum_distance = camera.camera_range.minimum_distance
        maximum_distance = camera.camera_range.maximum_distance
        world_T_render_camera = camera.root_T_forward_view
        world_T_camera = camera.root.global_transform

        ray_tracer = world.ray_tracer
        segmentation, depth_m = self._render_segmentation_and_depth(
            ray_tracer=ray_tracer,
            camera_to_world=world_T_render_camera,
            resolution=resolution,
            field_of_view=field_of_view,
            min_distance=minimum_distance,
            max_distance=maximum_distance,
        )

        color_bgr, object_color_map = self._render_color_image(
            world=world,
            ray_tracer=ray_tracer,
            camera_to_world=world_T_render_camera,
            segmentation=segmentation,
            resolution=resolution,
            field_of_view=field_of_view,
        )
        depth_mm = self._depth_m_to_mm(depth_m)

        camera_info, camera_intrinsic = self._build_camera_models(
            frame_id=camera.root.name.name,
            camera_model=camera_model,
        )

        timestamp_ns = time.time_ns()
        camera_info.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
        camera_info.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)

        cas.set(CASViews.COLOR_IMAGE, color_bgr)
        cas.set(CASViews.DEPTH_IMAGE, depth_mm)
        cas.set(CASViews.CAMERA_INFO, camera_info)
        cas.set(CASViews.CAMERA_INTRINSIC, camera_intrinsic)
        cas.set(CASViews.COLOR2DEPTH_RATIO, self.camera_config.color2depth_ratio)
        cas.set(CASViews.OBJECT_IMAGE, segmentation)
        cas.set(CASViews.OBJECT_COLOR_MAP, object_color_map)
        # Shared SemDT ground-truth world for this frame. Consumers must treat as read-only.
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

    def _load_camera(self, world: World) -> Camera:
        """
        Select and initialize the semantic camera once.

        :param world: Runtime world containing the camera.
        :return: Camera used for all rendered frames.
        """
        if self._camera is not None:
            return self._camera

        camera = self._select_camera(world)
        self._apply_camera_pose(world, camera)
        self._camera = camera
        return camera

    def _apply_camera_pose(self, world: World, camera: Camera) -> None:
        """
        Apply the configured initial camera pose when one is present.

        :param world: Runtime world that owns the camera connection.
        :param camera: Selected semantic camera.
        :raises CameraPoseOverrideUnavailable: If the camera attachment is immutable.
        """
        camera_pose = self.camera_config.camera_pose
        if camera_pose is None:
            return

        parent_connection = camera.root.parent_connection
        if not isinstance(parent_connection, Connection6DoF):
            raise CameraPoseOverrideUnavailable(
                camera_name=camera.name.name,
                connection_type=(
                    None
                    if parent_connection is None
                    else parent_connection.__class__.__name__
                ),
            )

        reference_frame = camera_pose.reference_frame or parent_connection.parent
        reference_T_camera = HomogeneousTransformationMatrix(
            data=camera_pose,
            reference_frame=reference_frame,
            child_frame=camera.root,
        )
        with world.modify_world():
            parent_connection.origin = reference_T_camera

    def _select_camera(self, world: World) -> Camera:
        """
        Select the semantic camera used for rendering.

        :param world: Runtime world containing camera annotations.
        :return: Camera that provides the rendered frame.
        :raises CameraAnnotationMissing: If a configured camera does not exist.
        :raises CameraAnnotationAmbiguous: If automatic selection is ambiguous.
        """
        cameras = world.get_semantic_annotations_by_type(Camera)
        camera_name = self.camera_config.camera_name
        if camera_name is not None:
            named_cameras = [
                camera
                for camera in cameras
                if camera.name.name == camera_name or str(camera.name) == camera_name
            ]
            if len(named_cameras) == 0:
                raise CameraAnnotationMissing(camera_name=camera_name)
            if len(named_cameras) > 1:
                raise CameraAnnotationAmbiguous(
                    camera_names=tuple(str(camera.name) for camera in named_cameras)
                )
            return named_cameras[0]

        if len(cameras) == 1:
            return cameras[0]

        default_cameras = [
            camera
            for camera in cameras
            if isinstance(camera, RobotCamera) and camera.default_camera
        ]
        if len(default_cameras) == 1:
            return default_cameras[0]
        if len(cameras) > 1:
            raise CameraAnnotationAmbiguous(
                camera_names=tuple(str(camera.name) for camera in cameras)
            )
        raise CameraAnnotationMissing()

    def _load_runtime_world(self) -> World:
        """
        Load the configured SemDT world and install it as runtime world.

        :return: Runtime world instance used for rendering.
        """
        if self._world is not None:
            return self._world

        if self.camera_config.world is not None:
            self._world = self.camera_config.world
        else:
            world_descriptor = self.module_loader.load_world_descriptor(
                ros_pkg_name=self.camera_config.world_descriptor_ros_package,
                module_name=self.camera_config.world_descriptor_name,
            )
            self._world = world_descriptor.world

        rk_world.set_world(self._world)
        rk_world.init_world_entity_tracker_from_world(self._world)
        return self._world

    @staticmethod
    def _render_segmentation_and_depth(
        ray_tracer: RayTracer,
        camera_to_world: HomogeneousTransformationMatrix,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
        min_distance: float,
        max_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render object segmentation and projective depth.

        :param ray_tracer: Renderer that creates camera rays and returns their scene
            intersections.
        :param camera_to_world: Camera pose used by the RayTracer renderer.
        :param resolution: Image resolution.
        :param field_of_view: Camera field of view.
        :param min_distance: Minimum valid ray-hit distance.
        :param max_distance: Maximum valid ray-hit distance.
        :return: Segmentation indices and depth image in meters.
        """
        segmentation = (
            np.zeros((resolution.width, resolution.height), dtype=np.int32) - 1
        )
        depth_m = (
            np.zeros((resolution.width, resolution.height), dtype=np.float32) - 1.0
        )

        ray_origins, ray_directions, pixels = ray_tracer.create_camera_rays(
            camera_to_world, resolution=resolution, field_of_view=field_of_view
        )
        target_points = ray_origins + ray_directions * 10.0
        points, index_ray, bodies = ray_tracer.ray_test(
            ray_origins,
            target_points,
            multiple_hits=True,
            min_distance=min_distance,
            max_distance=max_distance,
        )

        if len(index_ray) == 0:
            return segmentation, depth_m

        unique_index = np.unique(index_ray, return_index=True)[1]
        index_ray = index_ray[unique_index]
        points = points[unique_index]
        body_indices = np.asarray([body.index for body in bodies], dtype=np.int32)[
            unique_index
        ]
        pixel_ray = pixels[index_ray]
        segmentation[pixel_ray[:, 0], pixel_ray[:, 1]] = body_indices

        # Trimesh camera looks along -z in its local frame. Convert hit points to
        # that camera frame and use projective z-depth (not range) for RGB-D.
        scene_camera_transform = ray_tracer.scene.graph[ray_tracer.scene.camera.name]
        if isinstance(scene_camera_transform, tuple):
            scene_camera_transform = scene_camera_transform[0]
        world_to_camera = np.linalg.inv(np.asarray(scene_camera_transform))
        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1
        )
        points_camera = (world_to_camera @ points_h.T).T
        z_depth = -points_camera[:, 2]
        valid_depth = z_depth > 0.0
        pixel_ray = pixel_ray[valid_depth]
        z_depth = z_depth[valid_depth]
        depth_m[pixel_ray[:, 0], pixel_ray[:, 1]] = z_depth.astype(np.float32)

        return segmentation, depth_m

    def _render_color_image(
        self,
        world: World,
        ray_tracer,
        camera_to_world: HomogeneousTransformationMatrix,
        segmentation: np.ndarray,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Render a BGR color image for the current frame.

        :param world: Runtime world that provides semantic body colors.
        :param ray_tracer: SemDT ray tracer used for optional mesh rendering.
        :param camera_to_world: Camera pose used by the RayTracer renderer.
        :param segmentation: Body-index segmentation image.
        :param resolution: Image resolution.
        :param field_of_view: Camera field of view.
        :return: BGR image and optional RGB-to-object-name color map.
        """
        rgb_mode = str(self.camera_config.rgb_mode).strip().lower()
        if rgb_mode == "trimesh":
            trimesh_image = self._try_render_trimesh_rgb(
                ray_tracer=ray_tracer,
                camera_to_world=camera_to_world,
                resolution=resolution,
                field_of_view=field_of_view,
            )
            if trimesh_image is not None:
                return trimesh_image, {}
            self.rk_logger.warning(
                "SemDTRayTracerCameraInterface: trimesh RGB rendering failed. Falling back to semantic colors."
            )

        semantic_rgb, object_color_map = self._render_semantic_rgb(
            world=world, segmentation=segmentation
        )
        return semantic_rgb[:, :, ::-1].copy(), object_color_map

    def _try_render_trimesh_rgb(
        self,
        ray_tracer,
        camera_to_world: HomogeneousTransformationMatrix,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
    ) -> np.ndarray | None:
        """
        Render textured mesh colors through the RayTracer scene.

        :param ray_tracer: SemDT ray tracer that owns the Trimesh scene.
        :param camera_to_world: Camera pose used by the RayTracer renderer.
        :param resolution: Image resolution.
        :param field_of_view: Camera field of view.
        :return: BGR image when rendering succeeds, otherwise ``None``.
        """
        try:
            # Keep RayTracer camera pose/FOV/resolution in sync with this frame.
            ray_tracer.create_camera_rays(
                camera_to_world, resolution=resolution, field_of_view=field_of_view
            )
            png_data = ray_tracer.scene.save_image(
                resolution=(resolution.width, resolution.height), visible=False
            )
            image = np.array(PILImage.open(io.BytesIO(png_data)).convert("RGB"))
            return image[:, :, ::-1].copy()
        except (NoSuchDisplayException, UnidentifiedImageError, OSError, ValueError):
            return None

    def _render_semantic_rgb(
        self, world: World, segmentation: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Render deterministic semantic RGB colors from segmentation labels.

        :param world: Runtime world that maps body indices to bodies.
        :param segmentation: Body-index segmentation image.
        :return: RGB image and RGB-to-object-name color map.
        """
        rgb = np.zeros(
            (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8
        )
        object_color_map: Dict[str, str] = {}
        unique_indices = np.unique(segmentation)
        for body_index in unique_indices:
            if body_index < 0:
                continue

            body = world.kinematic_structure[int(body_index)]
            body_rgb = self._rgb_for_body(body)
            rgb[segmentation == body_index] = body_rgb
            object_color_map[f"{body_rgb[0]},{body_rgb[1]},{body_rgb[2]}"] = (
                body.name.name
            )

        return rgb, object_color_map

    @staticmethod
    def _rgb_for_body(body: Body) -> np.ndarray:
        """
        Return the semantic RGB color for a world body.

        :param body: World body whose collision or visual color is used.
        :return: RGB color encoded as three unsigned bytes.
        """
        if len(body.collision) > 0:
            color = body.collision[0].color
        elif len(body.visual) > 0:
            color = body.visual[0].color
        else:
            return np.array([128, 128, 128], dtype=np.uint8)

        return np.array(
            [
                int(np.clip(round(color.R * 255.0), 0, 255)),
                int(np.clip(round(color.G * 255.0), 0, 255)),
                int(np.clip(round(color.B * 255.0), 0, 255)),
            ],
            dtype=np.uint8,
        )

    @staticmethod
    def _depth_m_to_mm(depth_m: np.ndarray) -> np.ndarray:
        """
        Convert meter depth values to unsigned millimeter depth values.

        :param depth_m: Depth image in meters with negative values for misses.
        :return: Depth image in millimeters with misses encoded as zero.
        """
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        valid_mask = depth_m >= 0.0
        depth_mm[valid_mask] = np.clip(
            np.round(depth_m[valid_mask] * 1000.0), 0, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        return depth_mm

    @staticmethod
    def _build_camera_models(
        frame_id: str, camera_model: PinholeCameraModel
    ) -> Tuple[CameraInfo, o3d.camera.PinholeCameraIntrinsic]:
        """
        Build ROS and Open3D pinhole camera models.

        :param frame_id: Camera frame name stored in the ROS camera info header.
        :param camera_model: Effective pinhole calibration of the frame.
        :return: ROS camera info and matching Open3D intrinsic model.
        """
        camera_info = CameraInfo()
        camera_info.header.frame_id = frame_id
        camera_info.width = camera_model.resolution.width
        camera_info.height = camera_model.resolution.height
        if camera_model.distortion.model == CameraDistortionModel.NONE:
            camera_info.distortion_model = CameraDistortionModel.PLUMB_BOB.value
            camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            camera_info.distortion_model = camera_model.distortion.model.value
            camera_info.d = list(camera_model.distortion.coefficients)
        camera_info.k = [
            camera_model.focal_length_x,
            0.0,
            camera_model.principal_point_x,
            0.0,
            camera_model.focal_length_y,
            camera_model.principal_point_y,
            0.0,
            0.0,
            1.0,
        ]
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [
            camera_model.focal_length_x,
            0.0,
            camera_model.principal_point_x,
            0.0,
            0.0,
            camera_model.focal_length_y,
            camera_model.principal_point_y,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera_model.resolution.width,
            camera_model.resolution.height,
            camera_model.focal_length_x,
            camera_model.focal_length_y,
            camera_model.principal_point_x,
            camera_model.principal_point_y,
        )

        return camera_info, camera_intrinsic
