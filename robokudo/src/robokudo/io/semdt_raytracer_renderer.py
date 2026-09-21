"""
Render RoboKudo RGB-D products with the Semantic Digital Twin ray tracer.

SemDT's :class:`~semantic_digital_twin.spatial_computations.raytracer.RayTracer`
provides camera rays and scene intersections. RoboKudo additionally requires
segmentation and projective optical-axis depth from the same hit query; SemDT's
standalone helpers would trace twice and report ray-direction depth. This module
adapts the shared hits into aligned segmentation, RGB, an object-color map, and
unsigned-millimeter depth.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage, UnidentifiedImageError
from pyglet.canvas.xlib import NoSuchDisplayException

from robokudo.descriptors.camera_configs.config_semdt_raytracer import SemDTRGBMode
from robokudo.exceptions import CameraResolutionUnavailable
from robokudo.io.semdt_camera_context import RayTracingContext
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% Rendered frame


@dataclass
class RenderedRGBDFrame:
    """
    Contain the aligned image products of one ray-traced camera frame.
    """

    color_bgr: np.ndarray
    """Color image encoded in OpenCV BGR channel order."""

    depth_mm: np.ndarray
    """
    Projective depth image encoded in unsigned millimeters.
    """

    segmentation: np.ndarray
    """Body-index segmentation image aligned with the color and depth images."""

    object_color_map: dict[str, str]
    """
    Mapping from semantic RGB color keys to object names.
    """


# %% Semantic Digital Twin rendering


class SemDTRayTracerRenderer:
    """
    Adapt SemDT ray-tracing primitives to one aligned RoboKudo RGB-D frame.

    The underlying :class:`RayTracer` remains responsible for scene maintenance, ray
    construction, and collision queries. This adapter owns the frame-level guarantees
    expected by RoboKudo: one shared set of visible surface hits, body-index
    segmentation, optical-axis depth, matching color output, and the required image
    encodings.

    Calling ``RayTracer.create_segmentation_mask()`` and
    ``RayTracer.create_depth_map()`` separately would trace the scene twice and would
    return ray-direction depth rather than the projective z-depth used to construct an
    RGB-D point cloud from pinhole intrinsics.
    """

    def __init__(
        self,
        rgb_mode: SemDTRGBMode | str,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize rendering behavior.

        :param rgb_mode: Method used to produce the color image.
        :param logger: Logger used to report mesh-rendering fallback.
        """
        self.rgb_mode = (
            rgb_mode
            if isinstance(rgb_mode, SemDTRGBMode)
            else SemDTRGBMode(rgb_mode.strip().lower())
        )
        """
        Normalized color-rendering mode.
        """
        self.logger = logger
        """
        Logger used when optional mesh rendering is unavailable.
        """

    def render(self, context: RayTracingContext) -> RenderedRGBDFrame:
        """
        Render one aligned RGB-D frame from ``context``.

        Segmentation and projective depth share one ray-intersection result. Color is
        then rendered for the same camera model and pose, either from Trimesh or from
        the body labels in that segmentation image.

        :param context: Resolved semantic world and camera.
        :return: Aligned color, depth, segmentation, and semantic color mapping.
        :raises CameraResolutionUnavailable: If the camera has no pinhole model.
        """
        world = context.world
        camera = context.camera
        camera_model = camera.camera_model
        if not isinstance(camera_model, PinholeCameraModel):
            raise CameraResolutionUnavailable(camera_name=camera.name.name)

        resolution = camera_model.resolution
        field_of_view = camera_model.field_of_view
        ray_tracer = world.ray_tracer
        camera_to_world = camera.root_T_forward_view
        segmentation, depth_m = self._render_segmentation_and_depth(
            ray_tracer=ray_tracer,
            camera_to_world=camera_to_world,
            resolution=resolution,
            field_of_view=field_of_view,
            min_distance=camera.camera_range.minimum_distance,
            max_distance=camera.camera_range.maximum_distance,
        )
        color_bgr, object_color_map = self._render_color_image(
            world=world,
            ray_tracer=ray_tracer,
            camera_to_world=camera_to_world,
            segmentation=segmentation,
            resolution=resolution,
            field_of_view=field_of_view,
        )
        return RenderedRGBDFrame(
            color_bgr=color_bgr,
            depth_mm=self._depth_m_to_mm(depth_m),
            segmentation=segmentation,
            object_color_map=object_color_map,
        )

    @staticmethod
    def _render_segmentation_and_depth(
        ray_tracer: RayTracer,
        camera_to_world: HomogeneousTransformationMatrix,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
        min_distance: float,
        max_distance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Render object segmentation and projective depth with one intersection pass.

        SemDT's camera rays identify the first visible body per pixel. Their hit points
        are transformed back into the Trimesh camera frame, whose viewing axis is
        negative z, so the returned depth is ``-z`` rather than distance along the
        oblique viewing ray. Ray distance and projective z-depth are equal on the
        optical axis but increasingly differ for off-axis pixels. RoboKudo’s pinhole
        back-projection expects projective z-depth.

        :param ray_tracer: Renderer that creates camera rays and scene intersections.
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
            camera_to_world,
            resolution=resolution,
            field_of_view=field_of_view,
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
        ray_tracer: RayTracer,
        camera_to_world: HomogeneousTransformationMatrix,
        segmentation: np.ndarray,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
    ) -> tuple[np.ndarray, dict[str, str]]:
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
        if self.rgb_mode == SemDTRGBMode.TRIMESH:
            trimesh_image = self._try_render_trimesh_rgb(
                ray_tracer=ray_tracer,
                camera_to_world=camera_to_world,
                resolution=resolution,
                field_of_view=field_of_view,
            )
            if trimesh_image is not None:
                return trimesh_image, {}
            self.logger.warning(
                "SemDT mesh RGB rendering failed; using semantic colors."
            )

        semantic_rgb, object_color_map = self._render_semantic_rgb(
            world=world,
            segmentation=segmentation,
        )
        return semantic_rgb[:, :, ::-1].copy(), object_color_map

    @staticmethod
    def _try_render_trimesh_rgb(
        ray_tracer: RayTracer,
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
            ray_tracer.create_camera_rays(
                camera_to_world,
                resolution=resolution,
                field_of_view=field_of_view,
            )
            png_data = ray_tracer.scene.save_image(
                resolution=(resolution.width, resolution.height),
                visible=False,
            )
            image = np.array(PILImage.open(io.BytesIO(png_data)).convert("RGB"))
            return image[:, :, ::-1].copy()
        except (NoSuchDisplayException, UnidentifiedImageError, OSError, ValueError):
            return None

    @classmethod
    def _render_semantic_rgb(
        cls,
        world: World,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, str]]:
        """
        Render deterministic semantic RGB colors from segmentation labels.

        :param world: Runtime world that maps body indices to bodies.
        :param segmentation: Body-index segmentation image.
        :return: RGB image and RGB-to-object-name color map.
        """
        rgb = np.zeros(
            (segmentation.shape[0], segmentation.shape[1], 3),
            dtype=np.uint8,
        )
        object_color_map: dict[str, str] = {}
        visible_body_indices = np.unique(segmentation[segmentation >= 0])
        for body_index in visible_body_indices:
            body = world.kinematic_structure[int(body_index)]
            body_rgb = cls._rgb_for_body(body)
            rgb[segmentation == body_index] = body_rgb
            object_color_map[cls._semantic_color_key(body_rgb)] = body.name.name
        return rgb, object_color_map

    @staticmethod
    def _semantic_color_key(rgb: np.ndarray) -> str:
        """
        Serialize an RGB color for the object color map.

        :param rgb: Three-channel unsigned-byte RGB color.
        :return: Comma-separated decimal channel values.
        """
        return ",".join(str(channel) for channel in rgb)

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
            np.round(depth_m[valid_mask] * 1000.0),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        return depth_mm
