from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, ClassVar, Optional, Tuple

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(slots=True)
class SemDTRayTracerCameraConfig(BaseCameraConfig):
    """
    Configuration for a simulated RGB-D camera backed by SemDT's RayTracer.
    """

    registry_name: ClassVar[str] = "semdt_raytracer"
    """
    Name under which the camera configuration is registered.
    """

    default_camera_frame: ClassVar[str] = "semdt_camera_optical_frame"
    """
    Camera frame used when adapting a legacy descriptor.
    """

    default_resolution: ClassVar[int] = 512
    """
    Image size used when a legacy descriptor supplies no resolution.
    """

    default_field_of_view_degrees: ClassVar[float] = 90.0
    """
    Viewing angle used when a legacy descriptor supplies no field of view.
    """

    default_minimum_distance: ClassVar[float] = 0.05
    """
    Nearest rendered distance used for a legacy descriptor.
    """

    default_maximum_distance: ClassVar[float] = 8.0
    """
    Farthest rendered distance used for a legacy descriptor.
    """

    interface_type: str = "SemDTRayTracer"
    """
    Camera interface selected by the collection-reader factory.
    """

    world_descriptor_ros_package: str = "robokudo"
    """
    ROS package containing the world descriptor module.
    """

    world_descriptor_name: str = "world_semdt_raytracer_tabletop"
    """
    Module name in descriptors/worlds that defines class WorldDescriptor.
    """

    world: Optional[World] = field(default=None, repr=False)
    """
    Existing semantic world to render instead of loading a descriptor.
    """

    world_frame: Optional[str] = None
    """
    World frame to use as camera pose reference.

    If None, descriptor root is used.
    """

    camera_name: Optional[str] = None
    """
    Semantic camera name.

    A unique or default camera is used when omitted.
    """

    camera_frame: Optional[str] = None
    """
    Frame name used only when creating a camera for a legacy world descriptor.
    """

    camera_x: float = -1.20
    """
    Legacy camera x position in the configured world frame.
    """

    camera_y: float = 0.40
    """
    Legacy camera y position in the configured world frame.
    """

    camera_z: float = 1.05
    """
    Legacy camera z position in the configured world frame.
    """

    camera_roll: float = -2.2689280275926285
    """
    Legacy camera roll in radians.
    """

    camera_pitch: float = 0.0
    """
    Legacy camera pitch in radians.
    """

    camera_yaw: float = -0.2707963267948965
    """
    Legacy camera yaw in radians.
    """

    resolution: Optional[int] = None
    """
    Optional square output-resolution override in pixels.
    """

    fov_deg: Optional[float] = None
    """
    Optional symmetric output field-of-view override in degrees.
    """

    min_distance: Optional[float] = None
    """
    Optional minimum rendered distance in meters.
    """

    max_distance: Optional[float] = None
    """
    Optional maximum rendered distance in meters.
    """

    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    """
    Scale factor from RGB image to depth image resolution.
    """

    rgb_mode: str = "semantic"
    """
    RGB rendering mode: 'semantic' or 'trimesh' (falls back to semantic on failure).
    """
