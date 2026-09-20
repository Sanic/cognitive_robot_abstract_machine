from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, ClassVar, Optional, Tuple

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
    from semantic_digital_twin.world import World


# %% Semantic Digital Twin ray-tracer configuration


@dataclass(slots=True)
class SemDTRayTracerCameraConfig(BaseCameraConfig):
    """
    Configuration for a simulated RGB-D camera backed by SemDT's RayTracer.
    """

    registry_name: ClassVar[str] = "semdt_raytracer"
    """
    Name under which the camera configuration is registered.
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

    camera_name: Optional[str] = None
    """
    Semantic camera name.

    A unique or default camera is used when omitted.
    """

    camera_pose: Optional[HomogeneousTransformationMatrix] = None
    """
    Optional initial pose for the selected camera root.

    A pose without a reference frame is interpreted relative to the camera root's parent
    body.
    """

    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    """
    Scale factor from RGB image to depth image resolution.
    """

    rgb_mode: str = "semantic"
    """
    RGB rendering mode: 'semantic' or 'trimesh' (falls back to semantic on failure).
    """
