from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, ClassVar

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


# %% World sources


@dataclass(slots=True)
class WorldDescriptorSource:
    """Load a standalone semantic world and camera from a world descriptor."""

    ros_package: str = "robokudo"
    """ROS package containing the world descriptor module."""

    descriptor_name: str = "world_semdt_raytracer_tabletop"
    """Module in ``descriptors/worlds`` that defines ``WorldDescriptor``."""

    camera_name: PrefixedName | None = None
    """Exact semantic camera name, or ``None`` for the descriptor's sole camera."""

    camera_pose: HomogeneousTransformationMatrix | None = None
    """Optional initial pose for the selected camera root."""


@dataclass(slots=True)
class RuntimeRobotWorldSource:
    """Select a robot camera from RoboKudo's runtime semantic world.

    .. todo::
        Replace the current global runtime-world lookup with the synchronized world
        reference once ``WorldSynchronizer`` provides that API.
    """

    robot_name: PrefixedName | None = None
    """Exact robot name, or ``None`` when the runtime world contains one robot."""


# %% Semantic Digital Twin ray-tracer configuration


class SemDTRGBMode(StrEnum):
    """Select how the ray tracer produces the color image."""

    SEMANTIC = "semantic"
    """Color visible bodies with their semantic geometry colors."""

    TRIMESH = "trimesh"
    """Render mesh materials and fall back to semantic colors when unavailable."""


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

    source: WorldDescriptorSource | RuntimeRobotWorldSource = field(
        default_factory=WorldDescriptorSource
    )
    """World source and camera-selection policy used for rendering."""

    color2depth_ratio: tuple[float, float] = (1.0, 1.0)
    """
    Scale factor from RGB image to depth image resolution.
    """

    rgb_mode: SemDTRGBMode = SemDTRGBMode.SEMANTIC
    """
    Method used to create the rendered color image.
    """
