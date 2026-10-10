"""Reusable pinhole RGB-D camera specification for world descriptors."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from robokudo.world_descriptor import CameraSpec
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    CameraRange,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3

# %% Camera identity


class PinholeRGBDCameraName(StrEnum):
    """Provide stable names used by the standard pinhole RGB-D camera."""

    DEFAULT = "semdt_camera_optical_frame"
    """Optical frame shared by the camera annotation and its root body."""


# %% Camera specification


@dataclass(init=False)
class PinholeRGBDCameraSpec(CameraSpec):
    """Describe a standard pinhole RGB-D camera with segmentation output.

    The caller supplies the pose because camera placement belongs to the containing
    world descriptor. Each instance owns its projection model and range so callers can
    adapt one specification without changing another.
    """

    MINIMUM_DISTANCE: ClassVar[float] = 0.05
    """Nearest usable distance in meters."""

    MAXIMUM_DISTANCE: ClassVar[float] = 8.0
    """Farthest usable distance in meters."""

    def __init__(self, pose: HomogeneousTransformationMatrix) -> None:
        """Initialize the standard camera at ``pose`` within its world.

        :param pose: Camera-root pose relative to the body that receives the camera.
        """
        super().__init__(
            name=PinholeRGBDCameraName.DEFAULT.value,
            pose=pose,
            forward_facing_axis=Vector3.Z(),
            camera_model=PinholeCameraModel.from_field_of_view(
                resolution=CameraResolution(),
                field_of_view=FieldOfView(),
            ),
            camera_range=CameraRange(
                minimum_distance=self.MINIMUM_DISTANCE,
                maximum_distance=self.MAXIMUM_DISTANCE,
            ),
            modalities=(
                CameraModality.COLOR,
                CameraModality.DEPTH,
                CameraModality.SEGMENTATION,
            ),
        )
