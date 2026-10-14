"""
Reusable pinhole RGB-D camera specification for world descriptors.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from robokudo.world_descriptor import CameraSpec
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    CameraRange,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3

# %% Camera specification


@dataclass(kw_only=True)
class PinholeRGBDCameraSpec(CameraSpec):
    """
    Describe a pinhole RGB-D camera with segmentation output and configurable defaults.
    """

    pose: HomogeneousTransformationMatrix
    """
    Camera-root pose relative to its parent body.
    """

    name: str = "semdt_camera_optical_frame"
    """
    Name shared by the camera annotation and its root body.
    """

    forward_facing_axis: Vector3 = field(default_factory=Vector3.Z)
    """
    Root-frame viewing axis selecting the fixed optical convention.
    """

    camera_model: PinholeCameraModel = field(
        default_factory=lambda: PinholeCameraModel.from_field_of_view(
            resolution=CameraResolution(), field_of_view=FieldOfView()
        )
    )
    """
    Projection model matching the camera resolution and field of view.
    """

    camera_range: CameraRange = field(
        default_factory=lambda: CameraRange(
            minimum_distance=0.05,
            maximum_distance=8.0,
        )
    )
    """
    Usable distance interval of the camera.
    """

    modalities: tuple[CameraModality, ...] = (
        CameraModality.COLOR,
        CameraModality.DEPTH,
        CameraModality.SEGMENTATION,
    )
    """
    Kinds of image data the camera produces.
    """
