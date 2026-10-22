from __future__ import annotations

# %% imports

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from rclpy.qos import ReliabilityPolicy

from krrood.ormatic.utils import classproperty
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import (
    StaticCameraTransformComponent,
    TfComponent,
)
from robokudo.io.exceptions import InvalidDepthCameraConfiguration

# %% source identity


class DepthCameraSource(StrEnum):
    """
    Registered source and interface identifiers for rectified depth.
    """

    FOUNDATION_STEREO = "foundation_stereo"
    OLIVE = "olive"
    RECTIFIED_DEPTH = "rectified_depth"


# %% rectified depth configuration


@dataclass(kw_only=True)
class FoundationStereoCameraConfig(
    BaseCameraConfig, TfComponent, StaticCameraTransformComponent
):
    """
    Read rectified left images and metric Fast-FoundationStereo depth.
    """

    interface_type: str = DepthCameraSource.RECTIFIED_DEPTH
    """
    Interface identifier for rectified metric depth.
    """

    topic_color: str = "/left/image_rect_color"
    """
    Rectified left image topic.
    """

    topic_depth: str = "/ffs/depth"
    """
    Float32 meter depth topic aligned to the left image.
    """

    topic_camera_info: str = "/left/camera_info"
    """
    Per-frame left calibration topic.
    """

    tf_from: str = "olive_camera_left_optical_frame"
    """
    Rectified optical frame shared by all input messages.
    """

    lookup_viewpoint: bool = False
    """
    Whether to require acquisition-time TF before consuming a frame.
    """

    color_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for camera images.
    """

    depth_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for inferred depth.
    """

    calibration_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for calibration messages.
    """

    history_depth: int = 2
    """
    DDS history length for each subscription.
    """

    synchronization_queue_size: int = 30
    """
    Maximum messages per stream retained for delayed depth matching.
    """

    calibration_tolerance: float = 0.03
    """
    Maximum calibration timestamp difference in seconds.
    """

    max_frame_age: float = 1.0
    """
    Maximum acquisition age in seconds at receipt and consumption.
    """

    max_depth: float = 10.0
    """
    Maximum valid depth in meters.
    """

    @classproperty
    def registry_name(cls: type[FoundationStereoCameraConfig]) -> str:
        """
        The camera source name recognized by the descriptor factory.
        """
        return DepthCameraSource.FOUNDATION_STEREO

    def __post_init__(self) -> None:
        """
        Reject unbounded buffering and contradictory transform settings.
        """
        if self.history_depth <= 0 or self.synchronization_queue_size <= 0:
            raise InvalidDepthCameraConfiguration()
        if any(
            not isfinite(value) or value <= 0
            for value in (self.max_frame_age, self.max_depth)
        ):
            raise InvalidDepthCameraConfiguration()
        if not isfinite(self.calibration_tolerance) or self.calibration_tolerance < 0:
            raise InvalidDepthCameraConfiguration()
        if not self.tf_from or (
            self.lookup_viewpoint and self.static_camera_transform_enabled
        ):
            raise InvalidDepthCameraConfiguration()
        if (
            self.static_camera_transform_enabled
            and self.static_camera_frame != self.tf_from
        ):
            raise InvalidDepthCameraConfiguration()
