from __future__ import annotations

# %% imports

from dataclasses import dataclass

from rclpy.qos import ReliabilityPolicy

from krrood.ormatic.utils import classproperty
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    DepthCameraSource,
    FoundationStereoCameraConfig,
)

# %% Olive rectified depth


@dataclass(kw_only=True)
class OliveCameraConfig(FoundationStereoCameraConfig):
    """
    Read Olive rectified color and disparity-derived metric depth.
    """

    topic_color: str = "/left/image_rect_color"
    """
    Rectified left color image topic.
    """

    topic_depth: str = "/depth/image"
    """
    Float32 meter depth aligned to the left rectified image.
    """

    topic_camera_info: str = "/left/camera_info"
    """
    Per-frame left stereo calibration topic.
    """

    tf_from: str = "olive_left_optical_frame"
    """
    Optical frame shared by the left image, calibration and depth.
    """

    lookup_viewpoint: bool = False
    """
    Whether to require acquisition-time TF before consuming a frame.
    """

    color_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for rectified color images.
    """

    calibration_reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    """
    Requested reliability for per-frame calibration.
    """

    calibration_tolerance: float = 0.0
    """
    Maximum calibration timestamp difference in seconds.
    """

    @classproperty
    def registry_name(cls: type[OliveCameraConfig]) -> str:
        return DepthCameraSource.OLIVE
