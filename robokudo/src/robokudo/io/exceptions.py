from __future__ import annotations

# %% camera contract errors

from dataclasses import dataclass
from enum import StrEnum

from krrood.exceptions import DataclassException


class DepthFrameMismatch(StrEnum):
    """
    Input contracts violated by a rectified depth acquisition.
    """

    ACQUISITION = "different image acquisition timestamps"
    OPTICAL_FRAME = "different optical frames"
    DIMENSIONS = "different image dimensions"
    DEPTH_ENCODING = "depth is not float32 meters"
    IMAGE_REGION = "ROI or binning is unsupported"
    FOCAL_LENGTH = "invalid rectified focal lengths"
    PROJECTION = "unsupported left projection geometry"


@dataclass
class InvalidDepthCameraConfiguration(DataclassException):
    """
    A rectified depth camera has inconsistent configuration.
    """

    def error_message(self) -> str:
        return "The rectified depth camera configuration is inconsistent."

    def suggest_correction(self) -> str:
        return (
            "Use positive queue sizes, age and depth limits, a nonnegative calibration "
            "tolerance, and a single transform source with matching optical frames."
        )


@dataclass
class InvalidRectifiedDepthFrame(DataclassException):
    """
    Images and calibration do not describe one rectified optical image.
    """

    reason: DepthFrameMismatch
    """
    The violated input contract.
    """

    def error_message(self) -> str:
        return f"Invalid rectified depth frame: {self.reason}."

    def suggest_correction(self) -> str:
        return (
            "Supply aligned raw color and 32FC1 meter depth with identical acquisition "
            "stamps, matching optical frames and full-size left rectified projection."
        )
