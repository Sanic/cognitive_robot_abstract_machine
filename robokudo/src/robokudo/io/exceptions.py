"""
Errors raised by RoboKudo camera interfaces.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from robokudo.exceptions import InvalidCameraObservation, RoboKudoError
from robokudo.types.image import ImageRotation
from semantic_digital_twin.datastructures.camera_model import CameraDistortionModel


# %% Image rotation
@dataclass
class UnsupportedImageRotation(RoboKudoError, ValueError):
    """
    A requested image rotation is outside the supported orientations.
    """

    rotation: str | None
    """
    Unsupported rotation supplied to the camera interface.
    """

    def error_message(self) -> str:
        """
        Identify the unsupported rotation.
        """
        return f"Unsupported image rotation: {self.rotation!r}."

    def suggest_correction(self) -> str:
        """
        List the supported rotation choices.
        """
        choices = ", ".join(rotation.value for rotation in ImageRotation)
        return f"Use one of the supported image rotations: {choices}."


# %% Camera distortion metadata
@dataclass
class CameraDistortionModelMissing(InvalidCameraObservation):
    """
    Nonzero distortion coefficients have no declared distortion model.
    """

    reason: str = field(init=False)
    """
    Description of the missing model and its coefficients.
    """

    coefficients: tuple[float, ...]
    """
    Distortion coefficients supplied without a model.
    """

    def __post_init__(self) -> None:
        """
        Describe the coefficients that require a distortion model.
        """
        self.reason = f"nonzero distortion coefficients {self.coefficients!r} have no distortion model"

    def suggest_correction(self) -> str:
        """
        Identify the missing calibration metadata.
        """
        return "Declare the distortion model matching the supplied camera coefficients."


@dataclass
class UnsupportedCameraDistortionModel(InvalidCameraObservation):
    """
    Camera calibration declares a distortion model the adapter cannot interpret.
    """

    reason: str = field(init=False)
    """
    Description of the unsupported model.
    """

    distortion_model: str
    """
    Unrecognized distortion model supplied by the camera.
    """

    def __post_init__(self) -> None:
        """
        Describe the unrecognized calibration model.
        """
        self.reason = f"unsupported camera distortion model: {self.distortion_model!r}"

    def suggest_correction(self) -> str:
        """
        List the models supported at the camera-info boundary.
        """
        choices = ", ".join(
            model.value
            for model in CameraDistortionModel
            if model is not CameraDistortionModel.NONE
        )
        return f"Supply calibration using one of the supported distortion models: {choices}."
