from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Optional

from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.exceptions import (
    InvalidCameraDistortionError,
    InvalidCameraFieldOfViewError,
    InvalidCameraRangeError,
    InvalidPinholeCameraModelError,
)

# %% Projection models


@dataclass
class CameraModel(ABC):
    """Describe how a camera maps viewing rays to an image."""

    @property
    @abstractmethod
    def field_of_view(self) -> FieldOfView:
        """Return the angular extent represented by the model."""

    @property
    @abstractmethod
    def resolution(self) -> CameraResolution:
        """Return the image resolution represented by the model."""


@dataclass
class FieldOfViewCameraModel(CameraModel):
    """Describe an ideal camera from its angular extent and image resolution."""

    view: FieldOfView = field(kw_only=True)
    """Angular extent of the camera image."""

    image_resolution: CameraResolution = field(
        default_factory=CameraResolution, kw_only=True
    )
    """Pixel dimensions of images produced by the camera."""

    def __post_init__(self) -> None:
        """Validate that both viewing angles describe a physical camera."""
        _validate_field_of_view(self.view)

    @property
    def field_of_view(self) -> FieldOfView:
        """Return the declared angular extent."""
        return self.view

    @property
    def resolution(self) -> CameraResolution:
        """Return the image resolution represented by the model."""
        return self.image_resolution


@dataclass
class PinholeCameraModel(CameraModel):
    """Describe a calibrated pinhole projection for one image stream."""

    image_resolution: CameraResolution = field(kw_only=True)
    """Pixel dimensions of images produced with this calibration."""

    focal_length_x: float = field(kw_only=True)
    """Horizontal focal length in pixels."""

    focal_length_y: float = field(kw_only=True)
    """Vertical focal length in pixels."""

    principal_point_x: float = field(kw_only=True)
    """Horizontal principal point in pixels from the image's left edge."""

    principal_point_y: float = field(kw_only=True)
    """Vertical principal point in pixels from the image's top edge."""

    distortion: CameraDistortion = field(default_factory=lambda: CameraDistortion())
    """Lens-distortion parameters associated with the calibration."""

    def __post_init__(self) -> None:
        """Validate the focal lengths and principal point."""
        if self.focal_length_x <= 0.0 or self.focal_length_y <= 0.0:
            raise InvalidPinholeCameraModelError(
                reason="focal lengths must be positive"
            )
        if not 0.0 <= self.principal_point_x <= self.image_resolution.width:
            raise InvalidPinholeCameraModelError(
                reason="the horizontal principal point must lie within the image"
            )
        if not 0.0 <= self.principal_point_y <= self.image_resolution.height:
            raise InvalidPinholeCameraModelError(
                reason="the vertical principal point must lie within the image"
            )

    @classmethod
    def from_field_of_view(
        cls,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
        distortion: Optional[CameraDistortion] = None,
    ) -> PinholeCameraModel:
        """Create a centered pinhole model for an angular image extent.

        :param resolution: Pixel dimensions of the modeled image.
        :param field_of_view: Horizontal and vertical angular extent.
        :param distortion: Lens distortion associated with the image.
        :return: Centered pinhole calibration matching the requested extent.
        """
        _validate_field_of_view(field_of_view)
        focal_length_x = resolution.width / (
            2.0 * math.tan(field_of_view.horizontal_angle / 2.0)
        )
        focal_length_y = resolution.height / (
            2.0 * math.tan(field_of_view.vertical_angle / 2.0)
        )
        return cls(
            image_resolution=resolution,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            principal_point_x=resolution.width / 2.0,
            principal_point_y=resolution.height / 2.0,
            distortion=distortion if distortion is not None else CameraDistortion(),
        )

    @property
    def resolution(self) -> CameraResolution:
        """Return the image resolution associated with the calibration."""
        return self.image_resolution

    @property
    def field_of_view(self) -> FieldOfView:
        """Calculate the angular extent represented by the calibration."""
        horizontal_angle = math.atan(
            self.principal_point_x / self.focal_length_x
        ) + math.atan(
            (self.image_resolution.width - self.principal_point_x) / self.focal_length_x
        )
        vertical_angle = math.atan(
            self.principal_point_y / self.focal_length_y
        ) + math.atan(
            (self.image_resolution.height - self.principal_point_y)
            / self.focal_length_y
        )
        return FieldOfView(
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle,
        )


# %% Distortion and range


class CameraDistortionModel(StrEnum):
    """Identify the mathematical model used for lens distortion."""

    NONE = "none"
    """The image has no modeled lens distortion."""

    PLUMB_BOB = "plumb_bob"
    """The coefficients follow the ROS plumb-bob convention."""

    RATIONAL_POLYNOMIAL = "rational_polynomial"
    """The coefficients follow the ROS rational-polynomial convention."""

    EQUIDISTANT = "equidistant"
    """The coefficients follow the ROS equidistant convention."""

    @property
    def coefficient_count(self) -> int:
        """Return the number of coefficients defined by this model."""
        return {
            CameraDistortionModel.NONE: 0,
            CameraDistortionModel.PLUMB_BOB: 5,
            CameraDistortionModel.RATIONAL_POLYNOMIAL: 8,
            CameraDistortionModel.EQUIDISTANT: 4,
        }[self]


@dataclass
class CameraDistortion:
    """Describe lens distortion associated with an image calibration."""

    model: CameraDistortionModel = CameraDistortionModel.NONE
    """Mathematical distortion model used by the coefficients."""

    coefficients: tuple[float, ...] = field(default_factory=tuple)
    """Ordered coefficients defined by :attr:`model`."""

    def __post_init__(self) -> None:
        """Validate and store coefficients in their canonical representation."""
        self.coefficients = tuple(self.coefficients)
        expected_coefficient_count = self.model.coefficient_count
        if len(self.coefficients) != expected_coefficient_count:
            raise InvalidCameraDistortionError(
                model=self.model.value,
                coefficient_count=len(self.coefficients),
                expected_coefficient_count=expected_coefficient_count,
            )


@dataclass
class CameraRange:
    """Describe the usable distance interval of a camera."""

    minimum_distance: float = 0.0
    """Nearest usable distance in meters."""

    maximum_distance: float = math.inf
    """Farthest usable distance in meters."""

    def __post_init__(self) -> None:
        """Validate the distance interval."""
        if self.minimum_distance < 0.0:
            raise InvalidCameraRangeError(
                minimum_distance=self.minimum_distance,
                maximum_distance=self.maximum_distance,
            )
        if self.maximum_distance <= self.minimum_distance:
            raise InvalidCameraRangeError(
                minimum_distance=self.minimum_distance,
                maximum_distance=self.maximum_distance,
            )


class CameraModality(StrEnum):
    """Identify a kind of image data produced by a camera."""

    COLOR = "color"
    """Color intensity images."""

    DEPTH = "depth"
    """Per-pixel depth images."""

    INFRARED = "infrared"
    """Infrared intensity images."""

    SEGMENTATION = "segmentation"
    """Per-pixel semantic or instance labels."""


def _validate_field_of_view(field_of_view: FieldOfView) -> None:
    """Validate that both viewing angles lie between zero and pi.

    :param field_of_view: Angular extent to validate.
    :raises InvalidCameraFieldOfViewError: If either angle is outside the valid range.
    """
    if not 0.0 < field_of_view.horizontal_angle < math.pi:
        raise InvalidCameraFieldOfViewError(
            horizontal_angle=field_of_view.horizontal_angle,
            vertical_angle=field_of_view.vertical_angle,
        )
    if not 0.0 < field_of_view.vertical_angle < math.pi:
        raise InvalidCameraFieldOfViewError(
            horizontal_angle=field_of_view.horizontal_angle,
            vertical_angle=field_of_view.vertical_angle,
        )
