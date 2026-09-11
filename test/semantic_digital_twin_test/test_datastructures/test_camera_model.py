import numpy as np
import pytest

from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortion,
    CameraDistortionModel,
    CameraRange,
    FieldOfViewCameraModel,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.exceptions import (
    InvalidCameraDistortionError,
    InvalidCameraFieldOfViewError,
    InvalidCameraRangeError,
    InvalidPinholeCameraModelError,
)


def test_pinhole_camera_model_from_field_of_view_preserves_projection():
    resolution = CameraResolution(width=640, height=480)
    field_of_view = FieldOfView(
        horizontal_angle=np.radians(90.0),
        vertical_angle=np.radians(60.0),
    )

    model = PinholeCameraModel.from_field_of_view(
        resolution=resolution,
        field_of_view=field_of_view,
    )

    assert model.resolution == resolution
    assert np.isclose(
        model.field_of_view.horizontal_angle,
        field_of_view.horizontal_angle,
    )
    assert np.isclose(
        model.field_of_view.vertical_angle,
        field_of_view.vertical_angle,
    )


def test_field_of_view_camera_model_rejects_zero_angle():
    with pytest.raises(InvalidCameraFieldOfViewError):
        FieldOfViewCameraModel(
            view=FieldOfView(horizontal_angle=0.0, vertical_angle=np.radians(60.0))
        )


def test_pinhole_camera_model_rejects_nonpositive_focal_length():
    with pytest.raises(InvalidPinholeCameraModelError):
        PinholeCameraModel(
            image_resolution=CameraResolution(width=640, height=480),
            focal_length_x=0.0,
            focal_length_y=400.0,
            principal_point_x=320.0,
            principal_point_y=240.0,
        )


def test_camera_range_rejects_empty_interval():
    with pytest.raises(InvalidCameraRangeError):
        CameraRange(minimum_distance=1.0, maximum_distance=1.0)


@pytest.mark.parametrize(
    "model, coefficients",
    [
        (CameraDistortionModel.NONE, (0.0,)),
        (CameraDistortionModel.PLUMB_BOB, (0.0,) * 4),
        (CameraDistortionModel.RATIONAL_POLYNOMIAL, (0.0,) * 7),
        (CameraDistortionModel.EQUIDISTANT, (0.0,) * 3),
    ],
)
def test_camera_distortion_rejects_wrong_coefficient_count(
    model: CameraDistortionModel, coefficients: tuple[float, ...]
) -> None:
    with pytest.raises(InvalidCameraDistortionError):
        CameraDistortion(model=model, coefficients=coefficients)
