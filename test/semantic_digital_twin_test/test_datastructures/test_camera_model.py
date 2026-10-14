import numpy as np
import pytest

from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortion,
    CameraDistortionModel,
    CameraRange,
    EquidistantCameraDistortion,
    NoCameraDistortion,
    PlumbBobCameraDistortion,
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
    assert isinstance(model.distortion, NoCameraDistortion)


def test_field_of_view_camera_model_rejects_zero_angle():
    with pytest.raises(InvalidCameraFieldOfViewError):
        FieldOfViewCameraModel(
            view=FieldOfView(horizontal_angle=0.0, vertical_angle=np.radians(60.0))
        )


def test_field_of_view_camera_model_has_default_resolution():
    model = FieldOfViewCameraModel(view=FieldOfView())

    assert model.resolution == CameraResolution()


def test_field_of_view_camera_model_preserves_explicit_resolution():
    resolution = CameraResolution(width=320, height=240)

    model = FieldOfViewCameraModel(
        view=FieldOfView(),
        image_resolution=resolution,
    )

    assert model.resolution is resolution


def test_pinhole_camera_model_rejects_nonpositive_focal_length():
    with pytest.raises(InvalidPinholeCameraModelError):
        PinholeCameraModel(
            image_resolution=CameraResolution(width=640, height=480),
            focal_length_x=0.0,
            focal_length_y=400.0,
            principal_point_x=320.0,
            principal_point_y=240.0,
        )


def test_pinhole_camera_model_exposes_intrinsic_matrix():
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point_x=321.0,
        principal_point_y=239.0,
    )

    assert np.array_equal(
        model.intrinsic_matrix,
        np.array(
            [
                [model.focal_length_x, 0.0, model.principal_point_x],
                [0.0, model.focal_length_y, model.principal_point_y],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_pinhole_camera_model_scales_projection_and_preserves_distortion():
    distortion = PlumbBobCameraDistortion(radial_coefficient_1=0.1)
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point_x=321.0,
        principal_point_y=239.0,
        distortion=distortion,
    )

    scaled_model = model.scaled(scale_x=0.5, scale_y=0.25)

    assert scaled_model.resolution == CameraResolution(width=320, height=120)
    assert scaled_model.focal_length_x == model.focal_length_x * 0.5
    assert scaled_model.focal_length_y == model.focal_length_y * 0.25
    assert scaled_model.principal_point_x == model.principal_point_x * 0.5
    assert scaled_model.principal_point_y == model.principal_point_y * 0.25
    assert scaled_model.distortion is distortion


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
        CameraDistortion.from_ordered_coefficients(
            model=model,
            coefficients=coefficients,
        )


def test_plumb_bob_distortion_names_ordered_coefficients():
    coefficients = (0.1, -0.2, 0.01, 0.02, 0.03)

    distortion = CameraDistortion.from_ordered_coefficients(
        model=CameraDistortionModel.PLUMB_BOB,
        coefficients=coefficients,
    )

    assert isinstance(distortion, PlumbBobCameraDistortion)
    assert distortion.radial_coefficient_1 == coefficients[0]
    assert distortion.radial_coefficient_2 == coefficients[1]
    assert distortion.tangential_coefficient_1 == coefficients[2]
    assert distortion.tangential_coefficient_2 == coefficients[3]
    assert distortion.radial_coefficient_3 == coefficients[4]
    assert distortion.to_ordered_coefficients() == coefficients


def test_equidistant_distortion_names_ordered_coefficients():
    coefficients = (0.1, -0.2, 0.3, -0.4)

    distortion = CameraDistortion.from_ordered_coefficients(
        model=CameraDistortionModel.EQUIDISTANT,
        coefficients=coefficients,
    )

    assert isinstance(distortion, EquidistantCameraDistortion)
    assert distortion.radial_coefficient_1 == coefficients[0]
    assert distortion.radial_coefficient_2 == coefficients[1]
    assert distortion.radial_coefficient_3 == coefficients[2]
    assert distortion.radial_coefficient_4 == coefficients[3]
    assert distortion.to_ordered_coefficients() == coefficients


@pytest.mark.parametrize("coefficient", [np.nan, np.inf, -np.inf])
def test_camera_distortion_rejects_nonfinite_coefficient(coefficient: float):
    with pytest.raises(InvalidCameraDistortionError):
        PlumbBobCameraDistortion(radial_coefficient_1=coefficient)
