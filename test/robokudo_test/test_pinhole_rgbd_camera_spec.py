import numpy as np

from robokudo.descriptors.worlds.pinhole_rgbd_camera import (
    PinholeRGBDCameraName,
    PinholeRGBDCameraSpec,
)
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    CameraRange,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3

# %% Standard configuration


def test_pinhole_rgbd_camera_spec_uses_standard_imaging_configuration() -> None:
    pose = HomogeneousTransformationMatrix()

    camera_spec = PinholeRGBDCameraSpec(pose=pose)

    expected_camera_model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(),
        field_of_view=FieldOfView(),
    )
    assert camera_spec.name == PinholeRGBDCameraName.DEFAULT
    assert camera_spec.pose is pose
    np.testing.assert_array_equal(camera_spec.forward_facing_axis, Vector3.Z())
    assert camera_spec.camera_model == expected_camera_model
    assert camera_spec.camera_range == CameraRange(
        minimum_distance=PinholeRGBDCameraSpec.MINIMUM_DISTANCE,
        maximum_distance=PinholeRGBDCameraSpec.MAXIMUM_DISTANCE,
    )
    assert camera_spec.modalities == (
        CameraModality.COLOR,
        CameraModality.DEPTH,
        CameraModality.SEGMENTATION,
    )


def test_pinhole_rgbd_camera_specs_have_independent_camera_data() -> None:
    first_spec = PinholeRGBDCameraSpec(pose=HomogeneousTransformationMatrix())
    second_spec = PinholeRGBDCameraSpec(pose=HomogeneousTransformationMatrix())

    assert first_spec.camera_model is not second_spec.camera_model
    assert first_spec.camera_range is not second_spec.camera_range
