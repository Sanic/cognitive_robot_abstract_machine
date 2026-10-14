from __future__ import annotations

# %% Imports
import numpy as np

from robokudo.descriptors.worlds.pinhole_rgbd_camera import (
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
from semantic_digital_twin.world import World
from robokudo.world_descriptor import BaseWorldDescriptor
from ..semantic_digital_twin_test.conftest import table_world

# %% Standard configuration


def test_pinhole_rgbd_camera_spec_uses_standard_imaging_configuration() -> None:
    pose = HomogeneousTransformationMatrix()

    camera_spec = PinholeRGBDCameraSpec(pose=pose)

    expected_camera_model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(),
        field_of_view=FieldOfView(),
    )
    assert camera_spec.name == PinholeRGBDCameraSpec.name
    assert camera_spec.pose is pose
    np.testing.assert_array_equal(camera_spec.forward_facing_axis, Vector3.Z())
    assert camera_spec.camera_model == expected_camera_model
    assert camera_spec.camera_range == CameraRange(
        minimum_distance=0.05,
        maximum_distance=8.0,
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


# %% Configurable dataclass fields


def test_pinhole_rgbd_camera_spec_accepts_camera_field_overrides() -> None:
    """
    Dataclass construction retains caller-supplied camera settings.
    """
    pose = HomogeneousTransformationMatrix()
    camera_model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(), field_of_view=FieldOfView()
    )
    camera_range = CameraRange(minimum_distance=0.1, maximum_distance=2.0)
    forward_facing_axis = Vector3.X()
    modalities = (CameraModality.COLOR,)
    camera_spec = PinholeRGBDCameraSpec(
        pose=pose,
        camera_model=camera_model,
        camera_range=camera_range,
        forward_facing_axis=forward_facing_axis,
        modalities=modalities,
    )
    assert camera_spec.pose is pose
    assert camera_spec.camera_model is camera_model
    assert camera_spec.camera_range is camera_range
    assert camera_spec.forward_facing_axis is forward_facing_axis
    assert camera_spec.modalities == modalities


# %% Camera identity


def test_pinhole_rgbd_camera_spec_accepts_arbitrary_frame_name() -> None:
    """
    Callers can select a frame name outside the default camera identity.
    """
    frame_name = "custom_camera/optical_frame"
    camera_spec = PinholeRGBDCameraSpec(
        pose=HomogeneousTransformationMatrix(), name=frame_name
    )
    assert camera_spec.name == frame_name


# %% Root-relative optical convention


def test_camera_spec_viewing_axis_selects_fixed_optical_convention(
    table_world: World,
) -> None:
    """
    A caller-selected viewing axis determines the built camera's optical direction.
    """
    descriptor = BaseWorldDescriptor(world=table_world)
    spec = PinholeRGBDCameraSpec(
        pose=HomogeneousTransformationMatrix(reference_frame=table_world.root),
        forward_facing_axis=Vector3.X(),
    )
    camera = descriptor.build_camera(table_world.root, spec)
    np.testing.assert_array_equal(
        camera.viewing_direction.to_np(), spec.forward_facing_axis.to_np()
    )
