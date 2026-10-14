"""
Camera pose and pixel orientation across rendering conventions.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import Mock

import numpy as np
import pytest

from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.camera_rendering import RayIntersections
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2KinectV1
from semantic_digital_twin.robots.tiago import TiagoCamera
from semantic_digital_twin.robots.stretch import (
    StretchCameraColor,
    StretchCameraDepth,
    StretchCameraInfra1,
    StretchCameraInfra2,
)
from semantic_digital_twin.robots.robot_parts import Camera, RobotCamera
from semantic_digital_twin.robots.exceptions import CameraOpticalConventionUndefined
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.testing import ray_test_world
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


# %% Optical camera fixture
@pytest.fixture
def optical_camera(
    ray_test_world: tuple[World, Body, Body, Body, Body],
) -> Camera:
    """
    Provide a camera with a rectangular projection and a nonidentity optical pose.
    """
    world, body, *_ = ray_test_world
    return Camera(
        name=PrefixedName(name="optical_camera"),
        root=body,
        forward_facing_axis=Vector3.Z(),
        camera_model=PinholeCameraModel.from_field_of_view(
            resolution=CameraResolution(width=6, height=4), field_of_view=FieldOfView()
        ),
        root_T_optical=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, y=0.1, z=0.3, roll=0.4, pitch=0.2, yaw=np.pi / 2
        ),
    )


# %% Optical pose composition


def test_optical_pose_preserves_root_pose_and_camera_offset(
    optical_camera: Camera,
) -> None:
    """
    The optical pose composes the complete root pose and camera offset.
    """
    np.testing.assert_allclose(
        optical_camera.world_T_optical.to_np(),
        optical_camera.root.global_transform.to_np()
        @ optical_camera.root_T_optical.to_np(),
    )


@pytest.mark.parametrize("optical_roll", [0.0, np.pi / 2])
@pytest.mark.parametrize("root_pitch", [0.0, np.pi / 2])
def test_render_preserves_optical_orientation(
    optical_camera: Camera,
    monkeypatch: pytest.MonkeyPatch,
    optical_roll: float,
    root_pitch: float,
) -> None:
    """
    Rendering preserves the full optical orientation, including roll.
    """
    optical_camera.root_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(
        roll=optical_roll, pitch=root_pitch
    )
    ray_tracer = optical_camera.root._world.ray_tracer
    query = Mock(
        return_value=RayIntersections(
            hit_points=np.empty((0, 3)), ray_indices=np.empty(0, dtype=int), bodies=[]
        )
    )
    monkeypatch.setattr(ray_tracer, "ray_test", query)
    ray_tracer.render(optical_camera)
    np.testing.assert_allclose(
        ray_tracer.scene.camera_transform,
        optical_camera.world_T_optical.to_np() @ np.diag([1.0, -1.0, -1.0, 1.0]),
        atol=1e-12,
    )
    query.assert_called_once()


def test_render_places_top_left_optical_ray_at_top_left_pixel(
    optical_camera: Camera, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A hit on the optical top-left ray occupies the top-left image pixel.
    """
    ray_tracer = optical_camera.root._world.ray_tracer
    model = optical_camera.camera_model
    optical_V_top_left = np.array(
        [
            (0.5 - model.principal_point_x) / model.focal_length_x,
            (0.5 - model.principal_point_y) / model.focal_length_y,
            1.0,
        ]
    )
    optical_V_top_left /= np.linalg.norm(optical_V_top_left)
    world_T_optical = optical_camera.world_T_optical.to_np()
    world_V_top_left = world_T_optical[:3, :3] @ optical_V_top_left

    def intersect_top_left(
        origin_points: np.ndarray,
        target_points: np.ndarray,
        multiple_hits: bool,
        min_distance: float,
        max_distance: float,
    ) -> RayIntersections:
        """
        Return a hit only on the ray matching the optical top-left pixel center.
        """
        directions = target_points - origin_points
        distances = np.linalg.norm(directions - world_V_top_left, axis=1)
        ray_index = int(np.argmin(distances))
        np.testing.assert_allclose(directions[ray_index], world_V_top_left, atol=1e-12)
        return RayIntersections(
            hit_points=origin_points[[ray_index]] + directions[[ray_index]],
            ray_indices=np.array([ray_index]),
            bodies=[optical_camera.root],
        )

    monkeypatch.setattr(ray_tracer, "ray_test", intersect_top_left)
    frame = ray_tracer.render(optical_camera)
    expected = np.full(
        (model.resolution.width, model.resolution.height), -1, dtype=np.int32
    )
    expected[0, 0] = optical_camera.root.index
    np.testing.assert_array_equal(frame.segmentation, expected)


# %% Camera-relative image axes


@pytest.mark.parametrize(
    "root_pose",
    [
        HomogeneousTransformationMatrix.from_xyz_rpy(roll=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(pitch=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.8, y=1.4, z=0.9, roll=0.4, pitch=0.7, yaw=0.3
        ),
    ],
)
def test_forward_view_preserves_fixed_optical_attachment_under_robot_motion(
    optical_camera: Camera,
    root_pose: HomogeneousTransformationMatrix,
) -> None:
    """
    Camera motion changes the world pose without changing image axes relative to its
    root.
    """
    root = optical_camera.root
    world = root._world
    with world.modify_world():
        root.parent_connection.origin = HomogeneousTransformationMatrix(
            data=root_pose,
            reference_frame=root.parent_connection.parent,
            child_frame=root,
        )
    np.testing.assert_allclose(
        optical_camera.root_T_forward_view.to_np(),
        root.global_transform.to_np()
        @ optical_camera.root_T_optical.to_np()
        @ Camera.optical_R_forward_view.to_np(),
        atol=1e-12,
    )


# %% Robot optical roots
@pytest.mark.parametrize(
    "camera_type",
    [
        PR2KinectV1,
        TiagoCamera,
        StretchCameraColor,
        StretchCameraDepth,
        StretchCameraInfra1,
        StretchCameraInfra2,
    ],
)
def test_default_optical_camera_has_identity_attachment(
    camera_type: type[RobotCamera],
    table_world: World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Cameras rooted at optical bodies explicitly retain their model's image axes.
    """
    monkeypatch.setattr(
        table_world, "get_body_in_branch_by_name", Mock(return_value=table_world.root)
    )
    camera = camera_type.setup_default_configuration_in_world_below_robot_root(
        table_world.root
    )
    np.testing.assert_array_equal(camera.root_T_optical.to_np(), np.eye(4))


# %% Default fixed conventions
@pytest.mark.parametrize("viewing_axis", [Vector3.X(), Vector3.Z()])
@pytest.mark.parametrize(
    "root_pose",
    [
        HomogeneousTransformationMatrix.from_xyz_rpy(roll=np.pi / 2),
        HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 2),
    ],
)
def test_default_optical_attachment_stays_fixed_under_robot_motion(
    optical_camera: Camera,
    viewing_axis: Vector3,
    root_pose: HomogeneousTransformationMatrix,
) -> None:
    """
    Default image axes remain fixed relative to the root as the robot moves.
    """
    camera = Camera(
        name=optical_camera.name,
        root=optical_camera.root,
        forward_facing_axis=viewing_axis,
        camera_model=optical_camera.camera_model,
    )
    root_T_optical_before = (
        camera.root.global_transform.inverse() @ camera.world_T_optical
    )
    with camera.root._world.modify_world():
        camera.root.parent_connection.origin = HomogeneousTransformationMatrix(
            data=root_pose,
            reference_frame=camera.root.parent_connection.parent,
            child_frame=camera.root,
        )
    root_T_optical_after = (
        camera.root.global_transform.inverse() @ camera.world_T_optical
    )
    np.testing.assert_allclose(
        root_T_optical_after.to_np(), root_T_optical_before.to_np(), atol=1e-12
    )


@pytest.mark.parametrize("viewing_axis", [Vector3.X(), Vector3.Z()])
def test_default_optical_axes_follow_fixed_root_convention(
    optical_camera: Camera,
    viewing_axis: Vector3,
) -> None:
    """
    Positive z uses optical axes; positive x uses right-minus-y and down-minus-z axes.
    """
    camera = Camera(
        name=optical_camera.name,
        root=optical_camera.root,
        forward_facing_axis=viewing_axis,
        camera_model=optical_camera.camera_model,
    )
    if np.array_equal(viewing_axis.to_np(), Vector3.Z().to_np()):
        expected = np.eye(4)
    else:
        expected = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    root_T_optical = camera.root.global_transform.inverse() @ camera.world_T_optical
    np.testing.assert_allclose(root_T_optical.to_np(), expected, atol=1e-12)


# %% Explicit orientation for other viewing axes


def test_other_viewing_axis_requires_optical_attachment(optical_camera: Camera) -> None:
    """
    An axis without a standard convention requires explicit image right and down axes.
    """
    with pytest.raises(CameraOpticalConventionUndefined):
        Camera(
            name=optical_camera.name,
            root=optical_camera.root,
            forward_facing_axis=Vector3.Y(),
            camera_model=optical_camera.camera_model,
        )


def test_other_viewing_axis_accepts_explicit_optical_attachment(
    optical_camera: Camera,
) -> None:
    """
    A supplied optical attachment supports arbitrary root-relative viewing directions.
    """
    root_T_optical = HomogeneousTransformationMatrix.from_xyz_rpy(roll=-np.pi / 2)
    camera = Camera(
        name=optical_camera.name,
        root=optical_camera.root,
        forward_facing_axis=Vector3.Y(),
        camera_model=optical_camera.camera_model,
        root_T_optical=root_T_optical,
    )
    np.testing.assert_allclose(
        camera.viewing_direction.to_np(), Vector3.Y().to_np(), atol=1e-12
    )
