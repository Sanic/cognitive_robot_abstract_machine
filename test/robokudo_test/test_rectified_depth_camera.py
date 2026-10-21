from __future__ import annotations

# %% imports

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pytest
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry
from robokudo.io.exceptions import InvalidRectifiedDepthFrame
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

# %% synchronized input


@dataclass
class AlignedImages:
    """
    A rectified color image and its inferred depth and calibration.
    """

    color: Image
    """Left rectified image."""
    depth: Image
    """
    Depth in meters on the left pixel grid.
    """

    calibration: CameraInfo
    """Calibration with distinct raw and rectified intrinsics."""


@pytest.fixture
def aligned_images(node: Node) -> AlignedImages:
    """
    Provide aligned messages with invalid pixels and nontrivial RGB colors.
    """
    bridge = CVBridgeWorkaround()
    color = bridge.cv2_to_imgmsg(
        np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8), "rgb8"
    )
    depth = bridge.cv2_to_imgmsg(
        np.array([[1.25, np.nan, np.inf], [0.0, -1.0, 2.5]], dtype=np.float32),
        "32FC1",
    )
    color.header.stamp = node.get_clock().now().to_msg()
    color.header.frame_id = FoundationStereoCameraConfig().tf_from
    depth.header = deepcopy(color.header)
    calibration = CameraInfo(
        header=deepcopy(color.header),
        width=color.width,
        height=color.height,
        k=[100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0],
        p=[200.0, 0.0, 1.0, 0.0, 0.0, 210.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    return AlignedImages(color, depth, calibration)


@pytest.fixture
def camera(node: Node) -> RectifiedDepthCameraInterface:
    """
    Provide a camera whose message filters can be driven without publishers.
    """
    return RectifiedDepthCameraInterface(
        FoundationStereoCameraConfig(lookup_viewpoint=False), node=node
    )


# %% depth and projection


def test_metric_depth_becomes_native_float_millimeters(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Preserve valid depth precision and remove invalid pixels at the boundary.
    """
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    cas = CAS()
    camera.set_data(cas)
    depth = cas.get(CASViews.DEPTH_IMAGE)
    np.testing.assert_array_equal(depth, [[1250.0, 0.0, 0.0], [0.0, 0.0, 2500.0]])
    assert depth.dtype == np.dtype(np.float32)
    assert depth.flags.c_contiguous
    assert depth.dtype.isnative
    np.testing.assert_array_equal(cas.get(CASViews.COLOR_IMAGE)[0, 0], [30, 20, 10])


def test_rectified_projection_supplies_intrinsics(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Use rectified projection rather than the raw fisheye calibration.
    """
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    cas = CAS()
    camera.set_data(cas)
    expected = np.asarray(aligned_images.calibration.p).reshape(3, 4)[:, :3]
    np.testing.assert_array_equal(
        cas.get(CASViews.CAMERA_INTRINSIC).intrinsic_matrix, expected
    )
    np.testing.assert_array_equal(
        np.asarray(cas.get(CASViews.CAMERA_INFO).k).reshape(3, 3), expected
    )
    assert cas.get(CASViews.COLOR_TO_DEPTH_RATIO).is_identity


def test_generated_cloud_has_metric_scale(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Depth normalized for RK reconstructs the original meter distances.
    """
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    cas = CAS()
    camera.set_data(cas)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cas.get(CASViews.COLOR_IMAGE)),
        o3d.geometry.Image(cas.get(CASViews.DEPTH_IMAGE)),
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, cas.get(CASViews.CAMERA_INTRINSIC)
    )
    np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], [1.25, 2.5])


@pytest.mark.parametrize(
    "invalid_part", ["frame", "size", "encoding", "projection", "roi"]
)
def test_incompatible_geometry_is_rejected(
    camera: RectifiedDepthCameraInterface,
    aligned_images: AlignedImages,
    invalid_part: str,
) -> None:
    """
    Reject inputs that cannot describe the same rectified pinhole image.
    """
    if invalid_part == "frame":
        aligned_images.depth.header.frame_id += "_other"
    elif invalid_part == "size":
        aligned_images.calibration.width += 1
    elif invalid_part == "encoding":
        aligned_images.depth.encoding = "16UC1"
    elif invalid_part == "projection":
        aligned_images.calibration.p[0] = 0.0
    else:
        aligned_images.calibration.roi.width = 1
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    with pytest.raises(InvalidRectifiedDepthFrame):
        camera.set_data(CAS())


# %% acquisition matching


def test_delayed_depth_matches_original_color(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Later color arrivals must not replace the image used to infer depth.
    """
    camera.color_subscriber.signalMessage(aligned_images.color)
    newer_color = deepcopy(aligned_images.color)
    newer_color.header.stamp = Time(
        nanoseconds=Time.from_msg(newer_color.header.stamp).nanoseconds + 10_000_000
    ).to_msg()
    camera.color_subscriber.signalMessage(newer_color)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    camera.depth_subscriber.signalMessage(aligned_images.depth)
    cas = CAS()
    camera.set_data(cas)
    assert (
        cas.data_timestamp
        == Time.from_msg(aligned_images.color.header.stamp).nanoseconds
    )


def test_different_acquisitions_do_not_match(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Even a small RGB/depth timestamp difference prevents a match.
    """
    aligned_images.depth.header.stamp = Time(
        nanoseconds=Time.from_msg(aligned_images.color.header.stamp).nanoseconds + 1
    ).to_msg()
    camera.color_subscriber.signalMessage(aligned_images.color)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    camera.depth_subscriber.signalMessage(aligned_images.depth)
    assert camera.has_new_data() is False


def test_calibration_can_have_a_nearby_timestamp(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Calibration tolerance must not weaken exact RGB/depth matching.
    """
    aligned_images.calibration.header.stamp = Time(
        nanoseconds=Time.from_msg(aligned_images.color.header.stamp).nanoseconds + 1
    ).to_msg()
    camera.color_subscriber.signalMessage(aligned_images.color)
    camera.depth_subscriber.signalMessage(aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    assert camera.has_new_data() is True


def test_stale_frame_is_dropped_before_consumption(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    A frame can expire while waiting for the RK pipeline to consume it.
    """
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    camera.camera_config.max_frame_age = 1e-9
    cas = CAS()
    camera.set_data(cas)
    assert cas.data_timestamp is None
    assert camera.statistics.stale_frames == 1
    assert camera.has_new_data() is False


def test_transform_lookup_uses_acquisition_time(
    camera: RectifiedDepthCameraInterface,
    aligned_images: AlignedImages,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Delayed depth must use the pose at exposure, rather than the latest pose.
    """
    timestamps: list[Time] = []

    def record_lookup(timestamp: Time) -> bool:
        """
        Record the requested acquisition and report unavailable TF.
        """
        timestamps.append(timestamp)
        return False

    monkeypatch.setattr(camera, "lookup_transform", record_lookup)
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    cas = CAS()
    camera.set_data(cas)
    assert timestamps == [Time.from_msg(aligned_images.color.header.stamp)]
    assert cas.data_timestamp is None
    assert camera.statistics.missing_transforms == 1


# %% configuration


def test_foundation_stereo_config_is_discovered() -> None:
    """
    Expose the new source through RK's camera configuration registry.
    """
    config = CameraConfigRegistry.create_config(
        FoundationStereoCameraConfig.registry_name
    )
    assert type(config) is FoundationStereoCameraConfig


def test_stream_qos_is_configurable(node: Node) -> None:
    """
    Reliable depth and best-effort camera streams can coexist.
    """
    config = FoundationStereoCameraConfig(lookup_viewpoint=False)
    camera = RectifiedDepthCameraInterface(config, node=node)
    assert (
        camera.color_subscriber.sub.qos_profile.reliability == config.color_reliability
    )
    assert (
        camera.camera_info_subscriber.sub.qos_profile.reliability
        == config.calibration_reliability
    )
    assert (
        camera.depth_subscriber.sub.qos_profile.reliability
        == ReliabilityPolicy.RELIABLE
    )
