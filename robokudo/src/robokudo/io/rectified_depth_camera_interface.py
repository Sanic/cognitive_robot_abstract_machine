from __future__ import annotations

# %% imports

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from threading import Lock

import numpy as np
import open3d as o3d
from message_filters import (
    ApproximateTimeSynchronizer,
    SimpleFilter,
    Subscriber,
    TimeSynchronizer,
)
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.io.camera_interface import ROSCameraInterface
from robokudo.io.exceptions import DepthFrameMismatch, InvalidRectifiedDepthFrame
from robokudo.types.cv import ColorToDepthRatio
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

if TYPE_CHECKING:
    import numpy.typing as npt
    from rclpy.node import Node
    from std_msgs.msg import Header

    from robokudo.cas import CAS
    from robokudo.descriptors.camera_configs.config_foundation_stereo import (
        FoundationStereoCameraConfig,
    )


# %% image contracts


class CameraEncoding(StrEnum):
    """Image and calibration encodings at the RK camera boundary."""

    METRIC_DEPTH = "32FC1"
    COLOR = "bgr8"
    UNDISTORTED = "plumb_bob"


class CameraUnit(IntEnum):
    """Conversions between camera transport and RK units."""

    MILLIMETERS_PER_METER = 1000
    NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class ColorDepthImages:
    """A color image and depth inferred from that exact acquisition."""

    color: Image
    """Rectified left color image."""
    depth: Image
    """Aligned metric depth image."""

    @property
    def header(self) -> Header:
        """Acquisition header shared by both images."""
        return self.color.header


@dataclass
class RectifiedDepthFrame:
    """An exact image pair and its associated rectified calibration."""

    images: ColorDepthImages
    """Images sharing an acquisition timestamp."""
    calibration: CameraInfo
    """Calibration describing the rectified left image."""

    def validate(self, optical_frame: str) -> None:
        """Require full-size aligned images in a standard left pinhole frame."""
        color, depth = self.images.color, self.images.depth
        calibration = self.calibration
        if color.header.stamp != depth.header.stamp:
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.ACQUISITION)
        if any(
            message.header.frame_id != optical_frame
            for message in (color, depth, calibration)
        ):
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.OPTICAL_FRAME)
        if (
            color.width <= 0
            or color.height <= 0
            or any(
                message.width != color.width or message.height != color.height
                for message in (depth, calibration)
            )
        ):
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.DIMENSIONS)
        if depth.encoding != CameraEncoding.METRIC_DEPTH:
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.DEPTH_ENCODING)
        if (
            calibration.binning_x not in (0, 1)
            or calibration.binning_y not in (0, 1)
            or any(
                (
                    calibration.roi.x_offset,
                    calibration.roi.y_offset,
                    calibration.roi.width,
                    calibration.roi.height,
                    calibration.roi.do_rectify,
                )
            )
        ):
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.IMAGE_REGION)
        projection = np.asarray(calibration.p).reshape(3, 4)
        if (
            not np.isfinite(projection).all()
            or projection[0, 0] <= 0
            or projection[1, 1] <= 0
        ):
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.FOCAL_LENGTH)
        if not np.allclose(projection[2], [0, 0, 1, 0]) or not np.allclose(
            [projection[0, 1], projection[1, 0], projection[0, 3], projection[1, 3]], 0
        ):
            raise InvalidRectifiedDepthFrame(DepthFrameMismatch.PROJECTION)

    def millimeter_depth(
        self, bridge: CVBridgeWorkaround, max_depth: float
    ) -> npt.NDArray[np.float32]:
        """Return contiguous RK depth with invalid or distant values set to zero."""
        depth = np.empty(
            (self.images.depth.height, self.images.depth.width), dtype=np.float32
        )
        depth[:] = bridge.imgmsg_to_cv2(self.images.depth, CameraEncoding.METRIC_DEPTH)
        depth[~np.isfinite(depth) | (depth <= 0) | (depth > max_depth)] = 0
        depth *= CameraUnit.MILLIMETERS_PER_METER
        return depth

    def rectified_calibration(self) -> CameraInfo:
        """Return camera information describing the undistorted pixel grid."""
        calibration = deepcopy(self.calibration)
        calibration.k = (
            np.asarray(calibration.p).reshape(3, 4)[:, :3].reshape(-1).tolist()
        )
        calibration.d = [0.0] * len(calibration.d)
        calibration.r = np.eye(3).reshape(-1).tolist()
        calibration.distortion_model = CameraEncoding.UNDISTORTED
        return calibration


# %% bounded frame delivery


@dataclass
class DepthCameraStatistics:
    """Counts of synchronized and discarded depth frames."""

    matched_frames: int = 0
    """Image pairs associated with calibration."""
    stale_frames: int = 0
    """Frames outside the permitted acquisition age."""
    replaced_frames: int = 0
    """Pending frames superseded before consumption."""
    missing_transforms: int = 0
    """Frames discarded because acquisition-time TF was unavailable."""
    obsolete_frames: int = 0
    """Repeated or out-of-order acquisitions discarded."""


@dataclass(eq=False)
class RectifiedDepthCameraInterface(ROSCameraInterface):
    """Deliver timestamp-aligned rectified images and meter depth to RK."""

    camera_config: FoundationStereoCameraConfig
    """Topic, synchronization and transform configuration."""
    node: Node | None = None
    """ROS node; the registered RK node is used when omitted."""
    statistics: DepthCameraStatistics = field(
        default_factory=DepthCameraStatistics, init=False
    )
    """Frame acceptance and drop counts."""
    color_subscriber: Subscriber = field(init=False)
    """Raw rectified image subscription."""
    depth_subscriber: Subscriber = field(init=False)
    """Metric depth subscription."""
    camera_info_subscriber: Subscriber = field(init=False)
    """Per-frame calibration subscription."""
    image_synchronizer: TimeSynchronizer = field(init=False)
    """Exact acquisition matching for RGB and depth."""
    calibration_synchronizer: TimeSynchronizer = field(init=False)
    """Calibration association with exact image pairs."""
    image_pairs: SimpleFilter = field(default_factory=SimpleFilter, init=False)
    """Matched image pairs awaiting calibration."""
    pending_frame: RectifiedDepthFrame | None = field(default=None, init=False)
    """Latest frame awaiting pipeline consumption."""
    lock: Lock = field(default_factory=Lock, init=False)
    """Protects pending-frame replacement and consumption."""
    bridge: CVBridgeWorkaround = field(default_factory=CVBridgeWorkaround, init=False)
    """Converts ROS image payloads into arrays."""
    last_timestamp: int = field(default=-1, init=False)
    """Newest acquisition accepted for delivery, in nanoseconds."""

    def __post_init__(self) -> None:
        """Subscribe with bounded histories and acquisition-based matching."""
        super().__init__(self.camera_config, node=self.node)
        configuration = self.camera_config
        self.color_subscriber = Subscriber(
            self.node,
            Image,
            configuration.topic_color,
            qos_profile=QoSProfile(
                depth=configuration.history_depth,
                reliability=configuration.color_reliability,
            ),
        )
        self.depth_subscriber = Subscriber(
            self.node,
            Image,
            configuration.topic_depth,
            qos_profile=QoSProfile(
                depth=configuration.history_depth,
                reliability=configuration.depth_reliability,
            ),
        )
        self.camera_info_subscriber = Subscriber(
            self.node,
            CameraInfo,
            configuration.topic_camera_info,
            qos_profile=QoSProfile(
                depth=configuration.history_depth,
                reliability=configuration.calibration_reliability,
            ),
        )
        self.image_synchronizer = TimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            configuration.synchronization_queue_size,
        )
        self.image_synchronizer.registerCallback(self.receive_images)
        calibration_inputs = [self.image_pairs, self.camera_info_subscriber]
        if configuration.calibration_tolerance == 0:
            self.calibration_synchronizer = TimeSynchronizer(
                calibration_inputs, configuration.synchronization_queue_size
            )
        else:
            self.calibration_synchronizer = ApproximateTimeSynchronizer(
                calibration_inputs,
                configuration.synchronization_queue_size,
                configuration.calibration_tolerance,
            )
        self.calibration_synchronizer.registerCallback(self.receive_frame)

    def receive_images(self, color: Image, depth: Image) -> None:
        """Associate an exact image pair with its per-frame calibration."""
        self.image_pairs.signalMessage(ColorDepthImages(color, depth))

    def receive_frame(self, images: ColorDepthImages, calibration: CameraInfo) -> None:
        """Retain the newest synchronized acquisition for pipeline consumption."""
        timestamp = Time.from_msg(images.header.stamp).nanoseconds
        with self.lock:
            self.statistics.matched_frames += 1
            if not self.is_current(timestamp):
                self.statistics.stale_frames += 1
                return
            if timestamp <= self.last_timestamp:
                self.statistics.obsolete_frames += 1
                return
            if self.pending_frame is not None:
                self.statistics.replaced_frames += 1
            self.pending_frame = RectifiedDepthFrame(images, calibration)
            self.last_timestamp = timestamp
            self._has_new_data = True

    def is_current(self, timestamp: int) -> bool:
        """Whether an acquisition is within the configured age window."""
        age = (
            self.node.get_clock().now().nanoseconds - timestamp
        ) / CameraUnit.NANOSECONDS_PER_SECOND
        return 0 <= age <= self.camera_config.max_frame_age

    def set_data(self, cas: CAS) -> bool:
        """Return False from :meth:`ROSCameraInterface.set_data` when a frame is dropped."""
        with self.lock:
            frame = self.pending_frame
            self.pending_frame = None
            self._has_new_data = False
        if frame is None:
            return False
        timestamp = Time.from_msg(frame.images.header.stamp)
        if not self.is_current(timestamp.nanoseconds):
            self.statistics.stale_frames += 1
            return False
        frame.validate(self.camera_config.tf_from)
        if (
            not self.camera_config.static_camera_transform_enabled
            and not self.lookup_transform(timestamp)
        ):
            self.statistics.missing_transforms += 1
            return False
        self.store_frame(cas, frame)
        return True

    def store_frame(self, cas: CAS, frame: RectifiedDepthFrame) -> None:
        """Populate RK image views and the acquisition-time camera pose."""
        calibration = frame.rectified_calibration()
        projection = np.asarray(calibration.p).reshape(3, 4)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            calibration.width,
            calibration.height,
            projection[0, 0],
            projection[1, 1],
            projection[0, 2],
            projection[1, 2],
        )
        color = np.ascontiguousarray(
            self.bridge.imgmsg_to_cv2(frame.images.color, CameraEncoding.COLOR)
        )
        depth = frame.millimeter_depth(self.bridge, self.camera_config.max_depth)
        cas.set(CASViews.COLOR_IMAGE, color)
        cas.set(CASViews.DEPTH_IMAGE, depth)
        cas.set(CASViews.CAMERA_INFO, calibration)
        cas.set(CASViews.CAMERA_INTRINSIC, intrinsic)
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
        cas.data_timestamp = Time.from_msg(frame.images.header.stamp).nanoseconds
        cas.camera_frame = self.camera_config.tf_from
        if self.store_static_camera_transform_if_configured(cas, cas.data_timestamp):
            return
        self.store_camera_to_world_transform_from_tf(cas, frame.images.header.stamp)
