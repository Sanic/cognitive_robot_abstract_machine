"""
Camera interface module for RoboKudo.

This module provides base classes and implementations for interfacing with
various camera types in RoboKudo. It supports:

* ROS camera interfaces (raw and compressed)
* Kinect-style RGB-D cameras
* Camera calibration handling
* Transform lookups
* Synchronized data acquisition
* Thread-safe operation

The module handles:

* RGB and depth image acquisition
* Camera calibration information
* Camera-to-world transforms
* Data synchronization
* Format conversions
"""

from __future__ import annotations

import logging
import struct
from threading import Lock

import builtin_interfaces.msg
import cv2
import message_filters
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from tf2_ros import Buffer
from typing_extensions import Optional, List, Any, TYPE_CHECKING, Union, Tuple

from robokudo.cas import CASViews, CAS
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import (
    CameraAnnotationAmbiguous,
    CameraDataMissing,
)
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from robokudo.io.tf_listener_proxy import TFListenerProxy
from robokudo.types.camera import CameraObservation
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from robokudo.world import (
    init_world_entity_tracker_from_world,
    setup_world_for_camera_frame,
    update_connection_transform,
    world_instance,
)
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    import numpy.typing as npt


class CameraInterface(object):
    """
    Base class for all camera interfaces in RoboKudo.

    This class defines the basic interface that all camera implementations must provide.
    It handles configuration and data availability tracking.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the camera interface.

        :param camera_config: Configuration for the camera
        """
        self._has_new_data: bool = False
        """
        Whether new data is available.
        """
        self.camera_config: Any = camera_config
        """
        Camera configuration object.
        """
        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """
        RoboKudo logger instance.
        """

    def has_new_data(self) -> bool:
        """
        Check if new data is available.

        :return: True if new data is available, False otherwise
        """
        return self._has_new_data

    @staticmethod
    def bind_world_T_camera(
        world_frame: str,
        camera_frame: str,
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> HomogeneousTransformationMatrix:
        """Bind a sampled camera pose to bodies in the runtime world.

        :param world_frame: Name of the pose reference frame.
        :param camera_frame: Name of the camera frame.
        :param world_T_camera: Sampled numeric camera pose.
        :return: Camera pose bound to runtime-world bodies.
        """
        setup_world_for_camera_frame(world_frame=world_frame, camera_frame=camera_frame)

        world = world_instance()
        camera_body = world.get_body_by_name(name=camera_frame)
        world_body = world.get_body_by_name(name=world_frame)

        runtime_world_T_camera = HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )

        update_connection_transform(
            to_name=world_body.name,
            from_name=camera_body.name,
            transform=runtime_world_T_camera,
        )
        return runtime_world_T_camera

    def static_world_T_camera_if_configured(
        self,
    ) -> HomogeneousTransformationMatrix | None:
        """Return the configured static camera pose bound to the runtime world."""
        if not self.camera_config.static_camera_transform_enabled:
            return None

        return self.bind_world_T_camera(
            world_frame=self.camera_config.static_world_frame,
            camera_frame=self.camera_config.static_camera_frame,
            world_T_camera=self.camera_config.static_world_T_camera,
        )

    def store_camera_observation(
        self,
        cas: CAS,
        camera_model: PinholeCameraModel,
        camera_frame: str,
        timestamp_nanoseconds: int,
        modalities: tuple[CameraModality, ...],
        world_T_camera: HomogeneousTransformationMatrix | None,
    ) -> None:
        """Store semantic identity and effective calibration for a camera frame.

        :param cas: CAS receiving the observation.
        :param camera_model: Effective calibration of the delivered image.
        :param camera_frame: Optical frame associated with the calibration.
        :param timestamp_nanoseconds: Acquisition time in nanoseconds since the epoch.
        :param modalities: Kinds of image data delivered by the interface.
        :param world_T_camera: Sampled camera pose, if available.
        """
        camera = self._resolve_or_create_camera(
            camera_frame=camera_frame,
            camera_model=camera_model,
            modalities=modalities,
        )
        cas.camera_observation = CameraObservation(
            camera=camera,
            effective_camera_model=camera_model,
            world_T_camera=world_T_camera,
            timestamp_nanoseconds=timestamp_nanoseconds,
        )

    def _resolve_or_create_camera(
        self,
        camera_frame: str,
        camera_model: PinholeCameraModel,
        modalities: tuple[CameraModality, ...],
    ) -> Camera:
        """Resolve the stream camera or add a standalone semantic camera.

        :param camera_frame: Optical frame associated with the image stream.
        :param camera_model: First trustworthy calibration for a new camera.
        :param modalities: Kinds of image data produced by a new camera.
        :return: Camera annotation representing the physical image source.
        :raises CameraAnnotationAmbiguous: If several cameras use the stream frame.
        """
        runtime_world = world_instance()
        cameras = runtime_world.get_semantic_annotations_by_type(Camera)
        frame_cameras = [
            camera
            for camera in cameras
            if camera.root.name.name == camera_frame
            or str(camera.root.name) == camera_frame
        ]
        if len(frame_cameras) == 1:
            return frame_cameras[0]
        if len(frame_cameras) > 1:
            raise CameraAnnotationAmbiguous(
                camera_names=tuple(str(camera.name) for camera in frame_cameras)
            )

        camera_bodies = runtime_world.get_bodies_by_name(camera_frame)
        existing_root = runtime_world.root
        with runtime_world.modify_world():
            if len(camera_bodies) == 0:
                camera_body = Body(name=PrefixedName(name=camera_frame))
                runtime_world.add_body(camera_body)
                if existing_root is not None:
                    # Keep the semantic world connected without treating this
                    # unconstrained mount as an observed camera pose.
                    camera_mount = Connection6DoF.create_with_dofs(
                        parent=existing_root,
                        child=camera_body,
                        world=runtime_world,
                        name=PrefixedName(
                            name=f"{camera_frame}_T_{existing_root.name.name}"
                        ),
                    )
                    runtime_world.add_connection(camera_mount)
            else:
                camera_body = camera_bodies[0]
            camera = Camera(
                name=PrefixedName(name=camera_frame),
                root=camera_body,
                forward_facing_axis=Vector3.Z(),
                camera_model=camera_model,
                modalities=modalities,
            )
            runtime_world.add_semantic_annotation(camera)
        init_world_entity_tracker_from_world(runtime_world)
        return camera

    def set_data(self, cas: CAS) -> None:
        """
        This method is supposed to read in, convert (if needed) and put the data into
        the CAS. If you are running a CameraInterface which is getting data via callback
        methods, please make sure to keep callbacks light and do the main conversion
        work here! Callbacks should be short.

        :param cas: The CAS where the data should be placed in
        """
        raise NotImplementedError


class ROSCameraInterface(CameraInterface):
    """
    Base class for ROS-based camera interfaces.

    This class extends the base camera interface with ROS-specific functionality like
    transform lookups and camera intrinsics handling.
    """

    def __init__(self, camera_config: Any, node: Optional[Node] = None) -> None:
        """
        Initialize the ROS camera interface.

        :param camera_config: Configuration for the ROS camera
        :param node: A ROS node for transform lookups
        """
        super().__init__(camera_config)

        self.node = node if node is not None else ROSNodeRegistry().get()
        """
        ROS node for communication with ROS.
        """
        if hasattr(self.camera_config, "lookup_viewpoint"):
            self.lookup_viewpoint: bool = self.camera_config.lookup_viewpoint

            self.tf_from: str = camera_config.tf_from
            """
            Transform source frame.
            """
            self.tf_to: str = camera_config.tf_to
            """
            Transform target frame.
            """
        else:
            self.lookup_viewpoint: bool = False
            """Whether to look up camera transforms"""

        self.camera_translation: List[float] = [0.0, 0.0, 0.0]
        """Camera translation from TF"""

        self.camera_quaternion: List[float] = [0.0, 0.0, 0.0, 1.0]
        """Camera rotation from TF"""

        if self.lookup_viewpoint:
            self.tf_buffer: Buffer = TFListenerProxy().buffer_for_node(self.node)
            """TF buffer populated by the per-node transform listener."""

    def lookup_transform(self, timestamp: Time = Time()) -> bool:
        """Look up the camera transform from TF.

        :return: True if transform lookup succeeded, False otherwise
        """
        if self.lookup_viewpoint:
            try:
                world_T_camera = self.tf_buffer.lookup_transform(
                    target_frame=self.tf_to,
                    source_frame=self.tf_from,
                    time=timestamp,
                    timeout=Duration(seconds=1.0 / 30.0),
                )
                transform = world_T_camera.transform
                translation = transform.translation
                rotation = transform.rotation

                self.camera_translation = [
                    float(translation.x),
                    float(translation.y),
                    float(translation.z),
                ]
                self.camera_quaternion = [
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ]
            except Exception as err:
                self.rk_logger.warning(
                    f"cannot transform from {self.tf_from} to {self.tf_to} at ts {timestamp}: {err}"
                )
                return False
        return True

    def world_T_camera_from_tf(
        self, timestamp: builtin_interfaces.msg.Time
    ) -> HomogeneousTransformationMatrix | None:
        """Return the sampled TF camera pose bound to the runtime world.

        :param timestamp: Acquisition time associated with the cached TF sample.
        :return: Bound camera pose, or ``None`` when TF lookup is disabled.
        """
        if not self.lookup_viewpoint:
            return None
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=self.camera_translation[0],
            pos_y=self.camera_translation[1],
            pos_z=self.camera_translation[2],
            quat_x=self.camera_quaternion[0],
            quat_y=self.camera_quaternion[1],
            quat_z=self.camera_quaternion[2],
            quat_w=self.camera_quaternion[3],
        )
        return self.bind_world_T_camera(
            world_frame=self.tf_to,
            camera_frame=self.tf_from,
            world_T_camera=world_T_camera,
        )


def depth_convert_workaround(msg: CompressedImage) -> npt.NDArray:
    """
    Convert compressed depth image to proper depth format.

    This is a workaround for handling compressed depth images in ROS.
    Source: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/

    :param msg: Compressed depth image message
    :return: Depth image as numpy array
    :raises Exception: If compression type is wrong or decoding fails
    """
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(";")
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if "compressedDepth" not in compr_type:
        raise Exception(
            "Compression type is not 'compressedDepth'."
            "You probably subscribed to the wrong topic."
        )

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(
        np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
    )
    # replaced np.fromstring with np.frombuffer because np.fromstring is deprecated in newer versions of numpy
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception(
            "Could not decode compressed depth image."
            "You may need to change 'depth_header_size'!"
        )

    if depth_fmt == "16UC1":
        # write raw image data
        return depth_img_raw
    elif depth_fmt == "32FC1":
        raw_header = msg.data[:depth_header_size]
        # header: int, float, float
        [compfmt, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
        depth_img_scaled = depthQuantA / (
            depth_img_raw.astype(np.float32) - depthQuantB
        )
        # filter max values
        depth_img_scaled[depth_img_raw == 0] = 0

        # depth_img_scaled provides distance in meters as f32
        # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        depth_img_mm = (depth_img_scaled * 1000).astype(np.uint16)
        return depth_img_mm
    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


class KinectCameraInterface(ROSCameraInterface):
    """
    Interface for Kinect-style RGB-D cameras using ROS.

    This class implements a camera interface for RGB-D cameras that publish color and
    depth images through ROS topics. It supports both raw and compressed image formats.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the Kinect camera interface.

        Sets up ROS subscribers and synchronization for color, depth, and camera info
        topics.

        :param camera_config: Configuration for the Kinect camera
        """
        super().__init__(camera_config)

        self.color_subscriber: message_filters.Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_color_configured() else Image,
            camera_config.topic_color,
        )
        """
        Color image subscriber.
        """
        self.depth_subscriber: message_filters.Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_depth_configured() else Image,
            camera_config.topic_depth,
        )
        """Depth image subscriber"""

        self.camera_info_subscriber: message_filters.Subscriber = Subscriber(
            self.node, CameraInfo, camera_config.topic_camera_info
        )
        """Camera info subscriber"""
        # self.camera_info_sub = self.node.create_subscription(CameraInfo, camera_config.topic_camera_info,
        #                                                   self.blackhole_callback, 10)

        ts = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber, self.camera_info_subscriber],
            queue_size=10,
            slop=0.4,
        )
        ts.registerCallback(self.callback)

        self.rk_logger.info("Subscribed to: ")
        self.rk_logger.info(f"  {camera_config.topic_color}")
        self.rk_logger.info(f"  {camera_config.topic_depth}")
        self.rk_logger.info(f"  {camera_config.topic_camera_info}")

        self.color: Optional[npt.NDArray] = None
        """
        Latest color image.
        """
        self.depth: Optional[npt.NDArray] = None
        """
        Latest depth image
        """

        self.camera_info: Optional[CameraInfo] = None
        """
        Latest camera info message
        """

        self.color2depth_ratio: Optional[Tuple[float, float]] = None
        """
        Ratio between color and depth image sizes
        """

        self.timestamp: Optional[builtin_interfaces.msg.Time] = None
        """
        Latest message timestamp
        """

        self.lock: Lock = Lock()
        """Thread synchronization lock"""

        self.bridge: CVBridgeWorkaround = CVBridgeWorkaround()
        """NumPy-compatible replacement for cv_bridge."""

    def compressed_depth_configured(self) -> bool:
        """
        Check if compressed depth images are configured.

        :return: True if compressed depth is configured, False otherwise
        """
        return (
            hasattr(self.camera_config, "depth_hints")
            and self.camera_config.depth_hints == "compressedDepth"
        )

    def compressed_color_configured(self) -> bool:
        """
        Check if compressed color images are configured.

        :return: True if compressed color is configured, False otherwise
        """
        return (
            hasattr(self.camera_config, "color_hints")
            and self.camera_config.color_hints == "compressed"
        )

    def get_node(self) -> rclpy.node.Node:
        return self.node

    def blackhole_callback(self, data: Any) -> None:
        """
        This callback is just a dummy to receive data coming from a workaround
        subscription to handle problems with the ApproximateTimeSynchronizer.

        :param data: Dummy data
        """
        pass

    def callback(
        self,
        color_data: Union[Image, CompressedImage],
        depth_data: Optional[Union[Image, CompressedImage]] = None,
        camera_info: Optional[CameraInfo] = None,
    ) -> None:
        """
        Process synchronized camera data.

        This callback handles incoming color, depth, and camera info messages. It
        converts the data to OpenCV format and stores it for later use.

        TODO make this generic. handle the encoding and order properly. For standard and
        compressed images. this might also depend on the fix of image_transport_plugins
        being published as a package. Startpoint can be found at the bottom of this
        method.

        :param color_data: Color image message
        :param depth_data: Depth image message
        :param camera_info: Camera calibration message
        """
        self.lock.acquire()
        if self.rk_logger.isEnabledFor(logging.DEBUG):
            self.rk_logger.debug("Received data:")

            color_time = Time(
                seconds=color_data.header.stamp.sec,
                nanoseconds=color_data.header.stamp.nanosec,
            )

            if depth_data is not None:
                depth_time = Time(
                    seconds=depth_data.header.stamp.sec,
                    nanoseconds=depth_data.header.stamp.nanosec,
                )
                self.rk_logger.debug(
                    f"  Color time - Depth time: {(color_time - depth_time).nanoseconds / 1e9:.6f}"
                )

            if camera_info is not None:
                camera_info_time = Time(
                    seconds=camera_info.header.stamp.sec,
                    nanoseconds=camera_info.header.stamp.nanosec,
                )
                self.rk_logger.debug(
                    f"  Color time - Camera Info time: {(color_time - camera_info_time).nanoseconds / 1e9:.6f}"
                )

        if self.compressed_color_configured():
            color_arr = np.frombuffer(color_data.data, np.uint8)
            self.color = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
        else:
            self.color = self.bridge.imgmsg_to_cv2(color_data, "bgr8")

        self.timestamp = color_data.header.stamp

        if self.compressed_depth_configured():
            if depth_data is None:
                raise CameraDataMissing(
                    data_name="Depth data",
                    context="compressed depth conversion",
                )
            self.depth = depth_convert_workaround(depth_data)
        else:
            if depth_data is None:
                raise CameraDataMissing(
                    data_name="Depth data",
                    context="uncompressed depth conversion",
                )
            self.depth = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")

        self.camera_info = camera_info

        # self.rk_logger.info("Callback processing done - Final steps")
        if not self.lookup_transform():
            self._has_new_data = False
            self.lock.release()
            return

        self._has_new_data = True
        self.lock.release()

    def set_data(self, cas: CAS) -> None:
        if not self.has_new_data():
            return

        self.lock.acquire()
        if self.camera_config.hi_res_mode:
            self.color = self.color[0:960, 0:1280]

        width = self.camera_info.width
        height = self.camera_info.height
        if self.camera_config.hi_res_mode:
            height = 960
        self.camera_info.width = width
        self.camera_info.height = height

        self.color2depth_ratio = self.camera_config.color2depth_ratio

        cas.set(CASViews.COLOR_IMAGE, self.color)
        cas.set(CASViews.DEPTH_IMAGE, self.depth)
        cas.set(CASViews.COLOR2DEPTH_RATIO, self.color2depth_ratio)

        world_T_camera = self.world_T_camera_from_tf(self.timestamp)
        camera_frame = self.camera_info.header.frame_id or self.camera_config.tf_from
        self.store_camera_observation(
            cas=cas,
            camera_model=RosCameraModelAdapter.from_camera_info(self.camera_info),
            camera_frame=camera_frame,
            timestamp_nanoseconds=(
                self.timestamp.sec * 1_000_000_000 + self.timestamp.nanosec
            ),
            modalities=(CameraModality.COLOR, CameraModality.DEPTH),
            world_T_camera=world_T_camera,
        )

        self._has_new_data = False

        self.lock.release()
