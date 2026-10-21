from __future__ import annotations

# %% imports

from enum import StrEnum
from pathlib import Path

import yaml
from rclpy.qos import ReliabilityPolicy

from robokudo.descriptors.analysis_engines.foundation_stereo import AnalysisEngine
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.descriptors.camera_configs.config_olive import OliveCameraConfig

# %% configuration contracts


class StereoParameter(StrEnum):
    """
    ROS parameter names defining the stereo pipeline connections.
    """

    DEPTH_NODE = "fast_foundation_stereo"
    CLOUD_NODE = "foundation_stereo_pointcloud"
    PARAMETERS = "ros__parameters"
    LEFT_IMAGE = "left_image_topic"
    RIGHT_IMAGE = "right_image_topic"
    LEFT_CALIBRATION = "left_camera_info_topic"
    RIGHT_CALIBRATION = "right_camera_info_topic"
    DEPTH = "depth_topic"
    COLOR = "rgb_topic"
    CALIBRATION = "camera_info_topic"
    RELIABILITY = "input_reliability"
    RELIABLE = "reliable"


def test_foundation_stereo_defaults_use_live_olive_inputs() -> None:
    """
    Use Olive color and calibration while retaining FFS depth.
    """
    configuration = FoundationStereoCameraConfig()
    olive = OliveCameraConfig()
    assert configuration.topic_color == olive.topic_color
    assert configuration.topic_camera_info == olive.topic_camera_info
    assert configuration.tf_from == olive.tf_from
    assert configuration.topic_depth == "/ffs/depth"
    assert configuration.color_reliability is ReliabilityPolicy.RELIABLE
    assert configuration.calibration_reliability is ReliabilityPolicy.RELIABLE
    assert configuration.lookup_viewpoint is False


def test_foundation_stereo_engine_uses_registered_defaults() -> None:
    """
    The executable analysis engine must not retain the static demo frame.
    """
    configuration = AnalysisEngine().camera_config
    expected = FoundationStereoCameraConfig()
    assert type(configuration) is FoundationStereoCameraConfig
    assert configuration.topic_color == expected.topic_color
    assert configuration.topic_depth == expected.topic_depth
    assert configuration.topic_camera_info == expected.topic_camera_info
    assert configuration.tf_from == expected.tf_from
    assert configuration.lookup_viewpoint is expected.lookup_viewpoint
    assert configuration.color_reliability is expected.color_reliability
    assert configuration.depth_reliability is expected.depth_reliability
    assert configuration.calibration_reliability is expected.calibration_reliability


def test_olive_ros_parameters_match_rk_inputs() -> None:
    """
    Connect rectified stereo inference and optional cloud output to the RK reader.
    """
    parameter_path = (
        Path(__file__).parents[2]
        / "robokudo"
        / "config"
        / "foundation_stereo_olive.yaml"
    )
    configuration = yaml.safe_load(parameter_path.read_text())
    inference = configuration[StereoParameter.DEPTH_NODE][StereoParameter.PARAMETERS]
    cloud = configuration[StereoParameter.CLOUD_NODE][StereoParameter.PARAMETERS]
    camera = FoundationStereoCameraConfig()
    assert inference[StereoParameter.LEFT_IMAGE] == camera.topic_color
    assert inference[StereoParameter.LEFT_CALIBRATION] == camera.topic_camera_info
    assert inference[StereoParameter.RIGHT_IMAGE] == "/right/image_rect_color"
    assert inference[StereoParameter.RIGHT_CALIBRATION] == "/right/camera_info"
    assert inference[StereoParameter.DEPTH] == camera.topic_depth
    assert cloud[StereoParameter.COLOR] == camera.topic_color
    assert cloud[StereoParameter.CALIBRATION] == camera.topic_camera_info
    assert cloud[StereoParameter.DEPTH] == camera.topic_depth
    assert inference[StereoParameter.RELIABILITY] == StereoParameter.RELIABLE
    assert cloud[StereoParameter.RELIABILITY] == StereoParameter.RELIABLE
