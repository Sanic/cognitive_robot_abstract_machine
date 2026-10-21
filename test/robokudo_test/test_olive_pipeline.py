from __future__ import annotations

# %% imports

import numpy as np
from py_trees.common import Status
from py_trees.composites import Sequence
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.descriptors.analysis_engines.olive import AnalysisEngine
from robokudo.descriptors.camera_configs.config_olive import OliveCameraConfig
from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

from .test_rectified_depth_camera import AlignedImages, aligned_images

# %% camera configuration


def test_olive_configuration_matches_rectified_publisher() -> None:
    """
    Select the aligned color and meter depth streams with reliable delivery.
    """
    configuration = CameraConfigRegistry.create_config(OliveCameraConfig.registry_name)
    assert type(configuration) is OliveCameraConfig
    assert configuration.topic_color == "/left/image_rect_color"
    assert configuration.topic_depth == "/depth/image"
    assert configuration.topic_camera_info == "/left/camera_info"
    assert configuration.tf_from == "olive_left_optical_frame"
    assert configuration.lookup_viewpoint is False
    assert configuration.calibration_tolerance == 0
    assert configuration.color_reliability is ReliabilityPolicy.RELIABLE
    assert configuration.depth_reliability is ReliabilityPolicy.RELIABLE
    assert configuration.calibration_reliability is ReliabilityPolicy.RELIABLE


def test_olive_factory_selects_rectified_depth_reader(node: Node) -> None:
    """
    The registered Olive source uses the meter-depth reader.
    """
    descriptor = CollectionReaderDescriptorFactory.create_descriptor(
        OliveCameraConfig.registry_name
    )
    assert type(descriptor.parameters.camera_config) is OliveCameraConfig
    assert type(descriptor.parameters.camera_interface) is RectifiedDepthCameraInterface


# %% sensor display


def test_olive_demo_ends_at_image_preprocessor(
    node: Node,
    aligned_images: AlignedImages,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    The demo displays aligned sensor data and a metric cloud without later stages.
    """
    engine = AnalysisEngine()
    pipeline = engine.implementation()
    assert engine.name() == OliveCameraConfig.registry_name
    assert [type(child) for child in pipeline.children] == [
        Sequence,
        CollectionReaderAnnotator,
        ImagePreprocessorAnnotator,
    ]
    reader = pipeline.children[1]
    preprocessor = pipeline.children[2]
    interface = reader.descriptor.parameters.camera_interface
    assert type(interface) is RectifiedDepthCameraInterface
    assert type(interface.camera_config) is OliveCameraConfig
    assert interface.camera_config is engine.camera_config
    pipeline.setup()
    for message in (
        aligned_images.color,
        aligned_images.depth,
        aligned_images.calibration,
    ):
        message.header.frame_id = engine.camera_config.tf_from
    interface.color_subscriber.signalMessage(aligned_images.color)
    interface.depth_subscriber.signalMessage(aligned_images.depth)
    interface.camera_info_subscriber.signalMessage(aligned_images.calibration)
    assert reader.update() is Status.SUCCESS
    assert preprocessor.update() is Status.SUCCESS
    expected_depth = CVBridgeWorkaround().imgmsg_to_cv2(aligned_images.depth)
    valid_pixels = np.isfinite(expected_depth) & (expected_depth > 0)
    cloud = pipeline.cas.get(CASViews.CLOUD)
    np.testing.assert_allclose(
        np.asarray(cloud.points)[:, 2], expected_depth[valid_pixels]
    )
    assert (
        preprocessor.descriptor.parameters.depth_trunc == engine.camera_config.max_depth
    )
