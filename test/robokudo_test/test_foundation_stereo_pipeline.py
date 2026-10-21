from __future__ import annotations

# %% imports

from copy import deepcopy

import numpy as np
import pytest
from py_trees.common import Status
from rclpy.node import Node
from rclpy.time import Time

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.analysis_engines.foundation_stereo import AnalysisEngine
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.io.exceptions import InvalidDepthCameraConfiguration
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface

from .test_rectified_depth_camera import AlignedImages, aligned_images, camera

# %% pipeline compatibility


def test_demo_pipeline_generates_colored_metric_cloud(
    node: Node,
    aligned_images: AlignedImages,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    The example descriptor runs RK preprocessing without a depth-unit override.
    """
    engine = AnalysisEngine()
    pipeline = engine.implementation()
    pipeline.setup()
    reader = next(
        child
        for child in pipeline.children
        if isinstance(child, CollectionReaderAnnotator)
    )
    preprocessor = next(
        child
        for child in pipeline.children
        if isinstance(child, ImagePreprocessorAnnotator)
    )
    interface = reader.descriptor.parameters.camera_interface
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
    cloud = pipeline.cas.get(CASViews.CLOUD)
    np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], [1.25, 2.5])
    np.testing.assert_allclose(
        np.asarray(cloud.colors), np.tile(np.array([10, 20, 30]) / 255, (2, 1))
    )
    assert (
        preprocessor.descriptor.parameters.depth_trunc == engine.camera_config.max_depth
    )


# %% bounded buffering


def test_latest_complete_frame_replaces_pending_frame(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    Slow pipeline consumption retains only the latest complete acquisition.
    """
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    newer = deepcopy(aligned_images)
    for message in (newer.color, newer.depth, newer.calibration):
        message.header.stamp = camera.node.get_clock().now().to_msg()
    newer.depth.header.stamp = deepcopy(newer.color.header.stamp)
    newer.calibration.header.stamp = deepcopy(newer.color.header.stamp)
    camera.receive_images(newer.color, newer.depth)
    camera.camera_info_subscriber.signalMessage(newer.calibration)
    cas = CAS()
    camera.set_data(cas)
    assert cas.data_timestamp == Time.from_msg(newer.color.header.stamp).nanoseconds
    assert camera.statistics.replaced_frames == 1
    assert camera.has_new_data() is False


def test_unmatched_image_history_is_bounded(
    node: Node, aligned_images: AlignedImages
) -> None:
    """
    A depth stall cannot grow the RGB synchronization history indefinitely.
    """
    config = FoundationStereoCameraConfig(
        lookup_viewpoint=False, synchronization_queue_size=2
    )
    interface = RectifiedDepthCameraInterface(config, node=node)
    for offset in range(config.synchronization_queue_size + 1):
        color = deepcopy(aligned_images.color)
        color.header.stamp = Time(
            nanoseconds=Time.from_msg(color.header.stamp).nanoseconds + offset
        ).to_msg()
        interface.color_subscriber.signalMessage(color)
    interface.depth_subscriber.signalMessage(aligned_images.depth)
    interface.camera_info_subscriber.signalMessage(aligned_images.calibration)
    assert interface.has_new_data() is False
    assert (
        len(interface.image_synchronizer.queues[0]) == config.synchronization_queue_size
    )


def test_zero_calibration_tolerance_matches_equal_stamps(
    node: Node, aligned_images: AlignedImages
) -> None:
    """
    Exact calibration matching remains usable when tolerance is disabled.
    """
    interface = RectifiedDepthCameraInterface(
        FoundationStereoCameraConfig(lookup_viewpoint=False, calibration_tolerance=0),
        node=node,
    )
    interface.color_subscriber.signalMessage(aligned_images.color)
    interface.depth_subscriber.signalMessage(aligned_images.depth)
    interface.camera_info_subscriber.signalMessage(aligned_images.calibration)
    assert interface.has_new_data() is True


def test_unavailable_transform_keeps_collection_waiting(
    camera: RectifiedDepthCameraInterface,
    aligned_images: AlignedImages,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A missing acquisition-time pose cannot advance downstream annotators.
    """
    pipeline = AnalysisEngine(camera_config=camera.camera_config).implementation()
    reader = next(
        child
        for child in pipeline.children
        if isinstance(child, CollectionReaderAnnotator)
    )
    reader.descriptor.parameters.camera_interface = camera
    monkeypatch.setattr(camera, "lookup_transform", lambda timestamp: False)
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    assert reader.update() is Status.RUNNING
    assert camera.statistics.missing_transforms == 1


# %% configuration validation


@pytest.mark.parametrize(
    "configuration",
    [
        {"synchronization_queue_size": 0},
        {"history_depth": 0},
        {"max_frame_age": float("nan")},
        {"max_depth": -1.0},
        {"calibration_tolerance": -1.0},
        {"static_camera_transform_enabled": True},
        {"static_camera_transform_enabled": True, "lookup_viewpoint": False},
    ],
)
def test_invalid_configuration_is_rejected(configuration: dict[str, object]) -> None:
    """
    Reject invalid buffering, geometry limits and contradictory transform sources.
    """
    with pytest.raises(InvalidDepthCameraConfiguration):
        FoundationStereoCameraConfig(**configuration)
