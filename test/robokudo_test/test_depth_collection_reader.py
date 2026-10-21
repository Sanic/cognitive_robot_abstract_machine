from __future__ import annotations

# %% imports

import pytest
from py_trees.common import Status
from rclpy.node import Node

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface
from robokudo.pipeline import Pipeline

from .test_rectified_depth_camera import AlignedImages, aligned_images, camera

# %% collection delivery


def test_collection_reader_waits_when_frame_expires(
    camera: RectifiedDepthCameraInterface, aligned_images: AlignedImages
) -> None:
    """
    A dropped frame must not advance the pipeline with an empty CAS.
    """
    camera.receive_images(aligned_images.color, aligned_images.depth)
    camera.camera_info_subscriber.signalMessage(aligned_images.calibration)
    reader = CollectionReaderAnnotator(
        CollectionReaderAnnotator.Descriptor(camera.camera_config, camera)
    )
    Pipeline(children=[reader])
    camera.camera_config.max_frame_age = 1e-9
    assert reader.update() is Status.RUNNING


def test_factory_selects_rectified_depth_interface(node: Node) -> None:
    """
    The named source constructs the correct camera interface.
    """
    descriptor = CollectionReaderDescriptorFactory.create_descriptor(
        FoundationStereoCameraConfig.registry_name, lookup_viewpoint=False
    )
    assert type(descriptor.parameters.camera_interface) is RectifiedDepthCameraInterface
