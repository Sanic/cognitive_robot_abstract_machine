from __future__ import annotations

# %% imports

from dataclasses import dataclass, field

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import OpenVocabularyObjectDetectionAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

# %% demo pipeline


@dataclass
class AnalysisEngine(AnalysisEngineInterface):
    """
    Process live Olive images with Fast-FoundationStereo depth.
    """

    camera_config: FoundationStereoCameraConfig = field(
        default_factory=FoundationStereoCameraConfig
    )
    """
    Camera topics and optical frame for the demo or physical camera.
    """

    def name(self) -> str:
        return FoundationStereoCameraConfig.registry_name

    def implementation(self) -> Pipeline:
        reader = CollectionReaderAnnotator.Descriptor(
            camera_config=self.camera_config,
            camera_interface=RectifiedDepthCameraInterface(self.camera_config),
        )
        preprocessor = ImagePreprocessorAnnotator()
        preprocessor.descriptor.parameters.depth_trunc = self.camera_config.max_depth

        point_cloud_crop_descriptor = PointcloudCropAnnotator.Descriptor()
        point_cloud_crop_descriptor.parameters.min_x = -1.0
        point_cloud_crop_descriptor.parameters.max_x = 1.0
        point_cloud_crop_descriptor.parameters.min_y = -1.0
        point_cloud_crop_descriptor.parameters.max_y = 1.0
        point_cloud_crop_descriptor.parameters.min_z = 0.5
        point_cloud_crop_descriptor.parameters.max_z = 3.5
        point_cloud_crop_annotator = PointcloudCropAnnotator(
            descriptor=point_cloud_crop_descriptor
        )

        plane_annotator = PlaneAnnotator()

        point_cloud_cluster_extractor = PointCloudClusterExtractor()
        open_vocab_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        open_vocab_descriptor.parameters.classes = ["bottle", "apple"]
        open_vocab_descriptor.parameters.precision_mode = True
        return Pipeline(
            self.name(),
            children=[
                pipeline_init(),
                CollectionReaderAnnotator(reader),
                preprocessor,
                # point_cloud_crop_annotator,
                # plane_annotator,
                # point_cloud_cluster_extractor,
                # ClusterColorAnnotator(),
                # ClusterColorHistogramAnnotator(),
                OpenVocabularyObjectDetectionAnnotator(descriptor=open_vocab_descriptor),
                ClusterPositionAnnotator(),
            ],
        )
