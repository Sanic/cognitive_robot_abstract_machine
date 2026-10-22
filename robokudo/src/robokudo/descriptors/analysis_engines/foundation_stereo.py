from __future__ import annotations

# %% imports

from dataclasses import dataclass, field

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.analysis_engines.subtree_foundation_stereo import Subtree
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
        return Pipeline(
            self.name(),
            children=[
                pipeline_init(),
                self.create_reader(),
                *Subtree(camera_config=self.camera_config).create_annotators(),
            ],
        )

    def create_reader(self) -> CollectionReaderAnnotator:
        """
        Construct acquisition of registered color, depth and camera calibration.

        :return: Collection reader using this engine's camera configuration.
        """
        descriptor = CollectionReaderAnnotator.Descriptor(
            camera_config=self.camera_config,
            camera_interface=RectifiedDepthCameraInterface(self.camera_config),
        )
        return CollectionReaderAnnotator(descriptor=descriptor)
