"""
Object perception from registered Foundation Stereo images and depth.
"""

from __future__ import annotations

# %% imports

from dataclasses import dataclass, field

from py_trees.composites import Sequence
from typing_extensions import TYPE_CHECKING

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator

# %% perception subtree


@dataclass
class Subtree(SubtreeInterface):
    """
    Detect and segment objects, then annotate their colors and 3D positions.
    """

    camera_config: FoundationStereoCameraConfig = field(
        default_factory=FoundationStereoCameraConfig
    )
    """
    Camera configuration defining the valid depth range.
    """

    object_classes: list[str] = field(default_factory=lambda: ["bottle", "apple"])
    """
    Object labels supplied to open-vocabulary detection.
    """

    precision_mode: bool = True
    """
    Whether Segment Anything refines detected object masks.
    """

    subtree_name: str = "Foundation Stereo Object Perception"
    """
    Name of the processing sequence in the behavior tree.
    """

    def name(self) -> str:
        return self.subtree_name

    def implementation(self) -> Sequence:
        return Sequence(self.name(), memory=True, children=self.create_annotators())

    def create_annotators(self) -> list[BaseAnnotator]:
        """
        Construct preprocessing, object detection, color and position annotation.

        :return: Fresh annotators in their processing order.
        """
        preprocessor = ImagePreprocessorAnnotator()
        preprocessor.descriptor.parameters.depth_trunc = self.camera_config.max_depth
        detection_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        detection_descriptor.parameters.classes = list(self.object_classes)
        detection_descriptor.parameters.precision_mode = self.precision_mode
        return [
            preprocessor,
            OpenVocabularyObjectDetectionAnnotator(descriptor=detection_descriptor),
            ClusterColorAnnotator(),
            ClusterPositionAnnotator(),
        ]
