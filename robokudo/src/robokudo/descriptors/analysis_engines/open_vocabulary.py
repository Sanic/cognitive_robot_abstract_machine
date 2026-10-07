"""
Detect objects matching configured text labels in preprocessed images.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
)
from robokudo.pipeline import Pipeline
from robokudo.idioms import pipeline_init
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)


class AnalysisEngine(AnalysisEngineInterface):
    """
    Analysis engine for open-vocabulary object detection in Kinect images.
    """

    def name(self) -> str:
        return "open_vocabulary"

    def implementation(self) -> Pipeline:
        kinect_config = CollectionReaderDescriptorFactory.create_descriptor(
            "kinect_wo_tf"
        )

        open_vocabulary_descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
        open_vocabulary_descriptor.parameters.classes = [
            "electric device",
        ]
        open_vocabulary_descriptor.parameters.precision_mode = True

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                OpenVocabularyObjectDetectionAnnotator(
                    descriptor=open_vocabulary_descriptor
                ),
                ObjectHypothesisVisualizer(),
            ]
        )
        return seq
