"""
Analysis engine for simulated RGB-D input from SemDT RayTracer with query functionality.

This pipeline renders the synchronized robot world and returns detected objects with
their poses and colors.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.behaviours.ensure_world_synchronized import EnsureWorldSynchronized
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator, GenerateQueryResult
from robokudo.annotators.semdt_segmented_objects import SegmentedObjectAnnotator
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    RuntimeRobotWorldSource,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import non_query_pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """
    Build the query pipeline for a served simulated robot world.
    """

    def name(self) -> str:
        """
        Return the module's analysis-engine name.
        """
        return "semdt_raytracer_query_demo"

    def implementation(self) -> Pipeline:
        """
        Render and report every visible object with its pose and colors.
        """
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer",
            source=RuntimeRobotWorldSource(),
        )
        seq = Pipeline("SemDTRayTracerPipeline")
        seq.add_children(
            [
                # Waiting for an active goal at the beginning or end of the
                # pipeline can block a new query before it is processed.
                non_query_pipeline_init(),
                QueryAnnotator(),
                EnsureWorldSynchronized(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                SegmentedObjectAnnotator(),
                ClusterColorAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return seq
