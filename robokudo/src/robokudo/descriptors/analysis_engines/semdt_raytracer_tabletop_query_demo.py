"""
Query the served robot world using RGB-D tabletop segmentation.
"""

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.behaviours.ensure_world_synchronized import EnsureWorldSynchronized
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
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
    Find tabletop clusters from the synchronized camera's RGB-D frame.
    """

    def name(self) -> str:
        """
        Return the analysis-engine name used by RoboKudo's launcher.
        """
        return "semdt_raytracer_tabletop_query_demo"

    def implementation(self) -> Pipeline:
        """
        Return every posed tabletop cluster with its observed colors.
        """
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer",
            source=RuntimeRobotWorldSource(),
        )
        plane_desc = PlaneAnnotator.Descriptor()
        plane_desc.parameters.distance_threshold = 0.01
        plane_desc.parameters.random_seed = 0

        cluster_desc = PointCloudClusterExtractor.Descriptor()
        cluster_desc.parameters.dbscan_min_cluster_count = 8
        cluster_desc.parameters.min_cluster_count = 20
        cluster_desc.parameters.min_on_plane_point_count = 10
        cluster_desc.parameters.eps = 0.05

        pipeline = Pipeline("SemDTTabletopQueryPipeline")
        pipeline.add_children(
            [
                # QueryAnnotator waits for each goal; an active-goal gate here
                # would also block a goal that arrives during the previous reset.
                non_query_pipeline_init(),
                QueryAnnotator(),
                EnsureWorldSynchronized(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_desc),
                PointCloudClusterExtractor(descriptor=cluster_desc),
                ClusterColorAnnotator(),
                ClusterPoseBBAnnotator(),
                GenerateQueryResult(),
            ]
        )
        return pipeline
