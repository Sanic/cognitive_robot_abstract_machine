"""
The tabletop query engine segments rendered depth without body labels.
"""

from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.behaviours.ensure_world_synchronized import EnsureWorldSynchronized
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.query import GenerateQueryResult
from robokudo.annotators.semdt_segmented_objects import SegmentedObjectAnnotator
from robokudo.descriptors.analysis_engines.semdt_raytracer_tabletop_query_demo import (
    AnalysisEngine,
)
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    RuntimeRobotWorldSource,
)


def test_tabletop_query_pipeline_segments_depth_and_returns_all_colored_clusters():
    """
    The alternate AE uses a served RGB-D frame and the tabletop annotators.
    """
    children = AnalysisEngine().implementation().children
    reader = next(
        child for child in children if isinstance(child, CollectionReaderAnnotator)
    )
    assert isinstance(
        reader.descriptor.parameters.camera_config.source, RuntimeRobotWorldSource
    )
    assert next(
        index
        for index, child in enumerate(children)
        if isinstance(child, EnsureWorldSynchronized)
    ) < children.index(reader)
    assert not any(isinstance(child, SegmentedObjectAnnotator) for child in children)

    stages = (
        PlaneAnnotator,
        PointCloudClusterExtractor,
        ClusterColorAnnotator,
        ClusterPoseBBAnnotator,
        GenerateQueryResult,
    )
    positions = [
        next(index for index, child in enumerate(children) if isinstance(child, stage))
        for stage in stages
    ]
    assert positions == sorted(positions)
    result = children[positions[-1]]
    assert not result.descriptor.parameters.filter_by_query
