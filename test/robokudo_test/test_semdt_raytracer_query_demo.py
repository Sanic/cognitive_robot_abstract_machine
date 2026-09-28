"""
The ray-traced query engine reports visible objects without demo-specific classes.
"""

from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.query import GenerateQueryResult
from robokudo.annotators.semdt_segmented_objects import SegmentedObjectAnnotator
from robokudo.behaviours.ensure_world_synchronized import EnsureWorldSynchronized
from robokudo.descriptors.analysis_engines.semdt_raytracer_query_demo import (
    AnalysisEngine,
)
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    RuntimeRobotWorldSource,
)


def test_query_pipeline_returns_all_visible_objects_with_colors():
    """
    The camera pipeline keeps color and pose stages before an unfiltered result.
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

    stages = (SegmentedObjectAnnotator, ClusterColorAnnotator, GenerateQueryResult)
    positions = [
        next(index for index, child in enumerate(children) if isinstance(child, stage))
        for stage in stages
    ]
    assert positions == sorted(positions)
    result = children[positions[-1]]
    assert not result.descriptor.parameters.filter_by_query
