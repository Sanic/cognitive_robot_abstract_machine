"""
Query-driven object perception with Foundation Stereo depth.
"""

from __future__ import annotations

# %% imports

from dataclasses import dataclass

from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator
from robokudo.descriptors.analysis_engines.foundation_stereo import (
    AnalysisEngine as FoundationStereoAnalysisEngine,
)
from robokudo.descriptors.analysis_engines.subtree_foundation_stereo import Subtree
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

# %% query pipeline


@dataclass
class AnalysisEngine(FoundationStereoAnalysisEngine):
    """
    Acquire Foundation Stereo images for a query and return matching objects.
    """

    pipeline_name: str = "foundation_stereo_query"
    """
    Name of the query pipeline in the behavior tree.
    """

    def name(self) -> str:
        return self.pipeline_name

    def implementation(self) -> Pipeline:
        result_descriptor = GenerateQueryResult.Descriptor()
        result_descriptor.parameters.filter_by_query = True
        return Pipeline(
            self.name(),
            children=[
                pipeline_init(),
                QueryAnnotator(),
                self.create_reader(),
                Subtree(camera_config=self.camera_config).implementation(),
                GenerateQueryResult(descriptor=result_descriptor),
            ],
        )
