"""
The ontology, and the ground truth of the scene the tests judge runs against.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    build_taxonomy,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.articulate3d.ground_truth import GroundTruthBuilder

from .scenes import SceneName, kitchen_corner_parts, kitchen_corner_protocol

# %% the ontology


@pytest.fixture(scope="module")
def ontology():
    """
    :return: The ontology's classes by name.
    """
    load_annotation_modules()
    return annotation_classes(SemanticAnnotation)


@pytest.fixture(scope="module")
def taxonomy():
    """
    :return: The ontology as a run is given it.
    """
    return build_taxonomy(SemanticAnnotation)


# %% the scene's ground truth


@pytest.fixture
def ground_truth(ontology):
    """
    :return: The scene's ground truth, by the scene's protocol.
    """
    return GroundTruthBuilder(
        protocol=kitchen_corner_protocol(), ontology=ontology
    ).build(SceneName.KITCHEN_CORNER, kitchen_corner_parts())
