"""
Spatial inspection candidates must not depend on predicted semantic classes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from experiments.warsaw.evaluation.size import ObjectSize
from experiments.warsaw.evaluation.spatial_comparison import (
    ComparisonObject,
    ComparisonSettings,
    SpatialComparison,
)

# %% geometry independent of its semantic label


@pytest.fixture
def cabinet() -> ComparisonObject:
    """
    A metric unit cube with a semantic class and an export identity.
    """
    return ComparisonObject(
        identifier=1,
        name="cabinet",
        classes=["Cabinet"],
        parent_identifier=None,
        bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        size=ObjectSize(extents=[1.0, 1.0, 1.0], surface_area=6.0),
    )


def test_matching_ignores_the_predicted_class(cabinet: ComparisonObject) -> None:
    """
    A wrong class remains visible as a class disagreement on the same object.
    """
    predicted = replace(cabinet, identifier=10, classes=["Oven"])

    comparison = SpatialComparison.between([predicted], [cabinet])

    assert comparison.bodies[0].ground_truth_identifier == cabinet.identifier
    assert comparison.bodies[0].class_disagreement is True
    assert comparison.bodies[0].candidates[0].cost == 0.0


def test_distant_objects_are_left_unmatched(cabinet: ComparisonObject) -> None:
    """
    An available GT object must not force an association across the room.
    """
    predicted = replace(cabinet, identifier=10, bounds=[[10.0] * 3, [11.0] * 3])

    comparison = SpatialComparison.between([predicted], [cabinet])

    assert comparison.bodies[0].ground_truth_identifier is None
    assert comparison.unmatched_ground_truth == [cabinet.identifier]


def test_competing_candidates_and_ties_remain_visible(
    cabinet: ComparisonObject,
) -> None:
    """
    Equal geometric evidence is reported rather than hidden by deterministic order.
    """
    other = replace(cabinet, identifier=2)
    predicted = replace(cabinet, identifier=10)

    comparison = SpatialComparison.between([predicted], [cabinet, other])

    assert [
        candidate.ground_truth_identifier
        for candidate in comparison.bodies[0].candidates
    ] == [cabinet.identifier, other.identifier]
    assert comparison.bodies[0].margin == 0.0


def test_fragments_may_share_the_same_ground_truth(cabinet: ComparisonObject) -> None:
    """
    Inspection preserves many-to-one associations for fragmented predictions.
    """
    predicted = [replace(cabinet, identifier=10), replace(cabinet, identifier=11)]

    comparison = SpatialComparison.between(predicted, [cabinet])

    assert [body.ground_truth_identifier for body in comparison.bodies] == [
        cabinet.identifier
    ] * len(predicted)


def test_parent_disagreement_uses_the_matched_parents(
    cabinet: ComparisonObject,
) -> None:
    """
    An object at the root disagrees with a matched GT part below a whole.
    """
    part = replace(
        cabinet, identifier=2, parent_identifier=1, bounds=[[2.0] * 3, [3.0] * 3]
    )
    predicted = [
        replace(cabinet, identifier=10),
        replace(part, identifier=11, parent_identifier=None),
    ]

    comparison = SpatialComparison.between(predicted, [cabinet, part])

    assert comparison.bodies[1].ground_truth_identifier == part.identifier
    assert comparison.bodies[1].parent_disagreement is True


def test_an_unmatched_parent_cannot_be_judged(cabinet: ComparisonObject) -> None:
    """
    Missing parent correspondence is distinct from an incorrect relation.
    """
    predicted = replace(cabinet, identifier=10, parent_identifier=99)

    comparison = SpatialComparison.between([predicted], [cabinet])

    assert comparison.bodies[0].parent_disagreement is None


def test_comparison_survives_its_json_round_trip(cabinet: ComparisonObject) -> None:
    """
    The saved table is the same correspondence the viewer is meant to inspect.
    """
    comparison = SpatialComparison.between([cabinet], [cabinet], ComparisonSettings())

    assert SpatialComparison.from_json(comparison.to_json()) == comparison
