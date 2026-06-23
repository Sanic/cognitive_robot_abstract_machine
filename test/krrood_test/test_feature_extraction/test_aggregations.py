from dataclasses import dataclass

import pytest
from typing_extensions import List

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
    get_aggregation_class,
)
from krrood.parametrization.feature_extraction.feature_extractor import (
    FeatureExtractor,
)
from krrood.entity_query_language.core.mapped_variable import Call
from random_events.interval import SimpleInterval, Bound
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.example_classes import (
    SceneObject,
    SceneRoom,
    KRROODPosition,
    KRROODOrientation,
    SceneObjectType,
    TestExParts,
)


@pytest.fixture
def example_scenario():
    obj1 = SceneObject(type=SceneObjectType.TABLE)
    obj2 = SceneObject(type=SceneObjectType.CHAIR)
    obj3 = SceneObject(type=...)
    obj4 = SceneObject(type=...)
    obj5 = SceneObject(type=...)
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1, obj2, obj3, obj4, obj5],
    )
    return room


def test_single_aggregation(example_scenario):
    room = example_scenario
    aggregation_cls = get_aggregation_class(type(room))
    aggregation_instance = aggregation_cls(instance=room, field_name="objects")
    aggregation_features = aggregation_instance.aggregation_features
    aggregations = aggregation_instance.symbolic_aggregation_features()
    assert aggregation_features == aggregations
    values = aggregation_instance.apply_mapping()
    assert len(aggregations) == 3
    closed = Bound.CLOSED
    assert values == [
        SimpleInterval.from_data(1, 4, closed, closed),
        SimpleInterval.from_data(1, 4, closed, closed),
        5,
    ]


def test_feature_extraction_with_aggregation_statistics(example_scenario):
    room = example_scenario
    extractor = FeatureExtractor.from_instances([to_dao(room)])

    agg_features = [f for f in extractor.features if isinstance(f, Call)]
    assert len(agg_features) == 3

    names = {f._name_ for f in agg_features}
    assert any("table" in n for n in names)
    assert any("chair" in n for n in names)

    values = extractor.apply_mapping(to_dao(room))
    assert 1 in values


def test_multiple_exchangeable_parts():
    obj1 = SceneObject(type=SceneObjectType.TABLE)
    obj2 = SceneObject(type=SceneObjectType.CHAIR)
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1, obj2],
    )
    room2 = SceneRoom(
        position=KRROODPosition(1, 1, 1),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1],
    )
    test_ex_parts = TestExParts(objects=[obj1, obj2], rooms=[room, room2])

    extractor = FeatureExtractor.from_instances([to_dao(test_ex_parts)])
    assert len([f for f in extractor.features if isinstance(f, Call)]) == 4
    assert extractor.apply_mapping(to_dao(test_ex_parts)) == [1, 1, 2, 2]


def test_aggregation_count_values(example_scenario):
    room = example_scenario
    aggregation_cls = get_aggregation_class(type(room))
    aggregation_instance = aggregation_cls(instance=room, field_name="objects")
    values = aggregation_instance.apply_mapping()
    assert values[0] == SimpleInterval.from_data(1, 4, Bound.CLOSED, Bound.CLOSED)


def test_only_marked_methods_are_statistics():
    @dataclass
    class Owner:
        items: List[SceneObject]

    @dataclass
    class PartiallyMarkedAggregations(AggregationStatistic[Owner]):
        @aggregation_statistic("items")
        def marked_statistic(self) -> int:
            return 1

        def unmarked_helper(self) -> int:
            return 2

    owner = Owner(items=[SceneObject(type=SceneObjectType.TABLE)])
    instance = PartiallyMarkedAggregations(instance=owner, field_name="items")
    statistic_names = {function.__name__ for function in instance.aggregation_features}
    assert statistic_names == {"marked_statistic"}


def test_aggregation_class_discovered_for_concrete_subclasses():
    assert get_aggregation_class(SceneRoom) is not None
    assert get_aggregation_class(TestExParts) is not None


def test_feature_extraction_over_empty_exchangeable_part_does_not_raise():
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    extractor = FeatureExtractor.from_instances([to_dao(room)])
    assert extractor is not None
    assert all(not isinstance(feature, Call) for feature in extractor.features)


