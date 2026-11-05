"""
Working out how few questions a scene's overlaps actually need.

Two hundred overlapping pairs are not two hundred questions. A face belongs to one
object, so ownership is asked once per set of claimants; a set whose every internal pair
the ontology settled is not a question at all; and what is left repeats, so it is asked
once per pattern of classes. Membership is asked only where a part meets more than one
candidate.
"""

from __future__ import annotations

import pytest


from experiments.warsaw.pipeline.records import PairRecord, Relations, RelationStatus
from experiments.warsaw.pipeline.run import Run
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.evidence import MeasureScene
from experiments.warsaw.segment_relations import (
    PairEvidence,
    SegmentDescriptor,
    SegmentRelations,
)


@pytest.fixture
def step(tmp_path):
    """
    :return: The measuring step, for what it works out rather than what it renders.
    """
    return MeasureScene(settings=PipelineSettings(), run=Run(directory=tmp_path))


@pytest.fixture
def measured(relations) -> SegmentRelations:
    """
    :return: What was measured of a real scene, as the step holds it.
    """
    return SegmentRelations(
        descriptors=relations.descriptors,
        pairs=[one.evidence for one in relations.pairs],
    )


# %% what the ontology admits between two classes


def test_a_cabinet_can_hold_a_drawer_and_only_one_way(step, known):
    """
    Exactly one admissible relation leaves only the pair itself in question.
    """
    view = step.ontology_view(known["Cabinet"], known["Drawer"])
    assert view.status is RelationStatus.RELATION_KNOWN
    assert view.mounts[0].field_name == "drawers"
    assert view.mounts[0].whole == "Cabinet"


def test_two_classes_that_cannot_hold_one_another_leave_an_overlap_unexplained(
    step, known
):
    """
    An overlap between them is something other than a part, and the adjudication has to
    be told so rather than being asked which of them is the part.
    """
    assert (
        step.ontology_view(known["Cabinet"], known["Wall"]).status
        is RelationStatus.NO_LEGAL_RELATION
    )


def test_an_unmapped_label_leaves_the_ontology_with_nothing_to_say(step, known):
    """
    Before the vocabulary is answered, no pair has a status at all.
    """
    assert (
        step.ontology_view(None, known["Drawer"]).status is RelationStatus.CLASS_UNKNOWN
    )


def test_storing_one_in_the_other_is_not_admitted(step, known):
    """
    A drawer could be stored in a cabinet as well as be part of it, and only the part is
    a relation the pipeline mounts.
    """
    view = step.ontology_view(known["Cabinet"], known["Drawer"])
    assert [one.field_name for one in view.mounts] == ["drawers"]


# %% which object stands for a label


def test_the_object_shown_for_a_label_is_the_one_least_claimed_by_others(
    step, measured
):
    """
    Ranking by the *share* instead picks slivers, which are wholly unclaimed precisely
    because they are fragments: on one scene it offered a strip of 545 faces as the
    example of a cabinet while a whole cabinet front of 960 stood beside it.
    """
    for label, name in step.exemplars(measured).items():
        of_this_label = [
            one for one in measured.descriptors.values() if one.class_name == label
        ]
        assert measured.descriptors[name].exclusive_area == max(
            one.exclusive_area for one in of_this_label
        )


def test_every_label_of_the_scene_is_shown_by_something(step, measured):
    """
    A label nobody pictures is a label nobody can be asked about.
    """
    assert set(step.exemplars(measured)) == {
        one.class_name for one in measured.descriptors.values()
    }


# %% which whole a part belongs to


def test_a_part_meeting_one_candidate_is_not_asked_about(step, relations, known):
    """
    A drawer that meets exactly one cabinet has nothing to choose between.
    """
    memberships = step.memberships(
        relations,
        relations.labels,
        self_classes(relations, known),
        self_measured(relations),
    )
    asked = {one.part for one in memberships.to_ask_about}
    assert asked.isdisjoint({one.part for one in memberships.forced})


def test_a_part_meeting_several_candidates_is_asked_about(step, relations, known):
    """
    Which of them it sits in is what the pictures are for.
    """
    memberships = step.memberships(
        relations,
        relations.labels,
        self_classes(relations, known),
        self_measured(relations),
    )
    for question in memberships.to_ask_about:
        assert len(question.candidates) > 1
        assert question.shown[0] == question.part


def test_a_part_that_meets_a_whole_is_offered_only_wholes_it_meets(
    step, relations, known
):
    """
    A whole a part shares faces with or touches is where it is mounted, so what is
    merely nearby is offered only to a part that meets none.
    """
    memberships = step.memberships(
        relations,
        relations.labels,
        self_classes(relations, known),
        self_measured(relations),
    )
    for one in memberships.to_ask_about:
        meets = [c for c in one.candidates if c.shared_faces or c.touching_edges]
        assert meets == [] or meets == one.candidates


def test_a_pair_the_ontology_leaves_open_names_no_membership(step, relations, known):
    """
    Which field would hold the part is not known until exactly one relation is, so a
    pair the ontology has not settled cannot say where a mount would go.
    """
    memberships = step.memberships(
        relations,
        relations.labels,
        self_classes(relations, known),
        self_measured(relations),
    )
    known_pairs = {
        tuple(sorted((one.evidence.one, one.evidence.other)))
        for one in relations.pairs
        if one.status is RelationStatus.RELATION_KNOWN
    }
    for one in memberships.forced:
        assert tuple(sorted((one.part, one.whole))) in known_pairs


# %% a part that meets no whole


def blinds_and_rails(step, known, *meetings):
    """
    :param meetings: Per rail, how many edges the blinds touch it along and how far
        apart the two are.
    :return: The pairs between a pair of blinds and those rails, and every label.
    """
    pairs = [
        PairRecord(
            evidence=PairEvidence(
                one=rail,
                other="blinds_1",
                shared_faces=0,
                share_of_one=0.0,
                share_of_other=0.0,
                touching_edges=edges,
                distance=distance,
            ),
            classes={"rail": "Rail", "blinds": "Blinds"},
            view=step.ontology_view(known["Rail"], known["Blinds"]),
        )
        for rail, edges, distance in meetings
    ]
    labels = {"blinds_1": "blinds", **{rail: "rail" for rail, _, _ in meetings}}
    classes = {"rail": known["Rail"], "blinds": known["Blinds"]}
    return Relations(scene="a room", pairs=pairs), labels, classes


def described(name: str, label: str) -> SegmentDescriptor:
    """
    :return: A segment measured to be a small square of scan.
    """
    return SegmentDescriptor(
        name=name,
        class_name=label,
        instance=1,
        faces=2,
        exclusive_faces=2,
        area=0.01,
        exclusive_area=0.01,
        centroid=(0.0, 0.0, 1.0),
        minimum_corner=(0.0, 0.0, 1.0),
        maximum_corner=(0.1, 0.1, 1.0),
        components=1,
        height=1.0,
    )


def test_the_whole_touched_along_the_most_edges_is_offered_first(step, known):
    """
    A model reads the candidates in the order they are listed and paints them in that
    order, so the order follows the contact rather than the names.
    """
    relations, labels, classes = blinds_and_rails(
        step, known, ("rail_1", 3, 0.0), ("rail_2", 12, 0.0), ("rail_3", 7, 0.0)
    )
    relations = Relations(
        scene=relations.scene,
        pairs=relations.pairs,
        segments=[described(name, label) for name, label in labels.items()],
    )

    [question] = step.memberships(
        relations, labels, classes, self_measured(relations)
    ).to_ask_about

    assert [one.name for one in question.candidates] == ["rail_2", "rail_3", "rail_1"]
    assert question.shown == ["blinds_1", "rail_2", "rail_3", "rail_1"]


def test_a_part_that_meets_no_whole_is_given_one_within_the_gap(step, known):
    """
    A scan leaves gaps between what is mounted together, as between blinds and the rail
    they hang from.
    """
    gap = step.settings.largest_gap_to_a_whole
    relations, labels, classes = blinds_and_rails(step, known, ("rail_1", 0, gap / 2))

    memberships = step.memberships(relations, labels, classes, self_measured(relations))

    assert [(one.part, one.whole) for one in memberships.forced] == [
        ("blinds_1", "rail_1")
    ]


def test_a_whole_past_the_gap_is_not_offered(step, known):
    gap = step.settings.largest_gap_to_a_whole
    relations, labels, classes = blinds_and_rails(step, known, ("rail_1", 0, gap * 2))

    memberships = step.memberships(relations, labels, classes, self_measured(relations))

    assert (memberships.forced, memberships.to_ask_about) == ([], [])


def test_a_part_that_meets_a_whole_is_not_offered_a_nearby_one(step, known):
    gap = step.settings.largest_gap_to_a_whole
    relations, labels, classes = blinds_and_rails(
        step, known, ("rail_1", 12, 0.0), ("rail_2", 0, gap / 2)
    )

    memberships = step.memberships(relations, labels, classes, self_measured(relations))

    assert [(one.part, one.whole) for one in memberships.forced] == [
        ("blinds_1", "rail_1")
    ]


def self_classes(relations: Relations, known):
    """
    :param relations: What the measurement wrote.
    :param known: The ontology's classes by name.
    :return: Per label, the class the file says it was read as.
    """
    read_as = {}
    for pair in relations.pairs:
        for label, class_name in pair.classes.items():
            read_as[label] = known.get(class_name)
    return read_as


def self_measured(relations: Relations) -> SegmentRelations:
    """
    :param relations: What the measurement wrote.
    :return: It, as the step holds it.
    """
    return SegmentRelations(
        descriptors=relations.descriptors,
        pairs=[one.evidence for one in relations.pairs],
    )
