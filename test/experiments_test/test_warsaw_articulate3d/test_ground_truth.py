"""
Building a scene's ground truth from its annotation and the protocol.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from semantic_digital_twin.semantic_annotations.part_whole import field_holding

from experiments.warsaw.articulate3d.exceptions import (
    AmbiguousPartFieldError,
    ClassNotInOntologyError,
)
from experiments.warsaw.articulate3d.ground_truth import (
    Expressibility,
    GroundTruthBuilder,
    GroundTruthRelation,
    SceneGroundTruth,
)
from experiments.warsaw.articulate3d.protocol import (
    EvaluationProtocol,
    FieldChoice,
    PartClass,
    PartNameMeaning,
    PartNameRuling,
)

from .scenes import (
    PartTwoFieldsAccept,
    SceneName,
    WholeWithTwoFieldsForOnePart,
    kitchen_corner_parts,
    kitchen_corner_protocol,
    part,
)

# %% finding one relation


def relation(built, whole, part_id):
    """
    :return: The relation holding one part in a whole.
    """
    [found] = [
        one for one in built.relations if (one.whole, one.part) == (whole, part_id)
    ]
    return found


# %% which parts are covered


def test_parts_without_faces_or_of_an_ignored_name_are_left_out(ground_truth):
    assert [one.part_id for one in ground_truth.parts] == [1, 2, 3, 4, 5, 6, 7]


def test_every_covered_part_keeps_the_faces_it_is_annotated_with(ground_truth):
    annotated = {
        one.part_id: one.face_indices.tolist() for one in kitchen_corner_parts()
    }

    assert {one.part_id: one.faces for one in ground_truth.parts} == {
        one.part_id: annotated[one.part_id] for one in ground_truth.parts
    }


def test_a_part_is_judged_against_the_class_the_protocol_gives_it(ground_truth):
    protocol = kitchen_corner_protocol()

    assert {one.part_id: one.semantic_class for one in ground_truth.parts} == {
        one.part_id: protocol.class_of(
            SceneName.KITCHEN_CORNER, one.part_id, one.part_name
        )
        for one in ground_truth.parts
    }


# %% the part hierarchy


def test_every_part_is_related_to_the_part_above_it(ground_truth):
    assert {(one.whole, one.part) for one in ground_truth.relations} == {
        (1, 2),
        (2, 3),
        (1, 4),
        (4, 5),
        (6, 7),
    }


def test_a_relation_is_held_through_the_field_the_ontology_holds_it_in(
    ground_truth, ontology
):
    door_handle = relation(ground_truth, 2, 3)

    assert door_handle.expressibility is Expressibility.EXPRESSIBLE
    assert (
        door_handle.field_name
        == field_holding(ontology["Door"], ontology["Handle"]).field_name
    )


def test_a_relation_with_an_end_whose_class_is_not_judged_is_undetermined(ground_truth):
    assert relation(ground_truth, 6, 7) == GroundTruthRelation(
        whole=6, part=7, expressibility=Expressibility.UNDETERMINED
    )


def test_a_relation_no_field_can_hold_is_inexpressible(ontology):
    protocol = replace(
        kitchen_corner_protocol(),
        part_classes=kitchen_corner_protocol().part_classes
        + [
            PartClass(
                scene_id=SceneName.KITCHEN_CORNER, part_id=7, semantic_class="Cabinet"
            )
        ],
    )

    built = GroundTruthBuilder(protocol=protocol, ontology=ontology).build(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts()
    )

    assert relation(built, 6, 7).expressibility is Expressibility.INEXPRESSIBLE


# %% a whole that could hold a part through several fields


@pytest.fixture
def two_fields():
    """
    :return: An ontology whose whole could hold its part through either of two fields,
        and a protocol naming both.
    """
    ontology = {
        "Whole": WholeWithTwoFieldsForOnePart,
        "Part": PartTwoFieldsAccept,
    }
    protocol = EvaluationProtocol(
        part_names=[
            PartNameMeaning(
                part_name="whole", ruling=PartNameRuling.CLASS, semantic_class="Whole"
            ),
            PartNameMeaning(
                part_name="part", ruling=PartNameRuling.CLASS, semantic_class="Part"
            ),
        ]
    )
    return ontology, protocol


def test_several_fields_and_no_choice_are_refused(two_fields):
    ontology, protocol = two_fields

    with pytest.raises(AmbiguousPartFieldError):
        GroundTruthBuilder(protocol=protocol, ontology=ontology).build(
            "two_fields", [part(1, "whole.1.1", [0]), part(2, "part.1.1.2_1", [1])]
        )


def test_several_fields_take_the_one_the_protocol_chooses(two_fields):
    ontology, protocol = two_fields
    choice = FieldChoice(whole_class="Whole", part_class="Part", field_name="back")
    protocol = replace(protocol, field_choices=[choice])

    built = GroundTruthBuilder(protocol=protocol, ontology=ontology).build(
        "two_fields", [part(1, "whole.1.1", [0]), part(2, "part.1.1.2_1", [1])]
    )

    assert relation(built, 1, 2).field_name == choice.field_name


# %% the protocol against the ontology


def test_a_protocol_naming_a_class_the_ontology_lacks_is_refused(ontology):
    protocol = replace(
        kitchen_corner_protocol(),
        part_classes=[
            PartClass(
                scene_id=SceneName.KITCHEN_CORNER, part_id=6, semantic_class="Switch"
            )
        ],
    )

    with pytest.raises(ClassNotInOntologyError):
        GroundTruthBuilder(protocol=protocol, ontology=ontology)


# %% the ground truth file


def test_the_ground_truth_reads_back_as_it_was_written(ground_truth, tmp_path):
    ground_truth.write(tmp_path / "ground_truth.json")

    assert SceneGroundTruth.read(tmp_path / "ground_truth.json") == ground_truth
