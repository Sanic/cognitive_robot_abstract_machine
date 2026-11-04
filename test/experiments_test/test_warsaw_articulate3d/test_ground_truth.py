"""
Building a scene's ground truth from its annotation and the protocol.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from semantic_digital_twin.semantic_annotations.part_whole import (
    field_holding,
    fields_accepting,
)

from experiments.warsaw.articulate3d.exceptions import (
    AmbiguousPartFieldError,
    AmbiguousWholeError,
    ClassNotInOntologyError,
)
from experiments.warsaw.articulate3d.ground_truth import (
    Expressibility,
    GroundTruthBuilder,
    GroundTruthFile,
    GroundTruthRelation,
    SceneGroundTruth,
    main,
)
from experiments.warsaw.articulate3d.protocol import (
    EvaluationProtocol,
    FieldChoice,
    PartClass,
    PartNameMeaning,
    PartNameRuling,
)

from experiments.warsaw.articulate3d.rules import (
    AnnotationRules,
    MeshMissing,
    MeshSize,
    RuleFindings,
    PartNameCorrection,
)

from .scenes import (
    PartTwoFieldsAccept,
    SceneName,
    WholeWithTwoFieldsForOnePart,
    kitchen_corner_parts,
    kitchen_corner_protocol,
    part,
    row_of_squares,
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


def test_a_relation_held_the_other_way_round_by_a_carrying_field_is_carried(ontology):
    protocol = EvaluationProtocol(
        part_names=[
            PartNameMeaning(
                part_name="phone cable",
                ruling=PartNameRuling.CLASS,
                semantic_class="Cable",
            ),
            PartNameMeaning(
                part_name="phone",
                ruling=PartNameRuling.CLASS,
                semantic_class="Telephone",
            ),
        ]
    )
    [carrying] = [
        one
        for one in fields_accepting(ontology["Telephone"], ontology["Cable"])
        if one.part_carries_whole
    ]

    built = GroundTruthBuilder(protocol=protocol, ontology=ontology).build(
        "hallway", [part(1, "phone cable.1.1", [0]), part(2, "phone.1.1.2_1", [1])]
    )

    assert (relation(built, 1, 2).expressibility, relation(built, 1, 2).field_name) == (
        Expressibility.CARRIED,
        carrying.field_name,
    )


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
            "two_fields",
            [part(1, "whole.1.1", [0]), part(2, "part.1.1.2_1", [1])],
        )


def test_several_fields_take_the_one_the_protocol_chooses(two_fields):
    ontology, protocol = two_fields
    choice = FieldChoice(whole_class="Whole", part_class="Part", field_name="back")
    protocol = replace(protocol, field_choices=[choice])

    built = GroundTruthBuilder(protocol=protocol, ontology=ontology).build(
        "two_fields",
        [part(1, "whole.1.1", [0]), part(2, "part.1.1.2_1", [1])],
    )

    assert relation(built, 1, 2).field_name == choice.field_name


# %% the protocol against the ontology


def test_a_protocol_naming_a_class_the_ontology_lacks_is_refused(ontology):
    protocol = replace(
        kitchen_corner_protocol(),
        part_classes=[
            PartClass(
                scene_id=SceneName.KITCHEN_CORNER,
                part_id=6,
                semantic_class="NoClassOfTheOntology",
            )
        ],
    )

    with pytest.raises(ClassNotInOntologyError):
        GroundTruthBuilder(protocol=protocol, ontology=ontology)


# %% a whole the hierarchy does not settle


def test_a_part_below_a_place_two_parts_share_is_refused(ontology):
    parts = kitchen_corner_parts() + [part(10, "drawer.1.1.2_1", [14])]

    with pytest.raises(AmbiguousWholeError):
        GroundTruthBuilder(protocol=kitchen_corner_protocol(), ontology=ontology).build(
            SceneName.KITCHEN_CORNER, parts
        )


# %% the ground truth file


def test_the_ground_truth_reads_back_as_it_was_written(ground_truth, tmp_path):
    ground_truth.write(tmp_path / "ground_truth.json")

    assert SceneGroundTruth.read(tmp_path / "ground_truth.json") == ground_truth


# %% building the ground truth of a dataset


@pytest.fixture
def built_dataset(dataset, tmp_path):
    """
    :return: The directory the ground truth of the kitchen corner and of a scene
        without a mesh is written to, by the kitchen corner's protocol with a mesh rule
        and one correction.
    """
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    dataset.write_scene(
        "hallway",
        [part(1, "door.1.1", [0])],
        mesh=None,
        recorded_mesh_size=MeshSize(vertices=3, faces=1),
    )
    protocol_path = tmp_path / "protocol.json"
    replace(
        kitchen_corner_protocol(),
        annotation_rules=renamed_part_rules(),
    ).write(protocol_path)
    output = tmp_path / "ground_truth"
    main(
        [
            "--dataset",
            str(dataset.articulate3d_directory),
            "--scannetpp",
            str(dataset.scannetpp_directory),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output),
        ]
    )
    return output


def renamed_part_rules():
    """
    :return: The mesh rule, and the part of the ignored name renamed a handle.
    """
    return AnnotationRules(
        exclusions=[MeshMissing()],
        corrections=[
            PartNameCorrection(
                written="unknown", meant="handle", mistake="a spelling mistake"
            )
        ],
    )


def test_a_scene_the_rules_leave_out_gets_no_ground_truth(built_dataset):
    assert sorted(path.name for path in built_dataset.iterdir()) == sorted(
        [
            GroundTruthFile.SUFFIX.path(built_dataset, SceneName.KITCHEN_CORNER).name,
            GroundTruthFile.RULE_FINDINGS.value,
        ]
    )


def test_a_corrected_part_is_judged_by_its_corrected_name(built_dataset):
    built = SceneGroundTruth.read(
        GroundTruthFile.SUFFIX.path(built_dataset, SceneName.KITCHEN_CORNER)
    )

    assert [one.semantic_class for one in built.parts if one.part_id == 9] == [
        kitchen_corner_protocol().meaning_of("handle").semantic_class
    ]


def test_every_rule_that_applied_is_written_beside_the_ground_truth(built_dataset):
    findings = RuleFindings.from_json(
        json.loads((built_dataset / GroundTruthFile.RULE_FINDINGS).read_text())
    )

    assert (
        [(one.scene_id, one.rule) for one in findings.exclusions],
        [(one.scene_id, one.rule) for one in findings.corrections],
    ) == (
        [("hallway", MeshMissing())],
        [(SceneName.KITCHEN_CORNER, renamed_part_rules().corrections[0])],
    )
