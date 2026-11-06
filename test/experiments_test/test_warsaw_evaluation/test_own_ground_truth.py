"""
Writing the ground truth of a scene of our own: the objects come from its labelled mesh,
and a person says what each is and which whole it belongs to.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.articulate3d.exceptions import (
    AmbiguousPartFieldError,
    ClassNotInOntologyError,
)
from experiments.warsaw.articulate3d.ground_truth import (
    Expressibility,
    GroundTruthFile,
    SceneGroundTruth,
)
from experiments.warsaw.evaluation.exceptions import (
    DraftAlreadyWrittenError,
    PartHierarchyCycleError,
    UndecidedGroundTruthError,
    UnknownWholeError,
)
from experiments.warsaw.evaluation.own_ground_truth import (
    Decision,
    DraftObject,
    GroundTruthDraft,
    LabelDecision,
    main,
)

from ..test_warsaw_articulate3d.test_ground_truth import (
    PartTwoFieldsAccept,
    WholeWithTwoFieldsForOnePart,
)
from ..test_warsaw_world_loader import write_scene

# %% a scene of our own


@pytest.fixture(scope="module")
def ontology():
    """
    :return: The ontology's classes by name.
    """
    load_annotation_modules()
    return annotation_classes(SemanticAnnotation)


@pytest.fixture
def scene_directory(tmp_path) -> pathlib.Path:
    """
    :return: A scene called kitchen: a cabinet of two faces, two drawers touching it, and a
        drawer standing far off. The cabinet's second face is also labelled as a drawer's.
    """
    directory = tmp_path / "kitchen"
    directory.mkdir()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [9.0, 9.0, 0.0],
            [9.5, 9.0, 0.0],
            [9.0, 9.5, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [5, 6, 7]])
    write_scene(
        directory / "scene.ply",
        vertices,
        faces,
        {"cabinet": [1, 1, 0, 0], "drawer": [0, 1, 1, 2]},
    )
    return directory


def decided(
    objects: dict[str, tuple[str, str]],
    classes: dict[str, str] | None = None,
    scene_id: str = "kitchen",
) -> GroundTruthDraft:
    """
    :param objects: Per object, its label and the whole it was said to belong to.
    :param classes: Per label, the class it was said to mean.
    :return: A draft with those decisions in it.
    """
    classes = classes or {"cabinet": "Cabinet", "drawer": "Drawer"}
    return GroundTruthDraft(
        scene_id=scene_id,
        labels=[LabelDecision(label, one) for label, one in classes.items()],
        objects=[
            DraftObject(name=name, label=label, faces=1, height=0.0, whole=whole)
            for name, (label, whole) in objects.items()
        ],
    )


FACES = {"cabinet_1": [0, 1], "drawer_1": [1, 2], "drawer_2": [3]}


# %% the draft


def test_a_draft_lists_every_object_of_the_scene_as_undecided(scene_directory):
    draft = GroundTruthDraft.of_scene(scene_directory)

    assert [one.name for one in draft.objects] == ["cabinet_1", "drawer_1", "drawer_2"]
    assert {one.whole for one in draft.objects} == {Decision.UNDECIDED}
    assert {one.label: one.semantic_class for one in draft.labels} == {
        "cabinet": Decision.UNDECIDED,
        "drawer": Decision.UNDECIDED,
    }


def test_a_draft_says_what_each_object_meets_so_its_whole_is_easy_to_name(
    scene_directory,
):
    draft = GroundTruthDraft.of_scene(scene_directory)

    meets = {one.name: one.meets for one in draft.objects}
    assert meets["drawer_1"] == ["cabinet_1"]
    assert meets["drawer_2"] == []


def test_a_draft_holds_plain_words_for_a_person_to_overwrite(scene_directory, tmp_path):
    path = tmp_path / "draft.json"

    GroundTruthDraft.of_scene(scene_directory).write(path)

    [first, *_] = json.loads(path.read_text())["objects"]["__items__"]
    assert (first["semantic_class"], first["whole"]) == (
        Decision.AS_ITS_LABEL.value,
        Decision.UNDECIDED.value,
    )


def test_a_draft_is_named_after_the_scene_as_a_run_of_it_is_scored(scene_directory):
    assert GroundTruthDraft.of_scene(scene_directory).scene_id == "kitchen"


# %% building the ground truth


def test_an_object_takes_the_class_of_its_label_and_the_faces_of_the_scene(ontology):
    built = decided(
        {
            "cabinet_1": ("cabinet", Decision.NO_WHOLE),
            "drawer_1": ("drawer", "cabinet_1"),
        }
    ).ground_truth(FACES, ontology)

    assert [(one.part_name, one.semantic_class, one.faces) for one in built.parts] == [
        ("cabinet_1", "Cabinet", [0, 1]),
        ("drawer_1", "Drawer", [1, 2]),
    ]


def test_an_object_can_be_given_another_class_than_its_label(ontology):
    draft = decided(
        {
            "cabinet_1": ("cabinet", Decision.NO_WHOLE),
            "drawer_1": ("drawer", "cabinet_1"),
        }
    )
    draft.objects[1].semantic_class = "Door"

    [_, door] = draft.ground_truth(FACES, ontology).parts

    assert door.semantic_class == "Door"


def test_a_label_can_be_left_unjudged(ontology):
    draft = decided(
        {"cabinet_1": ("cabinet", Decision.NO_WHOLE)},
        classes={"cabinet": Decision.NOT_JUDGED},
    )

    [cabinet] = draft.ground_truth(FACES, ontology).parts

    assert cabinet.semantic_class is None


def test_a_whole_becomes_a_relation_through_the_field_that_holds_it(ontology):
    built = decided(
        {
            "cabinet_1": ("cabinet", Decision.NO_WHOLE),
            "drawer_1": ("drawer", "cabinet_1"),
        }
    ).ground_truth(FACES, ontology)

    [relation] = built.relations
    ids = {one.part_name: one.part_id for one in built.parts}
    assert (relation.whole, relation.part, relation.field_name) == (
        ids["cabinet_1"],
        ids["drawer_1"],
        "drawers",
    )
    assert relation.expressibility is Expressibility.EXPRESSIBLE


def test_an_object_with_no_whole_makes_no_relation(ontology):
    built = decided({"cabinet_1": ("cabinet", Decision.NO_WHOLE)}).ground_truth(
        FACES, ontology
    )

    assert built.relations == []


def test_an_object_removed_from_the_draft_is_left_out(ontology):
    built = decided({"cabinet_1": ("cabinet", Decision.NO_WHOLE)}).ground_truth(
        FACES, ontology
    )

    assert [one.part_name for one in built.parts] == ["cabinet_1"]


# %% what a filled draft is refused for


def test_an_undecided_label_is_refused_by_name(ontology):
    draft = decided(
        {"cabinet_1": ("cabinet", Decision.NO_WHOLE)},
        classes={"cabinet": Decision.UNDECIDED},
    )

    with pytest.raises(UndecidedGroundTruthError) as refused:
        draft.ground_truth(FACES, ontology)

    assert refused.value.undecided == ["the class of label cabinet"]


def test_an_undecided_whole_is_refused_by_name(ontology):
    draft = decided({"cabinet_1": ("cabinet", Decision.UNDECIDED)})

    with pytest.raises(UndecidedGroundTruthError) as refused:
        draft.ground_truth(FACES, ontology)

    assert refused.value.undecided == ["the whole of cabinet_1"]


def test_a_class_the_ontology_does_not_hold_is_refused(ontology):
    draft = decided(
        {"cabinet_1": ("cabinet", Decision.NO_WHOLE)},
        classes={"cabinet": "Cupboardish"},
    )

    with pytest.raises(ClassNotInOntologyError):
        draft.ground_truth(FACES, ontology)


def test_a_whole_that_is_not_listed_is_refused(ontology):
    draft = decided({"drawer_1": ("drawer", "cabinet_9")})

    with pytest.raises(UnknownWholeError):
        draft.ground_truth(FACES, ontology)


def test_objects_holding_one_another_are_refused(ontology):
    draft = decided(
        {"cabinet_1": ("cabinet", "drawer_1"), "drawer_1": ("drawer", "cabinet_1")}
    )

    with pytest.raises(PartHierarchyCycleError):
        draft.ground_truth(FACES, ontology)


def test_a_whole_holding_a_part_through_several_fields_needs_one_named():
    ontology = {"Whole": WholeWithTwoFieldsForOnePart, "Part": PartTwoFieldsAccept}
    draft = decided(
        {"a_1": ("a", Decision.NO_WHOLE), "b_1": ("b", "a_1")},
        classes={"a": "Whole", "b": "Part"},
    )
    faces = {"a_1": [0], "b_1": [1]}

    with pytest.raises(AmbiguousPartFieldError):
        draft.ground_truth(faces, ontology)

    draft.objects[1].field_name = "back"
    [relation] = draft.ground_truth(faces, ontology).relations
    assert relation.field_name == "back"


# %% the command


def test_a_scene_is_drafted_filled_in_and_built(scene_directory, tmp_path):
    draft_path = tmp_path / "kitchen_draft.json"
    main(["draft", str(scene_directory), "--output", str(draft_path)])

    filled = GroundTruthDraft.read(draft_path)
    for label in filled.labels:
        label.semantic_class = label.label.capitalize()
    for one in filled.objects:
        one.whole = "cabinet_1" if one.label == "drawer" else Decision.NO_WHOLE
    filled.write(draft_path)

    output = tmp_path / "ground_truth"
    main(["build", str(scene_directory), str(draft_path), "--output", str(output)])

    built = SceneGroundTruth.read(GroundTruthFile.SUFFIX.path(output, "kitchen"))
    assert {one.part_name: one.faces for one in built.parts} == {
        "cabinet_1": [0, 1],
        "drawer_1": [1, 2],
        "drawer_2": [3],
    }
    assert len(built.relations) == 2


def test_a_draft_already_filled_in_is_not_written_over(scene_directory, tmp_path):
    draft_path = tmp_path / "kitchen_draft.json"
    draft_path.write_text(json.dumps({"hours": "of work"}))

    with pytest.raises(DraftAlreadyWrittenError):
        main(["draft", str(scene_directory), "--output", str(draft_path)])

    assert json.loads(draft_path.read_text()) == {"hours": "of work"}
