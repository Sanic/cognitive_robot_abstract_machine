"""
The shapes one step of the pipeline hands to the next.

A step writes a file and the step after it reads it, so the two only agree because one
declaration serves both. What is checked here is that the declaration still reads
everything a real run wrote, and writes back exactly what it read: a field quietly
dropped in the reading is a decision quietly dropped from the run.
"""

from __future__ import annotations

import json

import pytest

from experiments.warsaw.pipeline.records import (
    Adjudications,
    BodyAnswer,
    Classifications,
    LabelAnswer,
    OpenQuestions,
    Relations,
    RelationStatus,
    SplitRecord,
    Vocabulary,
    VocabularyRequest,
)
from experiments.warsaw.world_loader.viewpoints import PictureKind
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.scene_split import Pairing

READERS = {
    RunFile.RELATIONS: Relations,
    RunFile.VOCABULARY_REQUEST: VocabularyRequest,
    RunFile.VOCABULARY: Vocabulary,
    RunFile.QUESTIONS: OpenQuestions,
    RunFile.ADJUDICATIONS: Adjudications,
    RunFile.SPLIT: SplitRecord,
    RunFile.CLASSIFICATIONS: Classifications,
}
"""
Which type reads each file a run writes.
"""

# %% reading back what a run actually wrote


@pytest.mark.parametrize("run_file", sorted(READERS, key=lambda one: one.value))
def test_a_run_s_own_file_round_trips_unchanged(finished_run, run_file):
    """
    Reading a file and writing it again gives the same file, so nothing in it is lost on
    the way from the step that wrote it to the step that acts on it.
    """
    written = json.loads((finished_run.directory / run_file.value).read_text())
    read_back = READERS[run_file].from_json(written).to_json()
    assert json.loads(json.dumps(read_back)) == written


def test_the_measurement_names_every_segment_it_measured(relations):
    """
    The label a body carries is looked up by the segment's name in three later steps.
    """
    assert (
        relations.labels[relations.segments[0].name] == relations.segments[0].class_name
    )
    assert set(relations.descriptors) == {one.name for one in relations.segments}


def test_a_pair_carries_both_the_measurement_and_what_the_ontology_makes_of_it(
    relations,
):
    """
    The measurement is the scan's, the status is the ontology's, and the split needs
    both.
    """
    overlapping = next(one for one in relations.pairs if one.evidence.shared_faces)
    assert {overlapping.evidence.one, overlapping.evidence.other} <= set(
        relations.descriptors
    )
    assert overlapping.status in set(RelationStatus)


# %% what an answer is read as


def test_an_answer_written_as_a_bare_name_is_still_an_answer():
    """
    A mapping written by hand to try something out names the class and nothing else.
    """
    assert LabelAnswer.of("Drawer") == LabelAnswer(class_name="Drawer")
    assert LabelAnswer.of(None) == LabelAnswer(class_name=None)


def test_an_answer_naming_nothing_is_not_usable():
    """
    A label the ontology should hold nothing for maps to no class.
    """
    assert not LabelAnswer(class_name=None).is_usable


def test_an_answer_with_a_problem_is_not_usable():
    """
    An answer naming a class that cannot be used is worth no more than none.
    """
    assert not LabelAnswer(class_name="HasHandle", problems=["it is a mixin"]).is_usable


def test_a_body_s_answer_keeps_only_what_the_pipeline_reads():
    """
    A model's extra fields do not reach the file.
    """
    answer = BodyAnswer.from_json(
        {"class_name": "Drawer", "confidence": 0.9, "invented_by_the_model": "ignored"}
    )
    assert "invented_by_the_model" not in answer.to_json()
    assert answer.class_name == "Drawer"


# %% the adjudication's two kinds of answer


def test_the_two_kinds_of_answer_are_read_apart(adjudications):
    """
    Ownership decides whose a face is; membership decides which whole a part is in.

    They are answered in one list and acted on in two different places, so the two the
    file is read back through have to divide it between them: an answer neither reaches
    is one the split never applies.
    """
    assert adjudications.ownership and adjudications.membership
    assert len(adjudications.ownership) + len(adjudications.membership) == len(
        adjudications.answered
    )


def test_an_ownership_answer_is_looked_up_by_the_pattern_it_answers(adjudications):
    """
    One answer is given per class pattern and applied everywhere that pattern occurs.
    """
    answer = adjudications.ownership[0]
    assert adjudications.owner_by_pattern[tuple(answer.pattern)] == answer.owner


def test_the_sets_the_ontology_settles_are_carried_to_the_split(relations):
    """
    The split asks whether a set of claimants was settled, so it is carried as names.
    """
    assert relations.settled_claimants
    assert all(isinstance(one, tuple) for one in relations.settled_claimants)


# %% the mounts carried past the split


def test_a_pairing_round_trips_through_its_file():
    """
    A pairing is written to the split's file and read back out of it by the step that
    mounts it, so the two have to mean the same thing.
    """
    pairing = Pairing(whole="cabinet_5", part="door_11", field_name="doors")
    assert Pairing.from_json(pairing.to_json()) == pairing


def test_the_split_s_pairings_name_bodies_it_built(split):
    """
    A mount needs both ends to be bodies, which is what the split checked before
    writing.
    """
    built = {body.name for body in split.bodies}
    for pairing in split.pairings:
        assert pairing.whole in built
        assert pairing.part in built


# %% what a render's name says it shows


@pytest.mark.parametrize(
    "filename, shows",
    [
        ("cabinet__closeup_front_left.png", PictureKind.CLOSEUP),
        ("cabinet__drawer__context_back_right.png", PictureKind.CONTEXT),
        ("jar__plain_front_right.png", PictureKind.PLAIN),
    ],
)
def test_a_render_says_what_it_shows(filename, shows):
    """
    A question is put with its pictures captioned, and the caption comes from the name.
    """
    assert PictureKind.of_render(filename) is shows


def test_a_render_whose_name_says_nothing_is_not_guessed_at():
    """
    A file that is not one of the run's renders captions nothing.
    """
    assert PictureKind.of_render("something_else.png") is None


# %% the records that used to be keyed by name


def test_every_body_of_a_split_knows_its_own_name(split):
    """
    The split's bodies were a mapping keyed by name and are now a list, so each one has
    to carry the name it was keyed by; a body that lost it is one nothing can mount
    into.
    """
    names = [body.name for body in split.bodies]
    assert names
    assert all(names)
    assert len(set(names)) == len(names)


def test_every_answered_label_knows_which_label_it_answers(vocabulary):
    """
    An answer with no label is one no segment can be read through.
    """
    assert vocabulary.labels
    assert all(answer.label for answer in vocabulary.labels)
    assert len(vocabulary.by_label) == len(vocabulary.labels)


def test_every_named_body_knows_which_body_it_names(classifications):
    """
    The annotation step walks these to find the body each answer is about.
    """
    names = [answer.name for answer in classifications.bodies]
    assert names
    assert all(names)
    assert len(set(names)) == len(names)


def test_a_question_s_measurements_and_shares_name_what_they_measured(questions):
    """
    Both were mappings keyed by segment, and both are read out beside each other in the
    question put to a model, so each entry has to say which segment it is about.
    """
    asked = questions.ownership[0]
    assert {one.name for one in asked.measured} == set(asked.shown)
    assert {one.name for one in asked.shares} == set(asked.shown)


def test_a_membership_question_names_what_it_is_choosing_between(questions):
    """
    An answer is checked against these names, so they are what the question offers.
    """
    asked = questions.membership[0]
    assert asked.candidate_names == [one.name for one in asked.candidates]
    assert len(asked.candidate_names) > 1
