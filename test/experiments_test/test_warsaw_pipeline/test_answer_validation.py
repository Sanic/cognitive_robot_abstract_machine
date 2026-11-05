"""
Deciding whether an answer can be acted on before anything acts on it.

An answer naming a class that is not in the ontology is worth nothing, and it costs
nothing to say so while the model is still there to be asked again. Left unchecked it
comes back two steps later as an unmapped label or an unmounted part, where it looks
like the scene's fault rather than the answer's.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    build_taxonomy,
    declares_itself_a_category,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    CountedClaimants,
    LabelAnswer,
    LabelRequest,
    MembershipAnswer,
    MembershipCandidate,
    MembershipQuestion,
    OwnershipAnswer,
    OwnershipQuestion,
    Vocabulary,
)
from experiments.warsaw.pipeline.steps.adjudicate.step import (
    MembershipDecision,
    OwnershipDecision,
)
from experiments.warsaw.pipeline.steps.classify.step import (
    BodyChoiceQuestion,
    BodyGroupQuestion,
)
from experiments.warsaw.pipeline.steps.vocabulary.step import LabelQuestion
from experiments.warsaw.world_loader.loader import RenderedSegmentGroup
from experiments.warsaw.world_loader.scene import LabelSegment

import numpy as np


@pytest.fixture
def label_question(known, taxonomy, tmp_path):
    """
    :return: A question about one label, ready to judge answers to.
    """
    return LabelQuestion(
        label=LabelRequest(label="cabinet", instances=3, exemplar="cabinet_2"),
        every_label=["cabinet", "drawer"],
        taxonomy=taxonomy,
        known=known,
        renders_directory=tmp_path,
    )


@pytest.fixture
def a_category(known) -> str:
    """
    :return: The name of a class the ontology declares a category while Python can still
        instantiate it, which is the case a run has to decide rather than one it cannot.
    """
    return next(
        name
        for name, annotation_class in sorted(known.items())
        if declares_itself_a_category(annotation_class)
        and not inspect.isabstract(annotation_class)
    )


@pytest.fixture
def taxonomy_refusing_categories() -> dict:
    """
    :return: The live ontology as a run refusing categories exports it.
    """
    return build_taxonomy(SemanticAnnotation, categories_are_answers=False)


@pytest.fixture
def taxonomy_accepting_categories() -> dict:
    """
    :return: The live ontology as a run accepting categories exports it.
    """
    return build_taxonomy(SemanticAnnotation, categories_are_answers=True)


@pytest.fixture
def label_question_refusing_categories(known, taxonomy_refusing_categories, tmp_path):
    """
    :return: A question about one label, told that categories are no answer.
    """
    return LabelQuestion(
        label=LabelRequest(label="stairs", instances=32, exemplar="stairs_195"),
        every_label=["stairs", "wall"],
        taxonomy=taxonomy_refusing_categories,
        known=known,
        renders_directory=tmp_path,
    )


# %% what a label may be answered with


def test_a_class_of_the_ontology_is_usable(label_question):
    """
    The ordinary answer names something already there.
    """
    assert label_question.problems_with(LabelAnswer(class_name="Cabinet")) == []


def test_a_label_the_ontology_holds_nothing_for_is_usable(label_question):
    """
    Some labels name nothing the ontology should hold, and saying so is an answer.
    """
    assert label_question.problems_with(LabelAnswer(class_name=None)) == []


def test_a_mixin_given_as_a_class_is_refused(label_question):
    """
    A mixin says what a class can hold, not what something is, so an object cannot be
    one.

    This is the answer that left a real run's faucet unmapped.
    """
    problems = label_question.problems_with(LabelAnswer(class_name="HasDoors"))
    assert len(problems) == 1
    assert "HasDoors" in problems[0]


def test_a_category_given_as_a_class_is_refused(
    label_question_refusing_categories, a_category
):
    """
    A category says what kind of thing something is, not what it is, so a run told to
    refuse them asks again rather than finding out when it annotates.

    This is the answer that left a real run's 32 stairs as Furniture.
    """
    problems = label_question_refusing_categories.problems_with(
        LabelAnswer(class_name=a_category)
    )
    assert len(problems) == 1
    assert a_category in problems[0]


def test_a_new_class_deriving_from_a_category_is_usable(
    label_question_refusing_categories, a_category
):
    """
    Naming the kind, with the category as its superclass, is exactly what a category is
    there for.
    """
    assert (
        label_question_refusing_categories.problems_with(
            LabelAnswer(
                class_name="AKindOfACategoryTheOntologyLacks",
                is_new_class=True,
                superclass=a_category,
            )
        )
        == []
    )


def test_a_category_is_usable_where_the_run_accepts_categories(
    known, taxonomy_accepting_categories, tmp_path, a_category
):
    """
    Whether a category is an answer is the run's setting, and the question follows what
    the model was told rather than deciding it again.
    """
    question = LabelQuestion(
        label=LabelRequest(label="stairs", instances=32, exemplar="stairs_195"),
        every_label=["stairs", "wall"],
        taxonomy=taxonomy_accepting_categories,
        known=known,
        renders_directory=tmp_path,
    )
    assert question.problems_with(LabelAnswer(class_name=a_category)) == []


def test_a_class_that_is_neither_in_the_ontology_nor_proposed_is_refused(
    label_question,
):
    """
    A name nobody can look up and nobody proposed reaches no class at all.
    """
    problems = label_question.problems_with(LabelAnswer(class_name="Whatsit"))
    assert len(problems) == 1
    assert "Whatsit" in problems[0]


def test_a_proposal_of_a_class_that_already_exists_is_refused(label_question):
    """
    Proposing a class the ontology has says the model did not find what was in front of
    it.
    """
    problems = label_question.problems_with(
        LabelAnswer(class_name="Cabinet", is_new_class=True, superclass="Furniture")
    )
    assert any("already in the taxonomy" in one for one in problems)


def test_a_proposal_deriving_from_nothing_is_refused(label_question):
    """
    A class cannot be built from a superclass that is not there.
    """
    problems = label_question.problems_with(
        LabelAnswer(class_name="Gizmo", is_new_class=True, superclass="NotAClass")
    )
    assert any("superclass" in one for one in problems)


def test_a_proposal_composed_from_a_name_that_is_no_mixin_is_refused(label_question):
    """
    The mixins decide what the class can hold, so one that does not exist grants
    nothing.
    """
    problems = label_question.problems_with(
        LabelAnswer(
            class_name="Gizmo",
            is_new_class=True,
            superclass="Table",
            mixins=["NotAMixin"],
        )
    )
    assert any("NotAMixin" in one for one in problems)


def test_a_well_formed_proposal_is_usable(label_question):
    """
    A proposal shaped like the one a real run made for its kitchen island: a name the
    ontology lacks, a superclass it has, and a mixin it has.

    Named for what it stands for rather than after a class, so that the ontology gaining
    one does not turn this into a test of something else.
    """
    assert (
        label_question.problems_with(
            LabelAnswer(
                class_name="AClassTheOntologyLacks",
                is_new_class=True,
                superclass="Table",
                mixins=["HasDrawers"],
            )
        )
        == []
    )


def test_a_class_the_ontology_already_has_is_not_proposed_as_new(label_question):
    """
    Inventing a class beside one that already means the same thing splits a scene's
    vocabulary in two, and the run then names some bodies one and some the other.
    """
    problems = label_question.problems_with(
        LabelAnswer(
            class_name="KitchenIsland",
            is_new_class=True,
            superclass="Table",
            mixins=["HasDrawers"],
        )
    )

    assert any("already in the taxonomy" in one for one in problems)


# %% alternatives a label may be answered with


def test_alternatives_from_the_ontology_are_usable(label_question):
    """
    A label the pictures leave open between two classes names both.
    """
    answer = LabelAnswer(class_name="Cabinet", alternatives=["Drawer"])

    assert label_question.problems_with(answer) == []


def test_an_alternative_the_ontology_lacks_is_refused(label_question):
    answer = LabelAnswer(class_name="Cabinet", alternatives=["Gizmo"])

    assert len(label_question.problems_with(answer)) == 1


def test_a_mixin_given_as_an_alternative_is_refused(label_question):
    answer = LabelAnswer(class_name="Cabinet", alternatives=["HasDoors"])

    assert len(label_question.problems_with(answer)) == 1


def test_a_class_being_proposed_has_no_alternatives(label_question):
    """
    The choice among a label's classes is between classes the ontology already holds.
    """
    proposal = LabelAnswer(
        class_name="Bookcase",
        is_new_class=True,
        superclass="Furniture",
        alternatives=["Cabinet"],
    )

    assert len(label_question.problems_with(proposal)) == 1


# %% a label asked about while new classes are deferred


def test_a_label_asked_while_classes_are_deferred_is_not_given_a_new_one(
    label_question,
):
    """
    A class is proposed for it afterwards, once every label has been read.
    """
    deferring = replace(label_question, proposes_classes=False)
    proposal = LabelAnswer(
        class_name="Bookcase", is_new_class=True, superclass="Furniture"
    )

    assert len(deferring.problems_with(proposal)) == 1


def test_a_label_asked_while_classes_are_deferred_may_be_given_none(label_question):
    deferring = replace(label_question, proposes_classes=False)

    assert deferring.problems_with(LabelAnswer(class_name=None)) == []


# %% what an overlap may be answered with


@pytest.fixture
def ownership_decision(tmp_path):
    """
    :return: An ownership question, ready to judge answers to.
    """
    return OwnershipDecision(
        asked=OwnershipQuestion(
            name="cabinet__drawer",
            pattern=["cabinet", "drawer"],
            shown=["cabinet_8", "drawer_5"],
            covers=[CountedClaimants(claimants=("cabinet_8", "drawer_5"), faces=1503)],
        ),
        labels={"cabinet_8": "cabinet", "drawer_5": "drawer"},
        renders_directory=tmp_path,
    )


def test_an_owner_from_the_pattern_is_usable(ownership_decision):
    """
    The answer picks one of the labels that meet like this.
    """
    assert (
        ownership_decision.problems_with(
            OwnershipAnswer(name="cabinet__drawer", owner="drawer")
        )
        == []
    )


def test_an_owner_that_was_not_offered_is_refused(ownership_decision):
    """
    An answer naming something else settles nothing, and the split would reach no owner.
    """
    problems = ownership_decision.problems_with(
        OwnershipAnswer(name="cabinet__drawer", owner="kitchen_island")
    )
    assert len(problems) == 1
    assert "kitchen_island" in problems[0]


@pytest.fixture
def membership_decision(tmp_path):
    """
    :return: A membership question, ready to judge answers to.
    """
    return MembershipDecision(
        asked=MembershipQuestion(
            name="door_10",
            part="door_10",
            shown=["door_10", "cabinet_4", "cabinet_20"],
            candidates=[
                MembershipCandidate(
                    name="cabinet_4",
                    field_name="doors",
                    shared_faces=3789,
                    touching_edges=5626,
                    distance=0.0,
                ),
                MembershipCandidate(
                    name="cabinet_20",
                    field_name="doors",
                    shared_faces=9,
                    touching_edges=44,
                    distance=0.0,
                ),
            ],
        ),
        labels={"door_10": "door", "cabinet_4": "cabinet", "cabinet_20": "cabinet"},
        renders_directory=tmp_path,
    )


def test_a_whole_from_the_candidates_is_usable(membership_decision):
    """
    The answer picks one of the objects the part was measured to meet.
    """
    assert (
        membership_decision.problems_with(
            MembershipAnswer(name="door_10", part="door_10", whole="cabinet_4")
        )
        == []
    )


def test_a_whole_that_was_not_offered_is_refused(membership_decision):
    """
    A mount named against an object the part never met has no end to hang from.
    """
    problems = membership_decision.problems_with(
        MembershipAnswer(name="door_10", part="door_10", whole="cabinet_99")
    )
    assert len(problems) == 1
    assert "cabinet_99" in problems[0]


# %% what a group of bodies may be answered with


@pytest.fixture
def body_group(taxonomy):
    """
    :return: A question about two painted bodies, ready to judge answers to.
    """
    return BodyGroupQuestion(
        rendered=RenderedSegmentGroup(
            index=0,
            segments=[
                LabelSegment(
                    class_name="drawer",
                    instance=19,
                    face_indices=np.array([0, 1]),
                ),
                LabelSegment(
                    class_name="handle",
                    instance=23,
                    face_indices=np.array([2]),
                ),
            ],
            colors={},
            images={},
        ),
        taxonomy=taxonomy,
        vocabulary=Vocabulary(model="", scene=""),
    )


def test_naming_every_body_in_the_picture_is_usable(body_group):
    """
    The answer names each object once, by the name it was listed under.
    """
    assert (
        body_group.problems_with(
            {
                "drawer_19": BodyAnswer(class_name="Drawer"),
                "handle_23": BodyAnswer(class_name="Handle"),
            }
        )
        == []
    )


def test_a_body_the_answer_says_nothing_about_is_refused(body_group):
    """
    A body left unnamed gets no annotation, and every mount it took part in is dropped.
    """
    problems = body_group.problems_with({"drawer_19": BodyAnswer(class_name="Drawer")})
    assert any("handle_23" in one for one in problems)


def test_a_body_that_was_not_in_the_picture_is_refused(body_group):
    """
    An answer about an object nobody showed cannot reach a body: the names are how a
    mount finds its two ends.
    """
    problems = body_group.problems_with(
        {
            "drawer_19": BodyAnswer(class_name="Drawer"),
            "handle_23": BodyAnswer(class_name="Handle"),
            "sink_1": BodyAnswer(class_name="Sink"),
        }
    )
    assert any("sink_1" in one for one in problems)


def test_a_body_given_no_class_is_refused(body_group):
    """
    A body with no class is one the annotating step will leave alone.
    """
    problems = body_group.problems_with(
        {
            "drawer_19": BodyAnswer(class_name="Drawer"),
            "handle_23": BodyAnswer(class_name=None),
        }
    )
    assert any("handle_23 was given no class" in one for one in problems)


def test_a_body_given_a_category_is_refused(
    body_group, taxonomy_refusing_categories, a_category
):
    """
    The same holds of a body: answered with a category, it would be left unannotated by
    the annotate step, so it is asked about again while the model is still there.
    """
    body_group.taxonomy = taxonomy_refusing_categories
    problems = body_group.problems_with(
        {
            "drawer_19": BodyAnswer(class_name=a_category),
            "handle_23": BodyAnswer(class_name="Handle"),
        }
    )
    assert len(problems) == 1
    assert "drawer_19" in problems[0]
    assert a_category in problems[0]


def test_a_body_given_a_new_class_of_a_category_is_usable(
    body_group, taxonomy_refusing_categories, a_category
):
    """
    Proposing the kind under the category is the answer the category asks for.
    """
    body_group.taxonomy = taxonomy_refusing_categories
    assert (
        body_group.problems_with(
            {
                "drawer_19": BodyAnswer(
                    class_name="AClassTheOntologyLacks",
                    is_new_class=True,
                    superclass=a_category,
                ),
                "handle_23": BodyAnswer(class_name="Handle"),
            }
        )
        == []
    )


# %% choosing among the classes a label was left open between


@pytest.fixture
def choice_group(taxonomy):
    """
    :return: A question about a drawer and a handle whose labels were each left open
        between two classes.
    """
    return BodyChoiceQuestion(
        rendered=RenderedSegmentGroup(
            index=0,
            segments=[
                LabelSegment(
                    class_name="drawer", instance=19, face_indices=np.array([0, 1])
                ),
                LabelSegment(
                    class_name="handle", instance=23, face_indices=np.array([2])
                ),
            ],
            colors={},
            images={},
        ),
        taxonomy=taxonomy,
        vocabulary=Vocabulary(
            model="",
            scene="",
            labels=[
                LabelAnswer(
                    label="drawer", class_name="Drawer", alternatives=["Cabinet"]
                ),
                LabelAnswer(label="handle", class_name="Handle", alternatives=["Knob"]),
            ],
        ),
    )


def test_a_body_given_one_of_its_classes_is_usable(choice_group):
    answer = {
        "drawer_19": BodyAnswer(class_name="Cabinet"),
        "handle_23": BodyAnswer(class_name="Handle"),
    }

    assert choice_group.problems_with(answer) == []


def test_a_body_given_none_of_its_classes_is_usable(choice_group):
    """
    That it is none of them sends it on to be asked without the list.
    """
    answer = {
        "drawer_19": BodyAnswer(class_name=None),
        "handle_23": BodyAnswer(class_name="Knob"),
    }

    assert choice_group.problems_with(answer) == []


def test_a_body_given_a_class_outside_its_choices_is_refused(choice_group):
    answer = {
        "drawer_19": BodyAnswer(class_name="Door"),
        "handle_23": BodyAnswer(class_name="Handle"),
    }

    problems = choice_group.problems_with(answer)

    assert len(problems) == 1
    assert "drawer_19" in problems[0]


# %% a group of bodies asked about while new classes are deferred


def test_a_body_given_a_class_of_its_own_is_refused_while_classes_are_deferred(
    body_group,
):
    deferring = replace(body_group, proposes_classes=False)
    answer = {
        "drawer_19": BodyAnswer(class_name="Drawer"),
        "handle_23": BodyAnswer(class_name="Knobby", is_new_class=True),
    }

    problems = deferring.problems_with(answer)

    assert len(problems) == 1
    assert "handle_23" in problems[0]


def test_a_body_given_no_class_is_usable_while_classes_are_deferred(body_group):
    """
    None of the ontology's classes being what it is, it is left without one.
    """
    deferring = replace(body_group, proposes_classes=False)
    answer = {
        "drawer_19": BodyAnswer(class_name="Drawer"),
        "handle_23": BodyAnswer(class_name=None),
    }

    assert deferring.problems_with(answer) == []
