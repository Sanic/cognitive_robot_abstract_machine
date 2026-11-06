"""
What each question actually puts to a model.

The words are the experiment. A run's answers can only be compared against another run's
if both were asked the same thing, so what these templates render is pinned against what
the run in the fixtures was asked, byte for byte, rather than against a description of
it.

The pictures are left out: the fixture run's renders are not kept, and what is under
test is the text.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import pytest
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    TextPart,
)
from typing_extensions import List

from experiments.warsaw.painting import PaintColor
from experiments.warsaw.pipeline.records import AmendmentRecord
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.amend.step import MixinProposal
from experiments.warsaw.pipeline.steps.adjudicate.step import (
    MembershipDecision,
    OwnershipDecision,
)
from experiments.warsaw.pipeline.steps.classify.step import BodyGroupQuestion
from experiments.warsaw.world_loader.loader import RenderedSegmentGroup
from experiments.warsaw.pipeline.steps.vocabulary.step import (
    LabelQuestion,
    MapLabelVocabulary,
)
from experiments.warsaw.pipeline.asking import PromptHalf


@pytest.fixture
def without_pictures(monkeypatch) -> None:
    """
    Read every render as nothing, since the fixture run does not keep them.
    """
    monkeypatch.setattr(
        ImagePart, "from_file", classmethod(lambda cls, path: cls(image=b""))
    )


@pytest.fixture
def expected(dataset):
    """
    :return: A reader for what a question said before it was written from a template.
    """

    def read(name: str) -> str:
        return (dataset / "expected" / f"message_{name}.md").read_text()

    return read


def said(parts: List[MessagePart]) -> str:
    """
    :param parts: The message as it is put to a model.
    :return: Everything in it that is words.
    """
    return "\n\n".join(one.text for one in parts if isinstance(one, TextPart))


# %% which class a label means


def test_the_vocabulary_question_says_what_it_always_said(
    finished_run, vocabulary_request, relations, taxonomy, expected, without_pictures
):
    """
    The ontology, the label, and what the object meets.

    The fixture run keeps no renders, so this label carries none and the question is the
    one a run asking from the text alone puts: it does not announce pictures it has not
    got. What a question with renders adds is checked in
    ``test_asking_without_pictures``.
    """
    label = next(
        one for one in vocabulary_request.labels if one.label == "kitchen_island"
    )
    question = LabelQuestion(
        label=label,
        every_label=vocabulary_request.label_names,
        taxonomy=taxonomy,
        known={},
        renders_directory=finished_run.path(RunFile.EXEMPLARS),
        meets=MapLabelVocabulary.meetings(relations, label.exemplar),
    )
    assert said(question.message()) == expected("vocabulary")


# %% whose the contested faces are


def test_the_ownership_question_says_what_it_always_said(
    finished_run, questions, relations, expected, without_pictures
):
    """
    Including the shares, which are what the picture cannot say.
    """
    question = OwnershipDecision(
        asked=questions.ownership[0],
        labels=relations.labels,
        renders_directory=finished_run.path(RunFile.QUESTION_RENDERS),
    )
    assert said(question.message()) == expected("ownership")


def test_the_membership_question_says_what_it_always_said(
    finished_run, questions, relations, expected, without_pictures
):
    """
    Including every candidate and how the part was measured to meet it.
    """
    question = MembershipDecision(
        asked=questions.membership[0],
        labels=relations.labels,
        renders_directory=finished_run.path(RunFile.QUESTION_RENDERS),
    )
    assert said(question.message()) == expected("membership")


# %% what each body is


@dataclass
class Segment:
    """
    Stands in for one of the bodies painted into a group render.
    """

    name: str
    """
    What the body is called.
    """

    class_name: str
    """
    The label the scan gave it.
    """


def test_the_classification_question_says_what_it_always_said(
    classifications, vocabulary, taxonomy, expected, without_pictures
):
    """
    The ontology, then each painted body with its colour, its label, and what that label
    was read as.
    """
    group = [
        Segment(name=one.name, class_name=one.label or "")
        for one in classifications.bodies[:8]
    ]
    question = BodyGroupQuestion(
        rendered=RenderedSegmentGroup(
            index=0,
            segments=group,
            colors={one.name: PaintColor.BLUE for one in group},
            images={},
        ),
        taxonomy=taxonomy,
        vocabulary=vocabulary,
    )
    assert said(question.message()) == expected("classification")


# %% whether a class of the ontology is missing a structural part


def test_the_amendment_question_says_what_it_always_said(
    finished_run, vocabulary_request, taxonomy, known, expected, without_pictures
):
    """
    Read against a proposal made up for the purpose: this question asks for a change to
    the ontology's own source, so what it puts to a model is worth pinning exactly.
    """
    question = MixinProposal(
        record=AmendmentRecord(
            whole="CounterTop",
            part="Drawer",
            mixin="HasDrawers",
            whole_labels=["countertop"],
            part_labels=["drawer"],
            measured_pairs=7,
            shared_faces=1234,
        ),
        known=known,
        taxonomy=taxonomy,
        request=vocabulary_request,
        renders_directory=finished_run.path(RunFile.EXEMPLARS),
    )
    assert said(question.message()) == expected("amendment")


def test_every_question_is_put_with_both_halves_of_its_own_prompt():
    """
    A question names one prompt and both halves are found from that name, in the
    directory of the step that puts it.

    Naming one thing rather than two files is what stops a question being asked with one
    half of one prompt and one half of another, which would tell a model it was deciding
    something other than what it was shown. Keeping them beside the step is what stops a
    step being moved without the words it asks in.
    """
    named = set()
    for question in (
        LabelQuestion,
        OwnershipDecision,
        MembershipDecision,
        BodyGroupQuestion,
        MixinProposal,
    ):
        prompt = question.prompt
        assert prompt not in named, f"{question.__name__} shares a prompt"
        named.add(prompt)
        assert (
            question.prompts_directory.parent
            == pathlib.Path(sys.modules[question.__module__].__file__).resolve().parent
        ), f"{question.__name__} reads prompts from another step"
        for half in PromptHalf:
            assert question.templates.render_document(f"{prompt.value}/{half.value}"), (
                f"{question.__name__} has no {half.value}"
            )


# %% what the model is told it is deciding, with the contested faces drawn


@pytest.fixture
def expected_instruction(dataset):
    """
    :return: A reader for what a question's instruction said in the kitchenlab run that
        drew the contested faces.
    """

    def read(name: str) -> str:
        return (dataset / "expected" / f"system_{name}.md").read_text()

    return read


def test_the_ownership_instruction_says_what_it_always_said(
    finished_run, questions, relations, expected_instruction
):
    """
    Pinned against what that run was told, so that a run asked from the text alone
    differs from it in the pictures and nothing else.
    """
    question = OwnershipDecision(
        asked=questions.ownership[0],
        labels=relations.labels,
        renders_directory=finished_run.path(RunFile.QUESTION_RENDERS),
    )
    assert question.system_prompt == expected_instruction("ownership")


def test_the_membership_instruction_says_what_it_always_said(
    finished_run, questions, relations, expected_instruction
):
    """
    The same for which whole a part belongs to.
    """
    question = MembershipDecision(
        asked=questions.membership[0],
        labels=relations.labels,
        renders_directory=finished_run.path(RunFile.QUESTION_RENDERS),
    )
    assert question.system_prompt == expected_instruction("membership")
