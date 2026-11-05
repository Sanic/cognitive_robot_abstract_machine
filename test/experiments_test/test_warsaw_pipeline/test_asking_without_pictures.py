"""
Asking a run's questions from the text alone, without rendering anything.

Rendering is most of what a run spends its time on -- 44 minutes of room 13's 55 -- and
three pictures per label is most of what it spends its money on. A run told to show no
pictures renders none, and the two questions that would have carried them say so rather
than describing pictures that are not there.

The label is still given. What is being measured is what the pictures add on top of it,
so everything else about the run stays as it was.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest
from semantic_digital_twin.adapters.vision_language_model.client import (
    ModelResponse,
    VisionLanguageModel,
)
from semantic_digital_twin.adapters.vision_language_model.message import MessagePart
from typing_extensions import Dict, Sequence

from experiments.warsaw.pipeline.steps import step as pipeline_step
from experiments.warsaw.pipeline.records import (
    Adjudications,
    LabelRequest,
    OpenQuestions,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.adjudicate.step import (
    AdjudicateOverlaps,
    MembershipDecision,
    OwnershipDecision,
)
from experiments.warsaw.pipeline.steps.classify.step import BodyGroupQuestion
from experiments.warsaw.pipeline.steps.vocabulary.step import LabelQuestion
from experiments.warsaw.world_loader.loader import RenderedSegmentGroup

# %% a question of each kind, with pictures and without

TAXONOMY = {"classes": [], "part_whole_mixins": [], "root_name": "SemanticAnnotation"}
"""
The least a question will take, since what is under test is the pictures and not the
ontology.
"""


@pytest.fixture
def three_renders(tmp_path) -> list:
    """
    :return: One render of each kind, named as the pipeline names them.
    """
    from PIL import Image

    names = []
    for kind in ("context", "plain", "closeup"):
        name = f"chair__{kind}_front_left.png"
        Image.new("RGB", (4, 4), (200, 0, 0)).save(tmp_path / name)
        names.append(name)
    return names


@pytest.fixture
def label_question(tmp_path):
    """
    :return: A maker of label questions carrying the renders it is given.
    """

    def made(images: list) -> LabelQuestion:
        return LabelQuestion(
            taxonomy=TAXONOMY,
            label=LabelRequest(
                label="chair", instances=1, exemplar="chair_1", images=list(images)
            ),
            every_label=["chair"],
            renders_directory=tmp_path,
        )

    return made


@pytest.fixture
def body_group_question():
    """
    :return: A maker of body-group questions carrying the renders it is given.
    """

    def made(images: dict) -> BodyGroupQuestion:
        return BodyGroupQuestion(
            taxonomy=TAXONOMY,
            vocabulary=Vocabulary(scene="", model=""),
            rendered=RenderedSegmentGroup(
                index=0, segments=[], colors={}, images=dict(images)
            ),
        )

    return made


# %% the setting


def test_a_run_shows_its_pictures_unless_told_otherwise() -> None:
    """
    Every run already made was asked with pictures, and its numbers stay comparable.
    """
    assert PipelineSettings().show_the_pictures is True


# %% what a question puts to a model when it has no pictures


def test_a_label_question_without_pictures_carries_no_image(label_question) -> None:
    """
    The message is built from what the question has, so a label that was never rendered
    puts text and nothing else.
    """
    assert [type(one).__name__ for one in label_question([]).message()] == ["TextPart"]


def test_a_label_question_with_pictures_still_carries_them(
    label_question, three_renders
) -> None:
    """
    The other half of the same switch: nothing changes for a run that renders.
    """
    kinds = [type(one).__name__ for one in label_question(three_renders).message()]
    assert kinds.count("ImagePart") == 3


def test_a_label_question_with_pictures_says_which_object_they_show(
    label_question, three_renders
) -> None:
    """
    The message names the object pictured, so the model knows which of the label's
    objects it is looking at.
    """
    [text, *_] = label_question(three_renders).message()

    assert '"chair_1"' in text.text


def test_a_label_question_without_pictures_names_no_pictured_object(
    label_question,
) -> None:
    [text] = label_question([]).message()

    assert "pictures" not in text.text.lower()


def test_a_label_question_without_pictures_is_not_told_to_read_any(
    label_question,
) -> None:
    """
    The standing instruction promises pictures. Sending it with none asks a model to
    describe what it cannot see, which it will answer rather than refuse.
    """
    assert "picture" not in label_question([]).system_prompt.lower()


def test_a_label_question_with_pictures_is_told_to_read_them(
    label_question, three_renders
) -> None:
    """
    And the instruction a run has always been asked with is unchanged.
    """
    assert "picture" in label_question(three_renders).system_prompt.lower()


def test_a_body_group_question_without_pictures_carries_no_image(
    body_group_question,
) -> None:
    """
    The same for the question that names each body.
    """
    parts = body_group_question({}).message()
    assert [type(one).__name__ for one in parts] == ["TextPart"]


def test_a_body_group_question_without_pictures_is_not_told_to_read_any(
    body_group_question,
) -> None:
    """
    Its instruction opens by describing the pictures, so it needs the same care.
    """
    assert "picture" not in body_group_question({}).system_prompt.lower()


def test_a_body_group_question_without_pictures_names_no_colour(
    body_group_question,
) -> None:
    """
    A body is named to a model by the colour it was painted in the picture. With no
    picture there is no paint, and naming one would be describing something that was
    never drawn.
    """
    [text] = body_group_question({}).message()
    assert "painted" not in text.text


# %% the groups a run walks, drawn and undrawn


def test_an_undrawn_group_carries_no_picture_and_no_colour() -> None:
    """
    Where the time is saved: a run asking from the text alone paints nothing and places
    no camera, so a group comes back with neither a render nor the colour it would have
    been painted.
    """
    from experiments.warsaw.world_loader.loader import WarsawWorldLoader

    walked = WarsawWorldLoader.label_segment_groups
    assert "pictured" in walked.__code__.co_varnames


def test_the_render_step_is_told_what_the_run_was_told(tmp_path) -> None:
    """
    The setting has to reach the rendering, not only the prompts: a run that rendered and
    then hid the pictures would pay the whole cost for none of the saving.
    """
    from experiments.warsaw.pipeline.pipeline import WarsawPipeline
    from experiments.warsaw.pipeline.run import Run
    from experiments.warsaw.pipeline.steps.evidence import MeasureScene

    def measuring(showing: bool) -> list:
        settings = PipelineSettings(scene_directory=tmp_path, show_the_pictures=showing)
        planned = WarsawPipeline(settings=settings).run_steps(Run(directory=tmp_path))
        return [
            one.renders_exemplars for one in planned if isinstance(one, MeasureScene)
        ]

    # A scene is measured twice and only the first measuring renders the exemplars, so
    # what the setting decides is that first one.
    assert measuring(showing=True) == [True, False]
    assert measuring(showing=False) == [False, False]


# %% the third render site, which no HM3D room can reach


def test_a_run_showing_no_pictures_renders_none_for_its_questions(tmp_path) -> None:
    """
    The site this missed. A question about contested faces is rendered by a step of its
    own, and no HM3D room reaches it: those files give every face to exactly one object,
    so a room has nought ownership and nought membership questions and the renders never
    ran. A scan contests faces constantly, and the window opened.
    """
    from experiments.warsaw.pipeline.pipeline import WarsawPipeline
    from experiments.warsaw.pipeline.run import Run
    from experiments.warsaw.pipeline.steps.evidence import MeasureScene

    def rendering(showing: bool) -> list:
        settings = PipelineSettings(
            scene_directory=tmp_path,
            show_the_pictures=showing,
            show_the_contested_faces=showing,
        )
        planned = WarsawPipeline(settings=settings).run_steps(Run(directory=tmp_path))
        return [
            one.renders_questions for one in planned if isinstance(one, MeasureScene)
        ]

    assert rendering(showing=False) == [0, 0]
    assert max(rendering(showing=True)) > 0


def test_the_contested_faces_are_asked_about_on_their_own_terms(tmp_path) -> None:
    """
    A different question from what an object is, and worth a different amount: a label
    says what a thing is and says nothing about whose a shared surface is. So a run can
    draw the contested faces while drawing nothing else.
    """
    from experiments.warsaw.pipeline.pipeline import WarsawPipeline
    from experiments.warsaw.pipeline.run import Run
    from experiments.warsaw.pipeline.steps.evidence import MeasureScene

    settings = PipelineSettings(
        scene_directory=tmp_path,
        show_the_pictures=False,
        show_the_contested_faces=True,
    )
    planned = WarsawPipeline(settings=settings).run_steps(Run(directory=tmp_path))
    measuring = [one for one in planned if isinstance(one, MeasureScene)]
    assert max(one.renders_questions for one in measuring) > 0
    assert [one.renders_exemplars for one in measuring] == [False, False]


# %% the contested faces, asked from the text alone


@pytest.fixture
def undrawn_questions(questions):
    """
    :return: The fixture run's open questions as a run that drew no contested faces
        would have written them: nothing rendered and nothing painted.
    """
    return OpenQuestions(
        scene=questions.scene,
        asked=[replace(one, images=[], legend={}) for one in questions.asked],
    )


@pytest.fixture
def undrawn_ownership(undrawn_questions, relations, tmp_path) -> OwnershipDecision:
    """
    :return: The fixture run's ownership question, with nothing drawn for it.
    """
    return OwnershipDecision(
        asked=undrawn_questions.ownership[0],
        labels=relations.labels,
        renders_directory=tmp_path,
    )


@pytest.fixture
def undrawn_membership(undrawn_questions, relations, tmp_path) -> MembershipDecision:
    """
    :return: The fixture run's membership question, with nothing drawn for it.
    """
    return MembershipDecision(
        asked=undrawn_questions.membership[0],
        labels=relations.labels,
        renders_directory=tmp_path,
    )


@dataclass
class AnsweringBySystemPrompt:
    """
    A model answering each kind of question with an answer written in advance, told
    apart by what it is told it is doing.
    """

    answers: Dict[str, dict]
    """
    Per system prompt, what the model answers.
    """

    def ask(self, content: Sequence[MessagePart], system: str) -> ModelResponse:
        """
        :param content: The question.
        :param system: What it is told it is doing.
        :return: The answer written for that kind of question.
        """
        spoken = json.dumps(self.answers[system])
        return ModelResponse.from_json({"choices": [{"message": {"content": spoken}}]})


def test_contested_faces_left_undrawn_are_still_adjudicated(
    tmp_path,
    monkeypatch,
    undrawn_questions,
    undrawn_ownership,
    undrawn_membership,
    relations,
) -> None:
    """
    A question about contested faces has something to go on without its picture: the
    ontology, the measured shares and the candidates are all text. So a run that draws
    none asks them from that text, rather than leaving the faces to nobody -- which would
    compare the pictures against no adjudication at all.
    """
    run = Run(directory=tmp_path)
    run.write_record(RunFile.QUESTIONS, undrawn_questions)
    run.write_record(RunFile.RELATIONS, relations)

    owner = undrawn_ownership.asked.pattern[0]
    whole = undrawn_membership.asked.candidate_names[0]
    scripted = AnsweringBySystemPrompt(
        answers={
            undrawn_ownership.system_prompt: {"owner": owner},
            undrawn_membership.system_prompt: {"whole": whole},
        }
    )
    monkeypatch.setattr(
        pipeline_step,
        VisionLanguageModel.__name__,
        lambda **asked: scripted,
    )

    settings = PipelineSettings(
        scene_directory=tmp_path, show_the_contested_faces=False
    )
    AdjudicateOverlaps(settings=settings, run=run).carry_out()

    adjudications = run.read_record(RunFile.ADJUDICATIONS, Adjudications)
    assert [one.owner for one in adjudications.ownership] == [owner]
    assert [one.whole for one in adjudications.membership] == [whole]


def test_an_undrawn_ownership_question_carries_no_image(undrawn_ownership) -> None:
    """
    The message is built from what the question has.
    """
    kinds = {type(one).__name__ for one in undrawn_ownership.message()}
    assert kinds == {"TextPart"}


def test_an_undrawn_ownership_question_mentions_no_picture_or_colour(
    undrawn_ownership,
) -> None:
    """
    Its instruction describes the colours each object is painted in and its message
    says what the pictures show. Neither was drawn, and naming what is absent invites a
    model to describe it.
    """
    [text] = undrawn_ownership.message()
    said = (undrawn_ownership.system_prompt + text.text).lower()
    assert "picture" not in said
    assert "color" not in said


def test_an_undrawn_membership_question_carries_no_image(undrawn_membership) -> None:
    """
    The same for which whole a part belongs to.
    """
    kinds = {type(one).__name__ for one in undrawn_membership.message()}
    assert kinds == {"TextPart"}


def test_an_undrawn_membership_question_mentions_no_picture_or_colour(
    undrawn_membership,
) -> None:
    """
    Its instruction still said each candidate has its own colour after the pictures were
    made conditional.
    """
    [text] = undrawn_membership.message()
    said = (undrawn_membership.system_prompt + text.text).lower()
    assert "picture" not in said
    assert "color" not in said
