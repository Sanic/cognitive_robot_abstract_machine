"""
When a new class may be proposed: the moment a label is asked about, or only afterwards,
once per label no class of the ontology was found for, with every other label's reading
in view.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field

import pytest
from semantic_digital_twin.adapters.vision_language_model.client import (
    ModelResponse,
    VisionLanguageModel,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    MessagePart,
    TextPart,
)
from typing_extensions import Dict, List, Sequence

from experiments.warsaw.pipeline.records import Vocabulary
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps import step as pipeline_step
from experiments.warsaw.pipeline.steps.vocabulary.step import MapLabelVocabulary

# %% a model answering per label


@dataclass
class AnsweringPerLabel:
    """
    A model answering a label question and a class proposal with answers written in
    advance per label, and keeping every proposal it was asked for.
    """

    readings: Dict[str, dict]
    """
    Per label, what the label question is answered with.
    """

    proposals: Dict[str, dict]
    """
    Per label, what the class proposal is answered with.
    """

    proposals_asked: List[str] = field(default_factory=list)
    """
    The message of every class proposal asked for, in order.
    """

    def ask(self, content: Sequence[MessagePart], system: str) -> ModelResponse:
        """
        :param content: The question.
        :param system: What the model is told it is doing.
        :return: The answer written for that label and that kind of question.
        """
        text = "\n".join(part.text for part in content if isinstance(part, TextPart))
        label = re.search(r'The label is "([^"]+)"', text).group(1)
        if "An earlier question found no" in system:
            self.proposals_asked.append(text)
            answer = self.proposals[label]
        else:
            answer = self.readings[label]
        return ModelResponse.from_json(
            {"choices": [{"message": {"content": json.dumps(answer)}}]}
        )


@pytest.fixture
def run_to_map(finished_run, tmp_path) -> Run:
    """
    :return: A run holding what the label mapping reads: the labels to ask about, the
        taxonomy and the measured scene.
    """
    run = Run(directory=tmp_path / "run")
    run.directory.mkdir()
    for kept in (
        RunFile.VOCABULARY_REQUEST,
        RunFile.TAXONOMY,
        RunFile.RELATIONS,
    ):
        shutil.copy(finished_run.path(kept), run.path(kept))
    return run


def mapped(run, monkeypatch, model, defer: bool) -> Vocabulary:
    """
    :return: What the label mapping wrote for the cabinet and the drawer, asked of
        *model*.
    """
    monkeypatch.setattr(
        pipeline_step, VisionLanguageModel.__name__, lambda **asked: model
    )
    settings = PipelineSettings(scene_directory=run.directory, defer_new_classes=defer)
    MapLabelVocabulary(
        settings=settings, run=run, only_labels=["cabinet", "drawer"]
    ).carry_out()
    return run.read_record(RunFile.VOCABULARY, Vocabulary)


# %% deferring


def test_a_label_no_class_was_found_for_is_given_one_proposed_afterwards(
    run_to_map, monkeypatch
):
    model = AnsweringPerLabel(
        readings={"cabinet": {"class": "Cabinet"}, "drawer": {"class": None}},
        proposals={
            "drawer": {
                "class": "SlidingTray",
                "is_new_class": True,
                "superclass": "Furniture",
                "mixins": [],
            }
        },
    )

    vocabulary = mapped(run_to_map, monkeypatch, model, defer=True)

    assert (
        vocabulary.answer_for("cabinet").class_name,
        vocabulary.answer_for("drawer").class_name,
        vocabulary.answer_for("drawer").is_new_class,
    ) == ("Cabinet", "SlidingTray", True)


def test_a_class_is_proposed_with_every_other_label_s_reading_in_view(
    run_to_map, monkeypatch
):
    model = AnsweringPerLabel(
        readings={"cabinet": {"class": "Cabinet"}, "drawer": {"class": None}},
        proposals={"drawer": {"class": "Drawer"}},
    )

    mapped(run_to_map, monkeypatch, model, defer=True)

    [asked] = model.proposals_asked
    assert "cabinet: Cabinet" in asked


def test_a_label_a_class_was_found_for_is_not_asked_again(run_to_map, monkeypatch):
    model = AnsweringPerLabel(
        readings={"cabinet": {"class": "Cabinet"}, "drawer": {"class": "Drawer"}},
        proposals={},
    )

    mapped(run_to_map, monkeypatch, model, defer=True)

    assert model.proposals_asked == []


def test_without_deferring_a_label_is_given_its_new_class_at_once(
    run_to_map, monkeypatch
):
    model = AnsweringPerLabel(
        readings={
            "cabinet": {"class": "Cabinet"},
            "drawer": {
                "class": "SlidingTray",
                "is_new_class": True,
                "superclass": "Furniture",
                "mixins": [],
            },
        },
        proposals={},
    )

    vocabulary = mapped(run_to_map, monkeypatch, model, defer=False)

    assert model.proposals_asked == []
    assert vocabulary.answer_for("drawer").class_name == "SlidingTray"
