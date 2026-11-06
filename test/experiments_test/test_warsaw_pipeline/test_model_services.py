"""
Which service a model is asked at, and what asking it there takes.

The model is the one switch: choosing the group's own model sends every question to the
server on denver, with that server's credential and waiting times, and leaves room for
the reasoning it does before it answers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import Model, ModelService, PipelineSettings
from experiments.warsaw.pipeline.steps.classify.step import ClassifyBodies

# %% the service each model is asked at


def questioner_for(model, tmp_path):
    """
    :return: The client a step would put its questions to, for *model*.
    """
    settings = PipelineSettings(scene_directory=tmp_path, model=model)
    step = ClassifyBodies(settings=settings, run=Run(directory=tmp_path / "run"))
    return settings, step.questioner(RunFile.CLASSIFICATION_ANSWERS).model


def test_a_model_on_openrouter_is_asked_there(tmp_path):
    _, client = questioner_for(Model.QWEN3_VL_30B, tmp_path)

    assert (client.url, client.api_key_variable) == (
        ModelService.OPENROUTER.value,
        ModelService.OPENROUTER.api_key_variable,
    )


def test_the_group_s_own_model_is_asked_on_denver_with_its_own_credential(tmp_path):
    _, client = questioner_for(Model.QWEN3_8_27B_DENVER, tmp_path)

    assert (client.url, client.api_key_variable) == (
        ModelService.DENVER.value,
        ModelService.DENVER.api_key_variable,
    )


def test_the_server_on_denver_is_waited_for_as_long_as_it_takes_to_answer(tmp_path):
    """
    It sends nothing until its whole answer is written, so the usual wait would give up
    on every answer longer than a few hundred tokens.
    """
    _, client = questioner_for(Model.QWEN3_8_27B_DENVER, tmp_path)

    assert (client.timeout, client.seconds_per_question) == (
        ModelService.DENVER.seconds_per_attempt,
        ModelService.DENVER.seconds_per_question,
    )


def test_a_model_that_reasons_first_is_given_room_for_its_reasoning(tmp_path):
    settings, client = questioner_for(Model.QWEN3_8_27B_DENVER, tmp_path)

    assert client.maximum_answer_tokens == (
        settings.maximum_answer_tokens + Model.QWEN3_8_27B_DENVER.reasoning_tokens
    )


def test_a_model_that_answers_at_once_is_given_only_the_answer_limit(tmp_path):
    settings, client = questioner_for(Model.QWEN3_VL_30B, tmp_path)

    assert client.maximum_answer_tokens == settings.maximum_answer_tokens


# %% Qwen3.8 on OpenRouter


def test_qwen3_8_on_openrouter_uses_its_model_identifier_and_service(
    tmp_path: Path,
) -> None:
    """
    Keep the hosted model distinct from the Denver deployment.
    """
    _, client = questioner_for(Model.QWEN3_8_27B_OPENROUTER, tmp_path)

    assert client.model == Model.QWEN3_8_27B_OPENROUTER.value
    assert client.url == ModelService.OPENROUTER.value
    assert client.api_key_variable == ModelService.OPENROUTER.api_key_variable
    assert client.timeout == ModelService.OPENROUTER.seconds_per_attempt
    assert client.seconds_per_question == ModelService.OPENROUTER.seconds_per_question


def test_qwen3_8_on_openrouter_has_room_for_reasoning(tmp_path: Path) -> None:
    """
    Reserve the same completion allowance for both deployments of Qwen3.8.
    """
    settings, client = questioner_for(Model.QWEN3_8_27B_OPENROUTER, tmp_path)

    assert Model.QWEN3_8_27B_OPENROUTER.reasoning_tokens == (
        Model.QWEN3_8_27B_DENVER.reasoning_tokens
    )
    assert client.maximum_answer_tokens == (
        settings.maximum_answer_tokens + Model.QWEN3_8_27B_DENVER.reasoning_tokens
    )


# %% what a table calls a model


@pytest.mark.parametrize("model", list(Model))
def test_every_model_has_a_short_name_of_its_own(model):
    others = [one.short_name for one in Model if one is not model]

    assert model.short_name not in others
    assert len(model.short_name) < 16
