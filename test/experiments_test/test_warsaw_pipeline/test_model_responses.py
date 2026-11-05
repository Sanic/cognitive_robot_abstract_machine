"""
Reading an answer out of a reply that was written by a model rather than by a schema.

A model asked for JSON answers with it fenced, with a sentence in front of it, with the
array alone where an object was asked for, and sometimes with nothing at all. What is
checked here is which of those still carry an answer and which are refusals, because a
refusal read as an answer is a body that quietly gets no class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus

import pytest
import requests
from requests import Response
from typing_extensions import Any, List

from semantic_digital_twin.adapters.vision_language_model.client import (
    ModelResponse,
    ServiceFailure,
    VisionLanguageModel,
)
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    AnswerCutOffError,
    ApiKeyMissingError,
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    PartKind,
    TextPart,
)


def response_from(replies, name: str) -> ModelResponse:
    """
    :param replies: The directory holding replies as a model returned them.
    :param name: Which reply to read.
    :return: It, ready to be read.
    """
    return ModelResponse.from_json(json.loads((replies / name).read_text()))


# %% answers that are wrapped rather than absent


def test_a_fenced_answer_is_read(replies):
    """
    A model asked for JSON answers with it inside a code fence.
    """
    assert response_from(replies, "fenced.json").parse_json() == {
        "class": "Drawer",
        "is_new_class": False,
    }


def test_an_answer_with_prose_around_it_is_read(replies):
    """
    A sentence in front of the answer, and one after it, leave the answer readable.
    """
    assert response_from(replies, "prose_wrapped.json").parse_json() == {
        "class": "Drawer",
        "is_new_class": False,
    }


def test_an_array_answer_is_read(replies):
    """
    Asked for an object holding an array, a model will answer with the array alone.
    """
    assert response_from(replies, "array_only.json").parse_json() == [
        {"name": "drawer_1", "class": "Drawer"}
    ]


def test_a_real_reply_is_read(replies):
    """
    A reply as the service actually returned one carries the answer it was asked for.
    """
    answered = response_from(replies, "vocabulary_kitchen_island.json").parse_json()
    assert answered["class"] == "KitchenIsland"
    assert answered["is_new_class"] is True


def test_a_real_reply_answering_about_a_whole_group_is_read(replies):
    """
    A group is asked about in one question, and the reply carries one answer per body.
    """
    answered = response_from(replies, "classification_group.json").parse_json()

    assert [one["name"] for one in answered] == [
        "floor_1",
        "ceiling_1",
        "ceiling_2",
        "ceiling_3",
        "ceiling_4",
        "ceiling_5",
        "ceiling_6",
        "ceiling_7",
    ]
    assert answered[0]["class"] == "Floor"
    assert answered[0]["is_new_class"] is False


def test_a_real_reply_deciding_whose_a_face_is_is_read(replies):
    """
    An ownership question is answered with the one owner it settles on.
    """
    answered = response_from(
        replies, "ownership_cabinet_drawer_kitchen_island.json"
    ).parse_json()

    assert answered["owner"] == "drawer"


# %% replies that hold no answer


@pytest.mark.parametrize(
    "reply", ["empty.json", "null_content.json", "prose_only.json"]
)
def test_a_reply_without_json_is_a_refusal(replies, reply):
    """
    A model that answers with nothing, or with prose, has refused as surely as one that
    says so -- and a refusal has to raise rather than come back as a blank answer.
    """
    with pytest.raises(ModelRefusedError):
        response_from(replies, reply).parse_json()


def test_a_null_content_reads_as_empty_text(replies):
    """
    A reply can carry a null content, and a caller reading it as text should get text.
    """
    assert response_from(replies, "null_content.json").text == ""


# %% answers that ran on


def test_an_answer_stopped_at_the_length_limit_is_refused_as_cut_off(replies):
    """
    A model repeating itself is stopped at the limit, and what it said by then may even
    open like an answer.
    """
    with pytest.raises(AnswerCutOffError):
        response_from(replies, "cut_off.json").parse_json()


def test_a_correction_of_a_cut_off_answer_does_not_quote_it(replies):
    """
    Quoting what a model repeated back to it is how it is led to repeat it again.
    """
    cut_off = response_from(replies, "cut_off.json")

    with pytest.raises(AnswerCutOffError) as refused:
        cut_off.parse_json()

    assert cut_off.text[: ModelRefusedError.shown] not in str(refused.value)


@dataclass
class RecordedPost:
    """
    Stands in for ``requests.post``, keeping what it was sent and answering once.
    """

    sent: List[Any]
    """
    The bodies it was sent, in order.
    """

    def __call__(self, **kwargs) -> Response:
        """
        :param kwargs: What the client sends.
        :return: A successful response.
        """
        self.sent.append(json.loads(kwargs["data"]))
        response = Response()
        response.status_code = HTTPStatus.OK
        response._content = json.dumps({"answered": True}).encode()
        return response


def test_a_question_allows_no_longer_an_answer_than_its_limit(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "a-key")
    post = RecordedPost(sent=[])
    monkeypatch.setattr(
        "semantic_digital_twin.adapters.vision_language_model.client.requests.post",
        post,
    )
    model = VisionLanguageModel(model="a-model", maximum_answer_tokens=512)

    model.ask([TextPart("Which class is this?")], system="Name the class.")

    assert [body["max_tokens"] for body in post.sent] == [model.maximum_answer_tokens]


def test_a_question_is_sent_with_its_penalty_on_repeating_itself(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "a-key")
    post = RecordedPost(sent=[])
    monkeypatch.setattr(
        "semantic_digital_twin.adapters.vision_language_model.client.requests.post",
        post,
    )
    model = VisionLanguageModel(model="a-model", presence_penalty=1.5)

    model.ask([TextPart("Which class is this?")], system="Name the class.")

    assert [body["presence_penalty"] for body in post.sent] == [model.presence_penalty]


# %% the message a question is sent as


def test_words_are_sent_as_words():
    """
    A text part names itself the way the chat completions schema does.
    """
    assert TextPart("hello").to_json() == {
        "type": PartKind.TEXT.value,
        "text": "hello",
    }


def test_a_picture_is_carried_inline(tmp_path):
    """
    A picture is carried in the message rather than by URL, so nothing has to be hosted
    for a model to see it.
    """
    picture = tmp_path / "render.png"
    picture.write_bytes(b"not really a png")
    sent = ImagePart.from_file(picture).to_json()
    assert sent["type"] == PartKind.IMAGE.value
    assert sent["image_url"]["url"].startswith("data:image/png;base64,")


# %% what is worth asking again after


def test_the_service_being_busy_is_worth_asking_again_after():
    """
    A rate limit is the service being busy, not the question being wrong.
    """
    assert (
        ServiceFailure.RATE_LIMITED in VisionLanguageModel(model="any").asks_again_after
    )


def test_a_question_the_service_rejects_is_not_asked_again():
    """
    Asking a question the service will not answer a second time buys nothing.
    """
    asks_again_after = VisionLanguageModel(model="any").asks_again_after
    assert not any(one == 404 for one in asks_again_after)


def test_asking_without_a_credential_says_which_one_is_missing(monkeypatch):
    """
    A run that would spend a hundred questions should stop on the first one, saying what
    to set rather than what failed.
    """
    model = VisionLanguageModel(model="any")
    monkeypatch.delenv(model.api_key_variable, raising=False)
    with pytest.raises(ApiKeyMissingError) as raised:
        model.ask([TextPart("anything")], system="anything")
    assert raised.value.variable == model.api_key_variable


# %% asking again when the service, not the question, is at fault


@dataclass
class ScriptedPosts:
    """
    Stands in for ``requests.post``, answering each call from a script.

    Each entry is either a status code to answer with or an exception to raise, so one
    script describes a service that fails and then recovers.
    """

    answers: List[Any]
    """
    What to do on each successive call, in order.
    """

    calls: int = 0
    """
    How many times it was asked.
    """

    def __call__(self, **kwargs) -> Response:
        """
        :param kwargs: What the client sends, which this ignores.
        :return: The next scripted response.
        """
        answer = self.answers[self.calls]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        response = Response()
        response.status_code = answer
        response._content = json.dumps({"answered": self.calls}).encode()
        return response


@pytest.fixture
def unwaiting(monkeypatch) -> None:
    """
    Take the waiting out of the backoff, so the retries are not also a delay.
    """
    monkeypatch.setattr(
        "semantic_digital_twin.adapters.vision_language_model.client.time.sleep",
        lambda seconds: None,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "a-key")


def posting(monkeypatch, answers: List[Any]) -> ScriptedPosts:
    """
    :param monkeypatch: The fixture replacing the real call.
    :param answers: What the service answers on each successive call.
    :return: The stand-in, so a test can count how often it was asked.
    """
    posts = ScriptedPosts(answers=answers)
    monkeypatch.setattr(
        "semantic_digital_twin.adapters.vision_language_model.client.requests.post",
        posts,
    )
    return posts


def test_a_busy_service_is_asked_again_and_its_later_answer_is_read(
    monkeypatch, unwaiting
):
    """
    A rate limit is the service being busy, so the same question is put again.
    """
    posts = posting(monkeypatch, [ServiceFailure.RATE_LIMITED, HTTPStatus.OK])

    answered = VisionLanguageModel(model="a-model")._post("{}")

    assert posts.calls == 2
    assert answered == {"answered": 2}


def test_a_question_the_service_refuses_is_not_asked_again(monkeypatch, unwaiting):
    """
    A status that is not one of :class:`ServiceFailure` says the question was wrong, and
    asking it again would only be wrong again.
    """
    posts = posting(monkeypatch, [HTTPStatus.BAD_REQUEST])

    with pytest.raises(requests.exceptions.HTTPError):
        VisionLanguageModel(model="a-model")._post("{}")

    assert posts.calls == 1


def test_a_service_that_stays_busy_is_given_up_on(monkeypatch, unwaiting):
    """
    The retries are bounded: once no attempt is left the last failure is raised.
    """
    attempts = 3
    posts = posting(monkeypatch, [ServiceFailure.UNAVAILABLE] * attempts)

    with pytest.raises(requests.exceptions.HTTPError):
        VisionLanguageModel(model="a-model", maximum_attempts=attempts)._post("{}")

    assert posts.calls == attempts


def test_a_network_failure_is_asked_again(monkeypatch, unwaiting):
    """
    A connection that never arrived says nothing about the question.
    """
    posts = posting(
        monkeypatch,
        [requests.exceptions.ConnectionError("no route"), HTTPStatus.OK],
    )

    answered = VisionLanguageModel(model="a-model")._post("{}")

    assert posts.calls == 2
    assert answered == {"answered": 2}
