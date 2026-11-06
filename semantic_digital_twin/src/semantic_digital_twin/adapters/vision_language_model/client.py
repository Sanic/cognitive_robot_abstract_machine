"""
Ask a vision-language model on OpenRouter and read its answer back as data.

A caller asks several different questions -- which class a label means, which of two
overlapping objects holds the other -- and they differ only in what is said and what
pictures come with it. This holds what they have in common: building the message,
surviving a rate limit, and getting JSON out of a reply that may be wrapped in prose.
"""

from __future__ import annotations

import json
import logging
import os
import time
from time import perf_counter
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from http import HTTPStatus

import requests
from typing_extensions import Any, Dict, Optional, Sequence, Tuple

from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    AnswerCutOffError,
    ApiKeyMissingError,
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import MessagePart

# %% what a service says went wrong


class ServiceFailure(IntEnum):
    """
    An answer that says the service failed rather than that the question was wrong.

    Asking again is worth it for exactly these: the service is busy or broken, and the
    same question put again a moment later may be answered.
    """

    RATE_LIMITED = HTTPStatus.TOO_MANY_REQUESTS
    """
    Too many questions were asked too quickly.
    """

    SERVER_ERROR = HTTPStatus.INTERNAL_SERVER_ERROR
    """
    The service failed while answering.
    """

    BAD_GATEWAY = HTTPStatus.BAD_GATEWAY
    """
    Something in front of the service could not reach it.
    """

    UNAVAILABLE = HTTPStatus.SERVICE_UNAVAILABLE
    """
    The service is not taking questions at the moment.
    """

    GATEWAY_TIMEOUT = HTTPStatus.GATEWAY_TIMEOUT
    """
    Something in front of the service gave up waiting for it.
    """

    @classmethod
    def to_tuple(cls) -> Tuple[ServiceFailure, ...]:
        """
        :return: Every one of them, in an order and a shape a field can default to.
        """
        return tuple(cls)


class FinishReason(StrEnum):
    """
    Why a model stopped answering, as the service reports it.
    """

    LENGTH = "length"
    """
    The answer reached the longest one the question allowed.
    """


class Role(StrEnum):
    """
    Who a message in a conversation is from.
    """

    SYSTEM = "system"
    """
    What the model is told it is doing.
    """

    USER = "user"
    """
    The question itself.
    """


# %% reading what the model answered


@dataclass
class ModelResponse:
    """
    What a model answered, as it came back.
    """

    payload: Dict[str, Any]
    """
    The whole response, kept so that what was answered and what it cost stay together in
    whatever is written to disk.
    """

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> ModelResponse:
        """
        :param payload: A response as it was received or as it was kept.
        :return: It, ready to be read.
        """
        return cls(payload=payload)

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The response as it came back, so a kept copy answers as the original did.
        """
        return self.payload

    @property
    def text(self) -> str:
        """
        :return: What the model said, empty where it said nothing -- a reply can carry a
            null content, and a caller reading it as text should get text.
        """
        return self.payload["choices"][0]["message"].get("content") or ""

    @property
    def cut_off(self) -> bool:
        """
        :return: Whether the answer stopped at the longest one the question allowed.
        """
        return self.payload["choices"][0].get("finish_reason") == FinishReason.LENGTH

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        """
        :return: What the question cost, when the response says.
        """
        return self.payload.get("usage")

    def parse_json(self) -> Any:
        """
        Read the JSON out of the reply.

        Models asked for JSON answer with it in a fenced block, or with a sentence in
        front of it, often enough that a reply is worth searching rather than only parsed.

        :return: The JSON object or array in it.
        :raises AnswerCutOffError: If the answer stopped at the length limit.
        :raises ModelRefusedError: If there is none, an empty reply included.
        """
        answer = self.text
        if self.cut_off:
            raise AnswerCutOffError(answer=answer)
        if not answer.strip():
            raise ModelRefusedError(answer=answer)

        # The reply is written by a model rather than by us, so a failure to parse is the
        # expected shape of a wrapped answer rather than an illegal state.
        for candidate in self._candidates(answer):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ModelRefusedError(answer=answer)

    @staticmethod
    def _candidates(answer: str) -> Sequence[str]:
        """
        :param answer: What the model said.
        :return: The whole reply, then the widest object and the widest array in it.
        """
        found = [answer]
        for opening, closing in (("{", "}"), ("[", "]")):
            start, end = answer.find(opening), answer.rfind(closing)
            if start != -1 and end > start:
                found.append(answer[start : end + 1])
        return found


# %% asking the model


@dataclass
class VisionLanguageModel:
    """
    A model that reads pictures beside words, asked over OpenRouter.
    """

    model: str
    """
    The identifier the service knows the model by.
    """

    url: str = "https://openrouter.ai/api/v1/chat/completions"
    """
    Where a question is sent.
    """

    api_key_variable: str = "OPENROUTER_API_KEY"
    """
    The environment variable a credential for that service is read from.
    """

    asks_again_after: Tuple[ServiceFailure, ...] = ServiceFailure.to_tuple()
    """
    The answers worth asking again after.
    """

    timeout: int = 180
    """
    How long one attempt may take, in seconds.
    """

    maximum_attempts: int = 5
    """
    How often one question is asked before its failure is raised.
    """

    seconds_per_question: int = 900
    """
    How long one question may take in all, across every attempt made for it.

    Attempts alone do not bound the time a question takes, because :attr:`timeout` is what
    ``requests`` means by one: it bounds each socket operation rather than the request, so
    a service that sends something now and then resets it and the request waits for as
    long as the service holds the connection. Two runs were killed after fifty and thirty
    minutes with no attempt recorded, both at the same question.
    """

    temperature: float = 0.0
    """
    How much the model may explore when sampling.

    Zero by default: a question with one right answer, asked once, has nothing for
    sampling to explore. It does not make a run repeatable -- expert routing and batching
    still move under us -- but it removes the variance that is ours to remove.
    """

    presence_penalty: float = 0.0
    """
    How much less likely a token becomes once the answer has used it, which breaks a
    model out of repeating itself without making its answers random.
    """

    maximum_answer_tokens: int = 4096
    """
    The longest answer a question allows, in tokens.

    Without one, a model that starts repeating itself goes on until the service's own
    limit.
    """

    referer: str = "ai.uni-bremen.de"
    """
    Who the service is told is asking.
    """

    title: str = "Uni Bremen"
    """
    What the service is told the asking is for.
    """

    headers_extra: Dict[str, str] = field(default_factory=dict)
    """
    Anything else to send with the request.
    """

    def ask(self, content: Sequence[MessagePart], system: str) -> ModelResponse:
        """
        Put one question to the model.

        :param content: The message, in the order it is to be read.
        :param system: What the model is told it is doing.
        :return: What it answered.
        :raises ApiKeyMissingError: If no credential is configured.
        :raises requests.RequestException: If the question could not be asked, after
            every attempt worth making.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "presence_penalty": self.presence_penalty,
                "max_tokens": self.maximum_answer_tokens,
                "messages": [
                    {"role": Role.SYSTEM.value, "content": system},
                    {
                        "role": Role.USER.value,
                        "content": [part.to_json() for part in content],
                    },
                ],
            }
        )
        return ModelResponse.from_json(self._post(payload))

    def _post(self, payload: str) -> Dict[str, Any]:
        """
        Send one question, asking again while the failure is one that may pass.

        A rate limit and a gateway error are the service being busy rather than the
        question being wrong, which is why they are caught rather than raised.

        :param payload: The request body.
        :return: The response, parsed.
        :raises requests.RequestException: The last failure, once no attempt is left.
        """
        started = perf_counter()
        for attempt in range(self.maximum_attempts):
            try:
                response = requests.post(
                    url=self.url,
                    headers=self._headers(),
                    data=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_failure:
                failure = http_failure
                worth_retrying = response.status_code in self.asks_again_after
                reason = f"returned {response.status_code}"
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as network_failure:
                failure = network_failure
                worth_retrying = True
                reason = f"failed with {type(network_failure).__name__}"

            spent = perf_counter() - started
            last_attempt = attempt == self.maximum_attempts - 1
            out_of_time = spent >= self.seconds_per_question
            if not worth_retrying or last_attempt or out_of_time:
                if out_of_time:
                    logging.getLogger(__name__).warning(
                        "the request %s, and the question has now taken %.0fs of the "
                        "%ss it is allowed, so it is not asked again",
                        reason,
                        spent,
                        self.seconds_per_question,
                    )
                raise failure
            waited = 2**attempt
            logging.getLogger(__name__).warning(
                "the request %s, asking again in %ss", reason, waited
            )
            time.sleep(waited)

    def _headers(self) -> Dict[str, str]:
        """
        :return: What the request is sent with.
        :raises ApiKeyMissingError: If no credential is configured.
        """
        key = os.environ.get(self.api_key_variable)
        if not key:
            raise ApiKeyMissingError(variable=self.api_key_variable)
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
            **self.headers_extra,
        }
