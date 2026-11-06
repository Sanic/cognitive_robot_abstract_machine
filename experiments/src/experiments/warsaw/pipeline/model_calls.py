"""
Records of every request made to a model, compact enough to keep one per attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from krrood.adapters.json_field import JSONField
from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    PartKind,
    TextPart,
)
from typing_extensions import Any, Dict, List

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.exceptions import UnsupportedMessagePartError

# %% the parts sent to a model


@dataclass(frozen=True)
class ModelCallPart(JsonRecord):
    """
    One part of a message sent to a model, with the text it held or the image it named.
    """

    kind: PartKind
    """
    Whether the part is text or an image.
    """

    text: str | None = None
    """
    The text exactly as it was sent, for a text part.
    """

    image_sha256: str | None = None
    """
    The digest of the image's bytes, for an image part.
    """

    image_bytes: int | None = None
    """
    How large the image was, for an image part.
    """

    @classmethod
    def of(cls, part: MessagePart) -> ModelCallPart:
        """
        Record a part of a message, naming an image by its digest rather than copying it
        into every trace.

        :param part: The text or image that was sent.
        :return: The record of it.
        :raises UnsupportedMessagePartError: If the part is neither text nor an image.
        """
        if isinstance(part, TextPart):
            return cls(kind=PartKind.TEXT, text=part.text)
        if isinstance(part, ImagePart):
            return cls(
                kind=PartKind.IMAGE,
                image_sha256=sha256(part.image).hexdigest(),
                image_bytes=len(part.image),
            )
        raise UnsupportedMessagePartError(part_type=type(part))


# %% one complete attempt


@dataclass(frozen=True)
class ModelCallTrace:
    """
    What one attempt at a question sent, what came back, and what was wrong with it.
    """

    question: str
    """
    The key of the question the attempt answers.
    """

    attempt: int
    """
    Which attempt at the question this was, counting from one.
    """

    requested_model: str
    """
    The model the request was addressed to.
    """

    system_prompt: str
    """
    The system prompt exactly as it was sent.
    """

    message_parts: List[ModelCallPart]
    """
    The parts of the message, in the order they were sent.
    """

    response: Dict[str, Any]
    """
    The response exactly as the provider returned it.
    """

    problems: List[str]
    """
    What made the answer unusable, empty when nothing did.
    """

    started_at: str
    """
    When the attempt began, in UTC.
    """

    elapsed_seconds: float
    """
    How long the response took to arrive.
    """

    @classmethod
    def from_json(cls, written: Dict[str, Any]) -> ModelCallTrace:
        """
        :param written: A trace as :meth:`to_json` wrote it.
        :return: The trace.
        """
        return cls(
            question=written["question"],
            attempt=written["attempt"],
            requested_model=written["requested_model"],
            system_prompt=written["system_prompt"],
            message_parts=[
                ModelCallPart.from_json(part) for part in written["message_parts"]
            ],
            response=written["response"],
            problems=written["problems"],
            started_at=written["started_at"],
            elapsed_seconds=written["elapsed_seconds"],
        )

    def to_json(self) -> Dict[str, Any]:
        """
        Write the trace with the provider's response left exactly as it arrived.

        The dataclass serializer would write the response as the keys and values of a
        mapping, and a trace is read by people comparing it to what the provider sent.

        :return: The trace as JSON-ready data.
        """
        return {
            JSONField.TYPE: get_full_class_name(type(self)),
            "question": self.question,
            "attempt": self.attempt,
            "requested_model": self.requested_model,
            "system_prompt": self.system_prompt,
            "message_parts": [part.to_json() for part in self.message_parts],
            "response": self.response,
            "problems": self.problems,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
        }
