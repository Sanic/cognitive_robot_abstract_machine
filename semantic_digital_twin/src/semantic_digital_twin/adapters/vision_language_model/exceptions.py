from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class ApiKeyMissingError(DataclassException, RuntimeError):
    """
    Raised when no credential is configured for the service a question would go to.
    """

    variable: str
    """
    The environment variable the credential is read from.
    """

    def error_message(self) -> str:
        return f"'{self.variable}' is not set, and asking a model needs it."

    def suggest_correction(self) -> str:
        return (
            f"Export '{self.variable}' with a key for the service before asking, or ask "
            "a model that needs no credential."
        )


@dataclass
class ModelRefusedError(DataclassException, ValueError):
    """
    Raised when a reply holds no JSON, so there is no answer to read out of it.

    A model that answers with nothing has refused as surely as one that answers with
    prose, so an empty reply raises this too.
    """

    answer: str
    """
    What the model said instead.
    """

    shown: int = 400
    """
    How much of the answer the message quotes.
    """

    def error_message(self) -> str:
        return f"The model answered without JSON in it: {self.answer[: self.shown]}"

    def suggest_correction(self) -> str:
        return (
            "Put the question again saying what was wrong with the answer, or ask a "
            "model that holds to the shape it is asked for."
        )


@dataclass
class AnswerCutOffError(ModelRefusedError):
    """
    Raised when a reply stopped at the longest answer the question allowed, so whatever
    it holds is unfinished.

    A model that keeps repeating itself is stopped this way, and quoting what it
    repeated back to it is how it is led to repeat it again.
    """

    def error_message(self) -> str:
        return "The answer was cut off at the length limit before it was finished."

    def suggest_correction(self) -> str:
        return (
            "Answer with the JSON object alone, keep every reason to one sentence, and "
            "do not repeat yourself."
        )
