"""
Putting a question to a model, and putting an unusable answer back to it.

Four of the pipeline's steps ask a model something: which class a label means, whose a
contested face is, which whole a part belongs to, what each body is. They differ in what
is said and how the answer is read, and agree on everything around that -- keeping the
reply as it came back, checking the answer is usable, and asking once more with what was
wrong when it is not.

An answer naming a class that is not in the taxonomy is worth nothing, and saying so is
cheaper than either dropping the question or letting a person correct it by hand.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from semantic_digital_twin.adapters.vision_language_model.client import (
    ModelResponse,
    VisionLanguageModel,
)
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, Generic, List, Sequence, Set, Type, TypeVar

from krrood.ormatic.utils import classproperty

from experiments.warsaw.bases import HasLogger
from experiments.warsaw.pipeline.model_calls import ModelCallPart, ModelCallTrace
from experiments.warsaw.pipeline.templates import PipelineTemplates

AnswerType = TypeVar("AnswerType")
"""
What reading one kind of question's answer produces.
"""

# %% the two halves of a prompt


class PromptHalf(StrEnum):
    """
    One of the two halves of what a question puts to a model.

    They go into the request as two messages, in two roles, and they have different
    lifetimes: the standing instruction is sent unchanged however often an unusable answer
    is put back, while what is being asked about is built afresh each time. Both are kept
    in one directory per question, so the pair is found together.

    The value is the file each is kept in. Only one of them interpolates anything, so only
    one of them is named as a template; both are read the same way, since a template with
    nothing to fill in renders as itself.
    """

    SYSTEM = "system.md"
    """
    What the model is told it is doing.
    """

    MESSAGE = "message.md.jinja"
    """
    What it is being asked about this time.
    """


class Prompt(StrEnum):
    """
    One of the prompts the pipeline puts its questions with.

    A prompt is a directory holding both halves, and there is one per kind of question.
    Which one a question uses is a property of the kind of question it is rather than
    something an instance is told, so it is named here and answered by the class.
    """

    VOCABULARY = "vocabulary"
    """
    Which class of the ontology a label means.
    """

    OWNERSHIP = "ownership"
    """
    Whose the faces several labels all claim are.
    """

    MEMBERSHIP = "membership"
    """
    Which of several wholes a part belongs to.
    """

    CLASSIFICATION = "classification"
    """
    What each body of a split scene is.
    """

    CLASS_PROPOSAL = "class_proposal"
    """
    What new class a label means that no class of the ontology was found for.
    """

    CLASS_CHOICE = "class_choice"
    """
    Which of the classes its label was left open between each body of a split scene is.
    """

    SUPERCLASS = "superclass"
    """
    What a class a run proposed is a kind of.
    """

    TAXONOMY_AMENDMENT = "taxonomy_amendment"
    """
    Whether a class of the ontology should be given a mixin.
    """


# %% one thing to ask


@dataclass(kw_only=True)
class Question(ABC, Generic[AnswerType]):
    """
    One thing to ask a model, and the reading of what comes back.

    A question names the prompt it is put with, and both halves of that prompt are found
    from the name: naming one thing rather than two files is what keeps a question from
    being asked with one half of one prompt and one half of another.
    """

    @property
    @abstractmethod
    def prompt(self) -> Prompt:
        """
        :return: The prompt this question is put with, both halves of it.
        """

    @classproperty
    def prompts_directory(cls) -> Path:
        """
        The prompts a step puts its questions with are kept in the step's own directory,
        which is what makes a step one thing to read: what it asks, how it reads the
        answer, and the words it asks in are all in one place.

        :return: Where this question's prompts are, beside the module declaring it.
        """
        return Path(sys.modules[cls.__module__].__file__).resolve().parent / "prompts"

    @classproperty
    def templates(cls) -> PipelineTemplates:
        """
        :return: Where the words put to the model are written from.
        """
        return PipelineTemplates(directory=cls.prompts_directory)

    def prompt_template(self, half: PromptHalf) -> str:
        """
        :param half: Which half of the prompt is wanted.
        :return: The template it is written from, by file name.
        """
        return f"{self.prompt.value}/{half.value}"

    @property
    def message_template(self) -> str:
        """
        :return: What the model is asked about, by template name.
        """
        return self.prompt_template(PromptHalf.MESSAGE)

    @property
    @abstractmethod
    def key(self) -> str:
        """
        :return: What names this question, which is what its kept reply is filed under.
        """

    @property
    def shows_pictures(self) -> bool:
        """
        :return: Whether this question puts pictures to the model. A question that can be
            asked either way answers from what it was actually given, so the instruction
            and the message cannot disagree about whether there is anything to look at.
        """
        return True

    @property
    def prompt_variables(self) -> Dict[str, Any]:
        """
        :return: What the standing instruction is written for, by the name its template
            reads it under.
        """
        return {"shows_pictures": self.shows_pictures}

    @property
    def system_prompt(self) -> str:
        """
        :return: What the model is told it is doing.
        """
        return self.templates.render_document(
            self.prompt_template(PromptHalf.SYSTEM), **self.prompt_variables
        )

    @abstractmethod
    def message(self) -> List[MessagePart]:
        """
        :return: The question, in the order it is to be read.
        """

    @abstractmethod
    def read(self, response: ModelResponse) -> AnswerType:
        """
        :param response: What the model answered with.
        :return: The answer, in the shape the pipeline keeps it.
        :raises ModelRefusedError: If the reply holds no answer at all.
        """

    @abstractmethod
    def problems_with(self, answer: AnswerType) -> List[str]:
        """
        :param answer: What was read out of the reply.
        :return: One sentence per thing that makes it unusable, empty when nothing does.
        """

    @abstractmethod
    def refusal(self, refused: ModelRefusedError) -> AnswerType:
        """
        :param refused: The reply that held no answer.
        :return: The blank answer to keep in its place, so a refusal costs one question
            rather than the run.
        """


# %% the classes no object may be answered with


@dataclass
class TaxonomyCategories:
    """
    The classes a taxonomy marks as categories: they say what kind of thing something is,
    not what it is.

    Read from the taxonomy the model was shown rather than from the ontology, so that what
    an answer is refused for is what the model was told, and a run accepting categories
    exports none.
    """

    taxonomy: Dict[str, Any]
    """
    The ontology as a model reads it.
    """

    @property
    def names(self) -> Set[str]:
        """
        :return: The categories. A mixin is marked the same way and refused for being a
            mixin, so it is left out here.
        """
        return {
            node["name"]
            for node in self.taxonomy["classes"]
            if node.get("abstract") and not node.get("mixin")
        }

    def problems_with(self, class_name: str, is_new_class: bool) -> List[str]:
        """
        Say whether an answer names a category as what something is.

        :param class_name: The class answered.
        :param is_new_class: Whether that class was proposed rather than taken from the
            taxonomy. A proposal deriving from a category is what a category is for.
        :return: One sentence when it names a category, empty otherwise.
        """
        if is_new_class or class_name not in self.names:
            return []
        return [
            f"{class_name} is a category, so it says what kind of thing this is rather "
            f"than what it is; propose the kind as a new class with {class_name} as its "
            f"superclass instead of answering with it"
        ]


# %% a question asked about a class of the ontology


@dataclass(kw_only=True)
class QuestionAboutTheOntology(Question[AnswerType], Generic[AnswerType]):
    """
    A question about what a class of the ontology is or what it may hold.

    Both such questions are put the same way: the model is shown the taxonomy, is checked
    against the classes standing in it, and is given renders of an object of each class
    being discussed.
    """

    taxonomy: Dict[str, Any] = field(default_factory=dict)
    """
    The ontology as a model reads it.
    """

    known: Dict[str, Type] = field(default_factory=dict)
    """
    The ontology's classes by name, for checking an answer against.
    """

    renders_directory: Path = field(default_factory=Path)
    """
    Where the renders shown with the question are kept.
    """


# %% what came back


@dataclass
class Answered(Generic[AnswerType]):
    """
    What one question was answered with, after as many attempts as it was worth.
    """

    answer: AnswerType
    """
    What was read out of the last reply.
    """

    problems: List[str] = field(default_factory=list)
    """
    What still makes it unusable, empty when nothing does.
    """

    attempts: int = 1
    """
    How many times it was asked.
    """

    @property
    def is_usable(self) -> bool:
        """
        :return: Whether the answer can be acted on.
        """
        return not self.problems


# %% putting a question, and putting it again


@dataclass(frozen=True)
class QuestionExchange:
    """
    One reply, with the message that asked for it and how long it took.
    """

    response: ModelResponse
    """
    The reply as the model returned it.
    """

    message: List[MessagePart]
    """
    The parts of the message, in the order they were sent.
    """

    started_at: str
    """
    When the question was put, in UTC.
    """

    elapsed_seconds: float
    """
    How long the reply took to arrive.
    """


@dataclass
class Questioner(HasLogger):
    """
    A model being asked the pipeline's questions, with every reply kept.
    """

    model: VisionLanguageModel
    """
    The model to ask.
    """

    answers_directory: Path
    """
    Where the replies are kept, as they came back.
    """

    corrections: int = 1
    """
    How often an unusable answer is put back to the model with what was wrong with it.
    """

    correction_preamble: str = "Your previous answer could not be used:"
    """
    How the model is told what was wrong with what it said.
    """

    correction_closing: str = "Answer the same question again, correcting that."
    """
    How that correction ends.
    """

    traces_directory: Path | None = None
    """
    Where every attempt is kept, request and response together, or None to keep none.
    """

    requested_model: str = ""
    """
    The model the requests are addressed to, as every trace records it.
    """

    def answer(self, question: Question[AnswerType]) -> Answered[AnswerType]:
        """
        Put one question, correcting an unusable answer while it is worth doing.

        :param question: What to ask and how to read the answer.
        :return: The answer, and what is still wrong with it.
        """
        problems: Sequence[str] = ()
        answered = None
        for attempt in range(1 + self.corrections):
            exchange = self.exchange(question, problems)
            try:
                answer = question.read(exchange.response)
            except ModelRefusedError as refused:
                answer, problems = question.refusal(refused), [str(refused)]
            else:
                problems = question.problems_with(answer)
            self.keep_trace(question, attempt + 1, exchange, problems)
            answered = Answered(
                answer=answer, problems=list(problems), attempts=attempt + 1
            )
            if answered.is_usable:
                return answered
            if attempt < self.corrections:
                self.logger.info("%s: %s, asking again ...", question.key, problems[0])
        return answered

    def exchange(
        self, question: Question[AnswerType], problems: Sequence[str]
    ) -> QuestionExchange:
        """
        Put one question to the model and keep the reply as it came back.

        :param question: What to ask.
        :param problems: What was wrong with the answer to the same question, when this is
            another attempt at it.
        :return: The reply, with the message that asked for it and how long it took.
        """
        message = list(question.message())
        if problems:
            message.append(TextPart(self.correction_of(problems)))
        started_at = datetime.now(UTC).isoformat()

        start = perf_counter()
        response = self.model.ask(message, question.system_prompt)
        elapsed_seconds = perf_counter() - start
        self.answers_directory.mkdir(parents=True, exist_ok=True)
        self.kept_path(question).write_text(json.dumps(response.to_json(), indent=2))
        return QuestionExchange(
            response=response,
            message=message,
            started_at=started_at,
            elapsed_seconds=elapsed_seconds,
        )

    def keep_trace(
        self,
        question: Question[AnswerType],
        attempt: int,
        exchange: QuestionExchange,
        problems: Sequence[str],
    ) -> None:
        """
        Keep one attempt beside the others made at the same question.

        :param question: The question the attempt answers.
        :param attempt: Which attempt it was, counting from one.
        :param exchange: What was sent, what came back, and how long it took.
        :param problems: What made the answer unusable, empty when nothing did.
        """
        if self.traces_directory is None:
            return
        directory = self.traces_directory / question.key
        directory.mkdir(parents=True, exist_ok=True)
        trace = ModelCallTrace(
            question=question.key,
            attempt=attempt,
            requested_model=self.requested_model,
            system_prompt=question.system_prompt,
            message_parts=[ModelCallPart.of(part) for part in exchange.message],
            response=exchange.response.to_json(),
            problems=list(problems),
            started_at=exchange.started_at,
            elapsed_seconds=exchange.elapsed_seconds,
        )
        (directory / f"attempt_{attempt}.json").write_text(
            json.dumps(trace.to_json(), indent=2)
        )

    def kept_path(self, question: Question[AnswerType]) -> Path:
        """
        :param question: The question whose reply is kept.
        :return: Where that reply is kept.
        """
        return self.answers_directory / f"{question.key}.json"

    def correction_of(self, problems: Sequence[str]) -> str:
        """
        :param problems: What was wrong with the previous answer.
        :return: What to tell the model about it.
        """
        return (
            f"{self.correction_preamble}\n- "
            + "\n- ".join(problems)
            + f"\n{self.correction_closing}"
        )
