"""
Answer what the measurements and the ontology leave open about a scene's overlaps.

Two things stand between a labelled mesh and a world of bodies, and neither follows from
the geometry:

- **who owns a face two labels both claim**, without which the mesh cannot be split, since
  a face belongs to exactly one body;
- **which whole a part belongs to**, where a door meets more than one cabinet.

Both are asked as few times as they are actually open. Ownership is asked once per set of
claimants rather than once per pair of them, never where the ontology already settled every
relation inside the set, and once per *pattern* of classes rather than once per occurrence
-- a door and a window sharing a pane is one question however many glazed doors the room
has. Membership is asked where a part has more than one candidate, and, where the run is
told to, where it only lies near the wholes it could belong to.

This decides nothing itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    TextPart,
)
from typing_extensions import Any, Dict, Generic, List, Optional, TypeVar

from krrood.ormatic.utils import classproperty

from experiments.warsaw.pipeline.asking import AnswerType, Prompt, Question
from experiments.warsaw.pipeline.records import (
    Adjudications,
    MembershipAnswer,
    MembershipQuestion,
    OntologySlice,
    OpenQuestions,
    OwnershipAnswer,
    OwnershipQuestion,
    Relations,
)
from experiments.warsaw.world_loader.viewpoints import PictureKind
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep

AskedType = TypeVar("AskedType")
"""
The question record one kind of adjudication is put from.
"""

# %% what each picture of an overlap shows


class OverlapCaption(StrEnum):
    """
    What each render of an open question shows, in the order they are shown in.
    """

    CLOSEUP = "Picture 1 -- the objects alone, with nothing in front of them."
    """
    The objects by themselves.
    """

    CONTEXT = "Picture 2 -- where they are in the room, painted the same way."
    """
    The objects in the room around them.
    """

    PLAIN = "Picture 3 -- the same objects, in the colors they were scanned in."
    """
    The objects as they were scanned.
    """

    @property
    def shows(self) -> PictureKind:
        """
        :return: The kind of render this captions.
        """
        return PictureKind(self.name.lower())


# %% what is put to a model about an overlap


@dataclass(kw_only=True)
class OverlapDecision(Question[AnswerType], Generic[AnswerType, AskedType]):
    """
    What the pictures of a set of overlapping objects are shown to settle.
    """

    asked: AskedType
    """
    The question as the measurement wrote it.
    """

    labels: Dict[str, str]
    """
    Per segment, the label it carries.
    """

    renders_directory: Path
    """
    Where the renders shown with the question are kept.
    """

    @property
    def key(self) -> str:
        return f"{self.prompt.value}__{self.asked.name}"

    @property
    def shows_pictures(self) -> bool:
        """
        :return: Whether this question puts pictures to the model. A run told to show
            none renders none, and the question then reads as one asked from the measured
            shares alone.
        """
        return bool(self.asked.images)

    def pictures(self) -> List[MessagePart]:
        """
        :return: The renders as parts of a message, captioned.
        """
        named = {
            PictureKind.of_render(filename): filename for filename in self.asked.images
        }
        content: List[MessagePart] = []
        for caption in OverlapCaption:
            filename = named.get(caption.shows)
            if filename is None:
                continue
            content.append(TextPart(caption.value))
            content.append(ImagePart.from_file(self.renders_directory / filename))
        return content

    def painted(self) -> str:
        """
        :return: What each color in the pictures stands for.
        """
        lines = [
            f'{name} (labelled "{self.labels[name]}") is painted in the colour '
            f"{self.asked.legend[name]}"
            for name in self.asked.shown
            if name in self.asked.legend
        ]
        if "contested" in self.asked.legend:
            lines.append(
                "the faces all of them claim are painted in the colour "
                f"{self.asked.legend['contested']}"
            )
        return "\n".join(lines)

    def ontology(self) -> str:
        """
        :return: What the ontology holds about the objects in it, as a model reads it.
        """
        known: OntologySlice = self.asked.ontology
        if not known.read_as:
            return ""
        read_as = "\n".join(
            f"{name} was read as {class_name or 'no class'}"
            for name, class_name in known.read_as.items()
        )
        lines = [f"## What the ontology says\n{read_as}"]
        if known.classes:
            lines.append("\n".join(known.classes))
        lines.append(
            "\n".join(known.admits)
            if known.admits
            else "Between these classes it admits no mount at all."
        )
        return "\n\n".join(lines)

    def measured(self) -> str:
        """
        :return: What was measured of each object on its own.
        """
        return "\n".join(
            f"{one.name}: {one.faces} faces, {one.area} m2, "
            f"middle {one.height} m up, {one.pieces} piece(s)"
            for one in self.asked.measured
        )


# %% whose the contested faces are


@dataclass(kw_only=True)
class OwnershipDecision(OverlapDecision[OwnershipAnswer, OwnershipQuestion]):
    """
    Whose the faces a set of labels all claim are.
    """

    @classproperty
    def prompt(cls) -> Prompt:
        """
        :return: The prompt this question is put with, both halves of it.
        """
        return Prompt.OWNERSHIP

    def message(self) -> List[MessagePart]:
        return [
            TextPart(
                self.templates.render_document(
                    self.message_template,
                    pattern=self.asked.pattern,
                    ontology=self.ontology(),
                    painted=self.painted(),
                    measured=self.measured(),
                    shares=self.asked.shares,
                    shows_pictures=self.shows_pictures,
                    exemplar_faces=self.asked.exemplar_faces,
                    covers=len(self.asked.covers),
                    contested_faces=self.asked.contested_faces,
                )
            )
        ] + self.pictures()

    def read(self, response: ModelResponse) -> OwnershipAnswer:
        answered: Dict[str, Any] = response.parse_json()
        return OwnershipAnswer(
            name=self.asked.name,
            pattern=list(self.asked.pattern),
            owner=answered.get("owner"),
            confidence=answered.get("confidence"),
            reason=answered.get("reason"),
        )

    def refusal(self, refused: ModelRefusedError) -> OwnershipAnswer:
        return OwnershipAnswer(
            name=self.asked.name,
            pattern=list(self.asked.pattern),
            problems=[str(refused)],
        )

    def problems_with(self, answer: OwnershipAnswer) -> List[str]:
        if answer.owner in self.asked.pattern:
            return []
        return [
            f"{answer.owner!r} is not one of the owners to choose from: "
            f"{', '.join(self.asked.pattern)}"
        ]


# %% which whole a part belongs to


@dataclass(kw_only=True)
class MembershipDecision(OverlapDecision[MembershipAnswer, MembershipQuestion]):
    """
    Which of its candidates a part belongs to, if one of them has to be it.
    """

    @classproperty
    def prompt(cls) -> Prompt:
        """
        :return: The prompt this question is put with, both halves of it.
        """
        return Prompt.MEMBERSHIP

    @property
    def prompt_variables(self) -> Dict[str, Any]:
        return {
            **super().prompt_variables,
            "may_belong_to_none": self.asked.may_belong_to_none,
        }

    def message(self) -> List[MessagePart]:
        return [
            TextPart(
                self.templates.render_document(
                    self.message_template,
                    part=self.asked.part,
                    part_label=self.labels[self.asked.part],
                    ontology=self.ontology(),
                    measured=self.measured(),
                    candidates=self.asked.candidates,
                    painted=self.painted(),
                    shows_pictures=self.shows_pictures,
                )
            )
        ] + self.pictures()

    def read(self, response: ModelResponse) -> MembershipAnswer:
        answered: Dict[str, Any] = response.parse_json()
        fields = {one.name: one.field_name for one in self.asked.candidates}
        return MembershipAnswer(
            name=self.asked.name,
            part=self.asked.part,
            whole=answered.get("whole"),
            field_name=fields.get(answered.get("whole"), ""),
            confidence=answered.get("confidence"),
            reason=answered.get("reason"),
        )

    def refusal(self, refused: ModelRefusedError) -> MembershipAnswer:
        return MembershipAnswer(
            name=self.asked.name, part=self.asked.part, problems=[str(refused)]
        )

    def problems_with(self, answer: MembershipAnswer) -> List[str]:
        if answer.whole in self.asked.candidate_names:
            return []
        if answer.whole is None and self.asked.may_belong_to_none:
            return []
        return [
            f"{answer.whole!r} is not one of the wholes to choose from: "
            f"{', '.join(self.asked.candidate_names)}"
        ]


# %% settling every open question


@dataclass
class AdjudicateOverlaps(PipelineStep):
    """
    The answers to everything the measurements and the ontology leave open.
    """

    limit: Optional[int] = None
    """
    Ask only the first so many questions, rather than all of them.
    """

    @property
    def name(self) -> str:
        return "adjudicate what is left open"

    def carry_out(self) -> None:
        """
        Put every open question to the model and write the answers.

        A question that was drawn is put with its pictures and one that was not is put
        from its text alone, since the ontology, the measured shares and the candidates
        are text either way.
        """
        questions = self.run.read_record(RunFile.QUESTIONS, OpenQuestions)
        relations = self.run.read_record(RunFile.RELATIONS, Relations)
        labels = relations.labels
        renders = self.run.path(RunFile.QUESTION_RENDERS)
        questioner = self.questioner(RunFile.QUESTION_ANSWERS)

        ownership = [
            OwnershipDecision(asked=one, labels=labels, renders_directory=renders)
            for one in questions.ownership
        ]
        membership = [
            MembershipDecision(asked=one, labels=labels, renders_directory=renders)
            for one in questions.membership
        ]
        every_question = ownership + membership
        asked = every_question[: self.limit] if self.limit else every_question

        self.logger.info(
            "asking %s about %s questions, %s of them with pictures ...",
            self.settings.model.value,
            len(asked),
            sum(1 for one in asked if one.shows_pictures),
        )
        adjudications = Adjudications(
            model=self.settings.model.value, scene=relations.scene
        )
        for question in asked:
            answered = questioner.answer(question)
            answer = answered.answer
            answer.problems = answered.problems
            adjudications.answered.append(answer)
            if isinstance(answer, OwnershipAnswer):
                self.logger.info("  %-44s -> %s", answer.name, answer.owner)
            else:
                self.logger.info("  %-44s in %s", answer.name, answer.whole)
            for problem in answer.problems:
                self.logger.warning("      ! %s", problem)

        self.run.write_record(RunFile.ADJUDICATIONS, adjudications)
        self.report(adjudications)

    def report(self, adjudications: Adjudications) -> None:
        """
        :param adjudications: What was answered.
        """
        troubled = [one for one in adjudications.answered if one.problems]
        self.logger.info(
            "%s patterns and %s memberships answered, %s with problems",
            len(adjudications.ownership),
            len(adjudications.membership),
            len(troubled),
        )
        self.logger.info("written to %s", self.run.path(RunFile.ADJUDICATIONS))
