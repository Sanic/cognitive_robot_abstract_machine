"""
Ask a model which class of the ontology each of a scene's labels means.

A scanned scene labels its faces with words a human annotator chose: ``cabinet``,
``kitchen_island``, ``jar``. Those are not classes. Some have a class of the same name that
means something else, some have no class at all, and nothing in the mesh says which is
which -- so this is the one question in the pipeline that has to be asked before any
conflict between labels can be resolved, and it is asked once per label rather than once
per object.

An answer is either a class of the ontology or a new class named by the superclass and
mixins it is composed of, and those decide what it can hold: a ``KitchenIsland`` composed
with ``HasDrawers`` admits the drawers overlapping it as parts, one without it admits
nothing. Every answer is checked here -- names against the ontology, compositions by
building the class -- so what is written out is known to be usable rather than merely well
spelled.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from semantic_digital_twin.adapters.vision_language_model.client import ModelResponse
from semantic_digital_twin.adapters.vision_language_model.exceptions import (
    ModelRefusedError,
)
from semantic_digital_twin.adapters.vision_language_model.message import (
    ImagePart,
    MessagePart,
    TextPart,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    compose_class,
)
from typing_extensions import Any, Dict, List, Type

from krrood.ormatic.utils import classproperty

from experiments.warsaw.pipeline.asking import (
    Prompt,
    QuestionAboutTheOntology,
    TaxonomyCategories,
)
from experiments.warsaw.pipeline.records import (
    LabelAnswer,
    LabelRequest,
    Relations,
    Vocabulary,
    VocabularyRequest,
)
from experiments.warsaw.world_loader.viewpoints import PictureKind
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep

# %% what each picture of a label shows


class ExemplarCaption(StrEnum):
    """
    What each of an exemplar's three renders shows.

    The order is the order the question is answered in: where it stands, what it looks
    like, and then which faces are actually being asked about.
    """

    CONTEXT = "Picture 1 -- where in the room it is, painted in the colour {color}."
    """
    The object in the room around it.
    """

    PLAIN = "Picture 2 -- the object alone, in the colors it was scanned in."
    """
    The object as it was scanned.
    """

    CLOSEUP = (
        "Picture 3 -- the same object alone, painted in the colour {color}, which is "
        "exactly the faces the label covers."
    )
    """
    The object alone, painted.
    """

    @property
    def shows(self) -> PictureKind:
        """
        :return: The kind of render this captions.
        """
        return PictureKind(self.name.lower())


# %% what one label means


@dataclass(kw_only=True)
class LabelQuestion(QuestionAboutTheOntology[LabelAnswer]):
    """
    What one of a scene's labels means, put to a model with pictures of one such object.
    """

    label: LabelRequest
    """
    The label being asked about, and the object standing for it.
    """

    every_label: List[str]
    """
    Every label of the scene, since which of them exist alongside a label says what it was
    left to mean: a room that labels handles separately does not mean them by ``drawer``.
    """

    meets: str = ""
    """
    What the pictured object meets, as the scan measures it.
    """

    @classproperty
    def prompt(cls) -> Prompt:
        """
        :return: The prompt this question is put with, both halves of it.
        """
        return Prompt.VOCABULARY

    @property
    def key(self) -> str:
        return self.label.label

    @property
    def shows_pictures(self) -> bool:
        """
        :return: Whether this label was rendered. A run told to show no pictures renders
            none, and the question then reads as one asked from the label alone.
        """
        return bool(self.label.images)

    @property
    def mixin_names(self) -> List[str]:
        """
        :return: The names a new class may be composed from.
        """
        return [mixin["name"] for mixin in self.taxonomy["part_whole_mixins"]]

    @property
    def building_block_names(self) -> List[str]:
        """
        :return: The classes that exist to be built with, which the ontology derives
            concrete classes from -- a floor is a HasSupportingSurface -- but which name
            nothing standing in a room themselves.
        """
        return [node["name"] for node in self.taxonomy["classes"] if node.get("mixin")]

    def message(self) -> List[MessagePart]:
        named = {
            PictureKind.of_render(filename): filename for filename in self.label.images
        }
        content: List[MessagePart] = [
            TextPart(
                self.templates.render_document(
                    self.message_template,
                    taxonomy=json.dumps(self.taxonomy),
                    label=self.label.label,
                    instances=self.label.instances,
                    every_label=self.every_label,
                    meets=self.meets,
                    exemplar=self.label.exemplar,
                )
            )
        ]
        for caption in ExemplarCaption:
            filename = named.get(caption.shows)
            if filename is None:
                continue
            content.append(TextPart(caption.value.format(color=self.label.color)))
            content.append(ImagePart.from_file(self.renders_directory / filename))
        return content

    def read(self, response: ModelResponse) -> LabelAnswer:
        return LabelAnswer.spoken(response.parse_json())

    def refusal(self, refused: ModelRefusedError) -> LabelAnswer:
        return LabelAnswer(problems=[str(refused)])

    def problems_with(self, answer: LabelAnswer) -> List[str]:
        """
        Say what is wrong with an answer, if anything.

        An answer is only worth as much as it is usable: a class name that is not in the
        ontology, or a composition that cannot be built, would come back as an unmapped
        label two steps later, where it would look like the scene's fault rather than the
        answer's.

        :param answer: What the model said.
        :return: One sentence per problem, empty when there are none.
        """
        problems: List[str] = []
        if answer.class_name is None:
            return problems

        if answer.class_name in self.building_block_names:
            problems.append(
                f"{answer.class_name} is a mixin, so it says what a class can hold "
                f"rather than what it is; derive a class from it instead of answering "
                f"with it"
            )
        problems.extend(
            TaxonomyCategories(taxonomy=self.taxonomy).problems_with(
                answer.class_name, answer.is_new_class
            )
        )
        if not answer.is_new_class:
            if answer.class_name not in self.known:
                problems.append(
                    f"{answer.class_name} is not in the taxonomy and was not proposed "
                    f"as new"
                )
            return problems

        if answer.class_name in self.known:
            problems.append(
                f"{answer.class_name} is proposed as new but is already in the taxonomy"
            )
        if answer.superclass not in self.known:
            problems.append(
                f"the superclass {answer.superclass!r} is not in the taxonomy"
            )
        for mixin in answer.mixins:
            if mixin not in self.building_block_names:
                problems.append(f"{mixin!r} is not one of the taxonomy's mixins")
        if problems:
            return problems

        try:
            compose_class(
                answer.class_name,
                self.known[answer.superclass],
                [self.known[mixin] for mixin in answer.mixins],
            )
        except TypeError as failure:
            # A composition is a model's proposal; one Python refuses to build is an
            # answer to put back to it rather than a state the run should not have reached.
            problems.append(f"the composition cannot be built: {failure}")
        return problems


# %% mapping every label of the scene


@dataclass
class MapLabelVocabulary(PipelineStep):
    """
    Which class of the ontology each of a scene's labels means.
    """

    only_labels: List[str] = field(default_factory=list)
    """
    Ask about only these labels, rather than all of them.
    """

    @property
    def name(self) -> str:
        return "map the labels onto classes"

    def carry_out(self) -> None:
        """
        Ask about every label, check what comes back, and write the mapping.
        """
        request = self.run.read_record(RunFile.VOCABULARY_REQUEST, VocabularyRequest)
        taxonomy = self.run.read_json(RunFile.TAXONOMY)
        relations = self.run.read_record(RunFile.RELATIONS, Relations)
        known = self.ontology_classes()
        questioner = self.questioner(RunFile.VOCABULARY_ANSWERS)

        asked = [
            entry
            for entry in request.labels
            if not self.only_labels or entry.label in self.only_labels
        ]
        self.logger.info(
            "asking %s about %s of %s labels ...",
            self.settings.model.value,
            len(asked),
            len(request.labels),
        )

        vocabulary = Vocabulary(
            model=self.settings.model.value, scene=request.scene, labels=[]
        )
        for entry in asked:
            answered = questioner.answer(
                LabelQuestion(
                    label=entry,
                    every_label=request.label_names,
                    taxonomy=taxonomy,
                    known=known,
                    renders_directory=self.run.path(RunFile.EXEMPLARS),
                    meets=self.meetings(relations, entry.exemplar),
                )
            )
            answer = answered.answer
            answer.problems = answered.problems
            answer.exemplar = entry.exemplar
            answer.label = entry.label
            vocabulary.labels.append(answer)
            self.report_answer(entry.label, answer)

        self.run.write_record(RunFile.VOCABULARY, vocabulary)
        self.report(vocabulary)

    @staticmethod
    def meetings(relations: Relations, exemplar: str) -> str:
        """
        Say what the pictured object meets, as the scan measures it.

        Which labels cover the same surface is the evidence a composition needs and a
        picture does not carry: an island whose faces are also labelled ``drawer`` has to
        be given a class that can hold drawers, or nothing will ever be mounted into it. It
        is reported as a measurement and said to be one, since sharing a surface is not yet
        a relation.

        :param relations: What the measurement wrote.
        :param exemplar: The name of the segment being pictured.
        :return: The measurements as the text a model reads, empty when it meets nothing.
        """
        labels = relations.labels
        overlapping: Counter = Counter()
        touching: Counter = Counter()
        for pair in relations.pairs:
            if exemplar not in (pair.evidence.one, pair.evidence.other):
                continue
            other = (
                pair.evidence.other
                if pair.evidence.one == exemplar
                else pair.evidence.one
            )
            if pair.evidence.shared_faces:
                overlapping[labels[other]] += 1
            elif pair.evidence.touching_edges:
                touching[labels[other]] += 1

        def counted(counts: Counter) -> str:
            return ", ".join(
                f"{label} ({count})" for label, count in counts.most_common()
            )

        lines = []
        if overlapping:
            lines.append(
                f"Objects of these labels are labelled onto some of the same faces: "
                f"{counted(overlapping)}."
            )
        if touching:
            lines.append(
                f"Objects of these labels touch it along edges without sharing faces: "
                f"{counted(touching)}."
            )
        if not lines:
            return ""
        lines.append(
            "That is measurement only: sharing a surface says these labels cover the "
            "same geometry, not which of them holds the other."
        )
        return "## What it meets, measured on the scan\n" + "\n".join(lines)

    def report_answer(self, label: str, answer: LabelAnswer) -> None:
        """
        Say what was answered about one label.

        :param label: The label that was asked about.
        :param answer: What came back.
        """
        composition = (
            f" ({', '.join([answer.superclass or ''] + answer.mixins)})"
            if answer.is_new_class
            else ""
        )
        new = " [new]" if answer.is_new_class else ""
        self.logger.info(
            "  %-18s -> %s%s%s",
            label,
            answer.class_name or "-- nothing",
            composition,
            new,
        )
        for problem in answer.problems:
            self.logger.warning("      ! %s", problem)

    def report(self, vocabulary: Vocabulary) -> None:
        """
        :param vocabulary: What was answered about every label.
        """
        mapped = [one for one in vocabulary.labels if one.class_name]
        proposed = [one for one in mapped if one.is_new_class]
        troubled = [one for one in vocabulary.labels if one.problems]
        self.logger.info(
            "%s of %s labels mapped, %s of them to new classes, %s with problems",
            len(mapped),
            len(vocabulary.labels),
            len(proposed),
            len(troubled),
        )
        self.logger.info("written to %s", self.run.path(RunFile.VOCABULARY))
