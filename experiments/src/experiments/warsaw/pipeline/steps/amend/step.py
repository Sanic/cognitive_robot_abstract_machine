"""
Ask whether the ontology, rather than the room, is what is wrong -- and mend it.

When objects of two classes are measured to share a surface and no field of either
admits the other as a structural part, one of two things is true: those objects want a
class of their own, or the class they were read as is missing a mixin. This puts the
second to a model, one class pair at a time, and carries out what it accepts.

An amendment is not a pipeline output. It edits the class in the ontology's own source
and regenerates the ORM, so it holds for every room, every world already in the database
and everything built on the ontology afterwards -- which is why the model is asked about
the class rather than about the objects, and why applying it is a separate setting.

Which mixin would grant a relation is never guessed: it is found by composing the class
with each mixin and asking the same question a mount asks. The model decides only
whether the class should have it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from semantic_digital_twin.semantic_annotations.part_whole import part_whole_fields
from semantic_digital_twin.semantic_annotations.taxonomy_amendment import (
    CannotAmendClass,
    SourceAmendment,
    amend_class_source,
    granting_mixins,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    describe_class,
)
from typing_extensions import Any, Dict, List, Optional, Tuple, Type

from krrood.ormatic.utils import classproperty

from experiments.warsaw.pipeline.asking import Prompt, QuestionAboutTheOntology
from experiments.warsaw.pipeline.database.orm_rebuild import OrmRebuild
from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import (
    AmendmentRecord,
    Relations,
    SourceEdit,
    VocabularyRequest,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.amend.amended_classes_check import (
    AmendedClassesCheck,
    AmendedPair,
)
from experiments.warsaw.pipeline.steps.step import PipelineStep

# %% whether a class is missing a structural part


@dataclass(kw_only=True)
class MixinProposal(QuestionAboutTheOntology[AmendmentRecord]):
    """
    Whether one class of the ontology should be given one mixin.
    """

    record: AmendmentRecord
    """
    The amendment in question, and what was measured that raised it.
    """

    request: VocabularyRequest
    """
    The vocabulary question, for the exemplar of a label.
    """

    plain_render_marker: str = "__plain_"
    """
    What marks the render showing an object in the colors it was scanned in, which says
    nothing about which faces the label covers and is left out.
    """

    @classproperty
    def prompt(cls) -> Prompt:
        """
        :return: The prompt this question is put with, both halves of it.
        """
        return Prompt.TAXONOMY_AMENDMENT

    @property
    def key(self) -> str:
        return "-".join((self.record.whole, self.record.mixin, self.record.part))

    def held_parts(self, annotation_class: Type) -> List[Type]:
        """
        Report the classes a class can hold as structural parts.

        :param annotation_class: The class to inspect.
        :return: The type each of its part-whole fields accepts, each once, in the order
            the fields are declared.
        """
        held = []
        for relation in part_whole_fields(annotation_class):
            if isinstance(relation.part, type) and relation.part not in held:
                held.append(relation.part)
        return held

    def message(self) -> List[MessagePart]:
        introduces = next(
            mixin["introduces"]
            for mixin in self.taxonomy["part_whole_mixins"]
            if mixin["name"] == self.record.mixin
        )
        granted = ", ".join(
            f"{relation['field']} -> {relation['target']}" for relation in introduces
        )
        # What the class holds is not enough to judge by, since what *those* hold is a
        # relation further away and often decides it -- a cabinet holds doors and a door
        # holds a handle. Reported as structure and nothing else: what follows from it is
        # the question being asked, not something to answer in the asking.
        onwards = [
            describe_class(held)
            for held in self.held_parts(self.known[self.record.whole])
        ]
        content: List[MessagePart] = [
            TextPart(
                self.templates.render_document(
                    self.message_template,
                    whole=self.record.whole,
                    part=self.record.part,
                    mixin=self.record.mixin,
                    granted=granted,
                    whole_as_it_stands=describe_class(self.known[self.record.whole]),
                    part_as_it_stands=describe_class(self.known[self.record.part]),
                    held_onwards=onwards,
                    whole_labels=sorted(self.record.whole_labels),
                    part_labels=sorted(self.record.part_labels),
                    measured_pairs=self.record.measured_pairs,
                    shared_faces=self.record.shared_faces,
                )
            )
        ]
        content.extend(self.exemplar_pictures())
        return content

    def exemplar_pictures(self) -> List[MessagePart]:
        """
        :return: One object of each class, painted, so the classes can be seen.
        """
        entries = {entry.label: entry for entry in self.request.labels}
        content: List[MessagePart] = []
        for role, labels in (
            (self.record.whole, self.record.whole_labels),
            (self.record.part, self.record.part_labels),
        ):
            if not labels:
                continue
            label = sorted(labels)[0]
            entry = entries.get(label)
            if entry is None:
                continue
            for filename in entry.images:
                if self.plain_render_marker in filename:
                    continue
                content.append(
                    TextPart(
                        f'An object labelled "{label}", read as {role}, painted '
                        f"{entry.color}."
                    )
                )
                content.append(ImagePart.from_file(self.renders_directory / filename))
        return content

    def read(self, response: ModelResponse) -> AmendmentRecord:
        answered = response.parse_json()
        judged = replace(self.record)
        judged.amend = bool(answered.get("amend"))
        judged.confidence = answered.get("confidence")
        judged.reason = answered.get("reason")
        return judged

    def refusal(self, refused: ModelRefusedError) -> AmendmentRecord:
        judged = replace(self.record)
        judged.reason = str(refused)
        return judged

    def problems_with(self, answer: AmendmentRecord) -> List[str]:
        """
        :param answer: What the model said.
        :return: Nothing: yes and no are both usable answers, and a refusal is read as no.
        """
        return []


# %% proposing and carrying out the amendments


@dataclass
class AmendTaxonomy(PipelineStep):
    """
    The mixins a scene's measurements raise, judged and, when asked for, carried out.
    """

    only: List[str] = field(default_factory=list)
    """
    Consider only these amendments, named ``<Class>+<Mixin>``, rather than every one the
    measurements raise.
    """

    @property
    def name(self) -> str:
        return "ask whether the ontology is missing a relation"

    @property
    def is_optional(self) -> bool:
        return True

    def carry_out(self) -> None:
        """
        Judge every amendment the measurements raise, and apply the accepted ones if
        asked.
        """
        relations = self.run.read_record(RunFile.RELATIONS, Relations)
        request = self.run.read_record(RunFile.VOCABULARY_REQUEST, VocabularyRequest)
        vocabulary = self.run.read_record(RunFile.VOCABULARY, Vocabulary)
        taxonomy = self.run.read_json(RunFile.TAXONOMY)

        known = self.ontology_classes()
        mixins = [known[mixin["name"]] for mixin in taxonomy["part_whole_mixins"]]
        classes = VocabularyClasses(vocabulary=vocabulary, known=known).by_label()

        raised = self.candidates(relations, classes, known, mixins)
        if self.only:
            wanted = set(self.only)
            raised = [one for one in raised if f"{one.whole}+{one.mixin}" in wanted]
        self.logger.info("%s amendments are raised by what was measured:", len(raised))
        for one in raised:
            self.logger.info(
                "  %s + %s to hold %s  (%s pairs, %s & %s)",
                one.whole,
                one.mixin,
                one.part,
                one.measured_pairs,
                "/".join(sorted(one.whole_labels)),
                "/".join(sorted(one.part_labels)),
            )

        judged, accepted = self.judge(raised, known, taxonomy, request)
        self.run.write_json(
            RunFile.TAXONOMY_AMENDMENTS, [one.to_json() for one in judged]
        )
        self.logger.info(
            "%s of %s accepted; written to %s",
            len(accepted),
            len(judged),
            self.run.path(RunFile.TAXONOMY_AMENDMENTS),
        )

        if not accepted:
            return
        if not self.settings.amend_the_ontology:
            self.logger.info(
                "set amend_the_ontology to carry them out for the length of this run"
            )
            return

        self.logger.info("applying:")
        self.apply([amendment for _, amendment in accepted])
        written = {(record.whole, record.mixin) for record, _ in accepted}
        for record in judged:
            if (record.whole, record.mixin) in written:
                record.applied = True
        self.run.write_json(
            RunFile.TAXONOMY_AMENDMENTS, [one.to_json() for one in judged]
        )

    def candidates(
        self,
        relations: Relations,
        classes: Dict[str, Optional[Type]],
        known: Dict[str, Type],
        mixins: List[Type],
    ) -> List[AmendmentRecord]:
        """
        Find the amendments the scene's measurements raise.

        Only overlapping pairs are considered, and only classes the ontology has written
        down: a class proposed for this scene was given its mixins when it was proposed,
        and amending it would mean amending a proposal.

        Whether a relation is already admissible is asked of the classes rather than read
        from the status the measurements were written with, which was decided by whichever
        vocabulary that run was given and may not be the one being read now.

        :param relations: What the measurement wrote.
        :param classes: Per label, the class it was read as.
        :param known: The ontology's classes by name, before anything was composed.
        :param mixins: The mixins a class can be given.
        :return: One record per class, mixin and part, with what raised it gathered.
        """
        gathered: Dict[Tuple[str, str, str], AmendmentRecord] = {}
        for pair in relations.pairs:
            if not pair.evidence.shared_faces:
                continue
            labels = list(pair.classes)
            for one, other in (labels, labels[::-1]):
                whole, part = classes.get(one), classes.get(other)
                if whole is None or part is None:
                    continue
                if known.get(whole.__name__) is not whole:
                    continue
                for mixin in granting_mixins(whole, part, mixins):
                    key = (whole.__name__, mixin.__name__, part.__name__)
                    record = gathered.setdefault(
                        key,
                        AmendmentRecord(
                            whole=whole.__name__,
                            mixin=mixin.__name__,
                            part=part.__name__,
                        ),
                    )
                    if one not in record.whole_labels:
                        record.whole_labels.append(one)
                    if other not in record.part_labels:
                        record.part_labels.append(other)
                    record.measured_pairs += 1
                    record.shared_faces += pair.evidence.shared_faces
        return sorted(gathered.values(), key=lambda one: -one.measured_pairs)

    def judge(
        self,
        raised: List[AmendmentRecord],
        known: Dict[str, Type],
        taxonomy: Dict[str, Any],
        request: VocabularyRequest,
    ) -> Tuple[List[AmendmentRecord], List[Tuple[AmendmentRecord, SourceAmendment]]]:
        """
        Put every raised amendment to the model, and work out which can be written.

        :param raised: The amendments the measurements raise.
        :param known: The ontology's classes by name.
        :param taxonomy: The ontology as a model reads it.
        :param request: The vocabulary question, for the exemplar renders.
        :return: Every judgement, and the accepted ones with the edit each would make.
        """
        questioner = self.questioner(RunFile.AMENDMENT_ANSWERS)
        self.logger.info("asking %s about each ...", self.settings.model.value)

        judged: List[AmendmentRecord] = []
        accepted: List[Tuple[AmendmentRecord, SourceAmendment]] = []
        for record in raised:
            answered = questioner.answer(
                MixinProposal(
                    record=record,
                    known=known,
                    taxonomy=taxonomy,
                    request=request,
                    renders_directory=self.run.path(RunFile.EXEMPLARS),
                )
            )
            judgement = answered.answer
            self.logger.info(
                "  %-3s %s + %s: %s",
                "yes" if judgement.amend else "no",
                judgement.whole,
                judgement.mixin,
                judgement.reason,
            )
            if judgement.amend:
                self.plan_edit(judgement, known, accepted)
            judged.append(judgement)
        return judged, accepted

    def plan_edit(
        self,
        judgement: AmendmentRecord,
        known: Dict[str, Type],
        accepted: List[Tuple[AmendmentRecord, SourceAmendment]],
    ) -> None:
        """
        Work out the line an accepted amendment would change, and record why it cannot.

        :param judgement: The accepted amendment.
        :param known: The ontology's classes by name.
        :param accepted: The list to add it to when it can be written.
        """
        try:
            amendment = amend_class_source(
                known[judgement.whole], known[judgement.mixin]
            )
        except CannotAmendClass as refusal:
            # The ontology refusing an edit is an answer about this class, not a state
            # the run should not have reached.
            judgement.blocked = str(refusal)
            self.logger.warning("      ! %s", refusal)
            return
        if amendment is None:
            judgement.blocked = "the class already has that mixin"
            return
        judgement.edit = SourceEdit(
            file=str(amendment.path),
            line=amendment.line_number,
            before=amendment.before,
            after=amendment.after,
        )
        accepted.append((judgement, amendment))

    def apply(self, amendments: List[SourceAmendment]) -> None:
        """
        Write the accepted amendments and rebuild the ORM, undoing them if either fails.

        :param amendments: The amendments to carry out.
        :raises SubprocessStepFailedError: If the ORM cannot be rebuilt, after the
            amendments have been put back.
        """
        written: List[SourceAmendment] = []
        amended = False
        try:
            for amendment in amendments:
                amendment.apply()
                written.append(amendment)
                self.logger.info(
                    "  %s:%s  %s",
                    amendment.path.name,
                    amendment.line_number,
                    amendment.after,
                )
            self.verify(written)
            self.logger.info("regenerating the ORM ...")
            OrmRebuild(directory=self.run.directory).run_in_new_interpreter()
            amended = True
        finally:
            if not amended:
                for amendment in reversed(written):
                    amendment.reverted().apply()
                self.logger.error("the amendments were undone; nothing was changed.")
        self.logger.info("the ORM was regenerated; the ontology is amended.")
        self.logger.info("it is put back when the run ends.")

    def verify(self, applied: List[SourceAmendment]) -> None:
        """
        Check that the amended classes really hold what they were amended to hold.

        Asked in a new interpreter, since the classes in this one were built before the
        edit and a dataclass collects its fields once.

        :param applied: The amendments that were written.
        :raises SubprocessStepFailedError: If one of them did not take effect.
        """
        check = AmendedClassesCheck(
            pairs=[
                AmendedPair(
                    whole=amendment.annotation_class.__name__,
                    mixin=amendment.mixin.__name__,
                )
                for amendment in applied
            ]
        )
        self.in_new_interpreter(
            AmendedClassesCheck,
            what="checking the amended classes came back amended",
            arguments=check.arguments(),
        )


# %% putting the ontology back


@dataclass
class RevertAmendments(PipelineStep):
    """
    The ontology put back the way it was written.

    An amendment is applied so that a run reads the ontology the run decided on, not so
    that the ontology keeps it: the next scene should start from what was written by hand,
    where a mixin one room talked a model into is not quietly in force.
    """

    @property
    def name(self) -> str:
        return "put the ontology back as it was written"

    @property
    def is_optional(self) -> bool:
        return True

    def carry_out(self) -> None:
        """
        Undo every applied edit in reverse, and rebuild the ORM from the restored
        classes.
        """
        records = [
            AmendmentRecord.from_json(one)
            for one in self.run.read_json_if_written(RunFile.TAXONOMY_AMENDMENTS) or []
        ]
        applied = [one for one in records if one.applied]
        if not applied:
            self.logger.info("nothing is applied, so there is nothing to put back")
            return

        known = self.ontology_classes()
        self.logger.info("putting back %s amendment(s):", len(applied))
        for record in reversed(applied):
            # before and after swapped, so applying this checks the file really holds the
            # amended line before it undoes it.
            SourceAmendment(
                annotation_class=known[record.whole],
                mixin=known[record.mixin],
                path=Path(record.edit.file),
                line_number=record.edit.line,
                before=record.edit.after,
                after=record.edit.before,
            ).apply()
            record.applied = False
            record.reverted = True
            self.logger.info(
                "  %s:%s  %s",
                Path(record.edit.file).name,
                record.edit.line,
                record.edit.before,
            )

        self.logger.info("regenerating the ORM ...")
        OrmRebuild(directory=self.run.directory).run_in_new_interpreter()
        self.run.write_json(
            RunFile.TAXONOMY_AMENDMENTS, [one.to_json() for one in records]
        )
        self.logger.info("the ontology is back as it was written.")
