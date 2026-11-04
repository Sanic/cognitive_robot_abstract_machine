"""
Judge finished runs of Articulate3D scenes against the scenes' ground truth.

A run cuts the scan into bodies, says what each body is, and mounts parts into wholes.
Every judgement here rests on which body is which annotated part, and that is decided by
the faces they share: the converted scene keeps the scan's faces in the annotation's
order, so the same face index means the same triangle on both sides. Nothing is matched by
class or by position, so a misclassified body is still found, and counted as a
misclassification rather than as an object nobody found.

What is judged, each over the whole ground truth so that nothing a run missed drops out:

- **objects**: one part to at most one body, with the parts split into several bodies and
  the bodies covering several parts counted beside;
- **classes**: whether each found part was given its class, exactly and as a share of the
  class hierarchy, with a class the run generated read as the class it derives from;
- **links**: whether each part hangs from its whole in the tree the run built, how far
  up the hierarchy it is placed right where it is not, and whether link and classes are
  right together.

A share with nothing to divide by is reported as nothing, never as a perfect score.

Run ``python -m experiments.warsaw.articulate3d.score --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from typing_extensions import Any, Sequence

from experiments.warsaw.articulate3d.convert import ConvertedScene
from experiments.warsaw.articulate3d.exceptions import ClassNotInOntologyError
from experiments.warsaw.articulate3d.ground_truth import (
    GroundTruthFile,
    GroundTruthPart,
    GroundTruthRelation,
    SceneGroundTruth,
)
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.run_classes import GeneratedClasses

# %% a share of something


@dataclass(frozen=True)
class Share:
    """
    How many of some things have a property, out of how many there are.
    """

    count: int
    """
    How many have the property.
    """

    total: int
    """
    How many there are.
    """

    @property
    def value(self) -> float | None:
        """
        :return: The share, or nothing where there is nothing to take a share of.
        """
        return self.count / self.total if self.total else None

    def harmonic_mean_with(self, other: Share) -> float | None:
        """
        :param other: Another share, such as the recall to this precision.
        :return: The harmonic mean of the two, or nothing where either is undefined.
        """
        if self.value is None or other.value is None:
            return None
        if self.value + other.value == 0.0:
            return 0.0
        return 2 * self.value * other.value / (self.value + other.value)


# %% the ontology a run's classes are judged in


class TaxonomyKey(StrEnum):
    """
    The keys of the taxonomy a run was given that judging classes reads.
    """

    ROOT_NAME = "root_name"
    """
    The class every class derives from.
    """

    CLASSES = "classes"
    """
    Every class of the ontology.
    """

    NAME = "name"
    """
    A class's name.
    """

    BASES = "bases"
    """
    The classes a class directly derives from.
    """

    MIXIN = "mixin"
    """
    Whether a class is a mixin, which says what a class can hold rather than what it is.
    """


@dataclass(frozen=True)
class ClassLineage:
    """
    What each class a run could answer with derives from, mixins and the root left out.
    """

    bases: dict[str, list[str]]
    """
    The classes each class of the ontology derives from, by class name.
    """

    generated: dict[str, list[str]] = field(default_factory=dict)
    """
    The classes of the ontology each class the run generated derives from, by class name.
    """

    @classmethod
    def from_taxonomy(
        cls, taxonomy: dict[str, Any], generated: dict[str, list[str]]
    ) -> ClassLineage:
        """
        :param taxonomy: The ontology as the run was given it.
        :param generated: The bases each class the run generated declares.
        :return: What each of those classes derives from.
        """
        nodes = taxonomy[TaxonomyKey.CLASSES]
        left_out = {
            node[TaxonomyKey.NAME] for node in nodes if node.get(TaxonomyKey.MIXIN)
        }
        left_out.add(taxonomy[TaxonomyKey.ROOT_NAME])
        bases = {
            node[TaxonomyKey.NAME]: [
                base for base in node[TaxonomyKey.BASES] if base not in left_out
            ]
            for node in nodes
            if node[TaxonomyKey.NAME] not in left_out
        }
        return cls(
            bases=bases,
            generated={
                name: [base for base in declared if base in bases]
                for name, declared in generated.items()
                if name not in bases
            },
        )

    @classmethod
    def of_run(cls, run: Run) -> ClassLineage:
        """
        :param run: A finished run.
        :return: What the classes it was given and the classes it generated derive from.
        """
        return cls.from_taxonomy(
            run.read_json(RunFile.TAXONOMY),
            GeneratedClasses(directory=run.directory).declared_bases,
        )

    def projected(self, class_name: str | None) -> str | None:
        """
        :param class_name: A class a run answered with.
        :return: The class of the ontology it stands for: itself where the ontology holds
            it, the one class of the ontology a generated class derives from, and nothing
            where there is not exactly one.
        """
        if class_name in self.bases:
            return class_name
        derived_from = self.generated.get(class_name, [])
        return derived_from[0] if len(derived_from) == 1 else None

    def ancestry(self, class_name: str | None) -> frozenset[str]:
        """
        :param class_name: A class a run answered with, or a class of the ontology.
        :return: The classes of the ontology it is, itself included where the ontology
            holds it, and nothing for a class it does not know.
        """
        if class_name in self.generated:
            return frozenset().union(
                *(self.ancestry(base) for base in self.generated[class_name])
            )
        if class_name not in self.bases:
            return frozenset()
        found = {class_name}
        waiting = [class_name]
        while waiting:
            for base in self.bases.get(waiting.pop(), []):
                if base not in found:
                    found.add(base)
                    waiting.append(base)
        return frozenset(found)


# %% which body is which part


@dataclass(frozen=True)
class PartMatch(JsonRecord):
    """
    One annotated part, and the body a run made of it.
    """

    part_id: int
    """
    The part.
    """

    body: str
    """
    The body.
    """

    overlap: float
    """
    The faces they share over the faces either covers.
    """


@dataclass(frozen=True)
class FaceCorrespondence(JsonRecord):
    """
    Which body of a run is which annotated part, decided by the faces they share.
    """

    minimum_overlap: float
    """
    How much of the faces either covers a part and a body must share to be paired.
    """

    matched: list[PartMatch] = field(default_factory=list)
    """
    The pairs, each part and each body in at most one.
    """

    unmatched_parts: list[int] = field(default_factory=list)
    """
    The parts no body was paired with.
    """

    unmatched_bodies: list[str] = field(default_factory=list)
    """
    The bodies on annotated faces that no part was paired with.
    """

    outside_bodies: list[str] = field(default_factory=list)
    """
    The bodies lying mostly on faces the ground truth does not cover, which are not
    judged, since the annotation says nothing about what is there.
    """

    fragmented_parts: list[int] = field(default_factory=list)
    """
    The parts at least two bodies lie mostly inside of.
    """

    merged_bodies: list[str] = field(default_factory=list)
    """
    The bodies covering most of at least two parts.
    """

    @classmethod
    def between(
        cls,
        parts: Sequence[GroundTruthPart],
        bodies: dict[str, np.ndarray],
        minimum_overlap: float = 0.5,
        mostly: float = 0.5,
    ) -> FaceCorrespondence:
        """
        Pair parts and bodies so that the faces the pairs share add up to the most.

        :param parts: The annotated parts.
        :param bodies: The faces of each body, by the body's name.
        :param minimum_overlap: How much of the faces either covers a pair must share.
        :param mostly: What share of a body or of a part counts as most of it.
        :return: The pairs, and what was left over.
        """
        names = sorted(bodies)
        size = 1 + max(
            [max(part.faces) for part in parts if part.faces]
            + [int(faces.max()) for faces in bodies.values() if len(faces)]
            + [-1]
        )
        part_faces = cls._incidence([part.faces for part in parts], size)
        body_faces = cls._incidence([bodies[name] for name in names], size)
        shared = (body_faces @ part_faces.T).toarray().astype(float)
        body_sizes = np.asarray(body_faces.sum(axis=1), dtype=float).ravel()
        part_sizes = np.asarray(part_faces.sum(axis=1), dtype=float).ravel()
        union = body_sizes[:, None] + part_sizes[None, :] - shared
        overlap = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
        of_body = np.divide(
            shared,
            body_sizes[:, None],
            out=np.zeros_like(shared),
            where=body_sizes[:, None] > 0,
        )
        of_part = np.divide(
            shared,
            part_sizes[None, :],
            out=np.zeros_like(shared),
            where=part_sizes[None, :] > 0,
        )

        rows, columns = linear_sum_assignment(overlap, maximize=True)
        pairs = [
            (row, column)
            for row, column in zip(rows, columns)
            if overlap[row, column] >= minimum_overlap
        ]
        paired_bodies = {row for row, _ in pairs}
        paired_parts = {column for _, column in pairs}
        annotated = np.asarray(part_faces.sum(axis=0)).ravel() > 0
        inside = np.divide(
            body_faces @ annotated.astype(float),
            body_sizes,
            out=np.zeros_like(body_sizes),
            where=body_sizes > 0,
        )
        outside = {
            row
            for row in range(len(names))
            if row not in paired_bodies and inside[row] < mostly
        }
        return cls(
            minimum_overlap=minimum_overlap,
            matched=[
                PartMatch(
                    part_id=parts[column].part_id,
                    body=names[row],
                    overlap=float(overlap[row, column]),
                )
                for row, column in pairs
            ],
            unmatched_parts=[
                part.part_id
                for column, part in enumerate(parts)
                if column not in paired_parts
            ],
            unmatched_bodies=[
                name
                for row, name in enumerate(names)
                if row not in paired_bodies and row not in outside
            ],
            outside_bodies=[names[row] for row in sorted(outside)],
            fragmented_parts=[
                part.part_id
                for column, part in enumerate(parts)
                if (of_body[:, column] >= mostly).sum() >= 2
            ],
            merged_bodies=[
                name
                for row, name in enumerate(names)
                if (of_part[row, :] >= mostly).sum() >= 2
            ],
        )

    @staticmethod
    def _incidence(face_sets: Sequence[Sequence[int]], size: int) -> csr_matrix:
        """
        :param face_sets: The faces of each of some objects.
        :param size: How many faces there are to cover.
        :return: Per object, a row holding one in the column of every face it covers.
        """
        rows = np.concatenate(
            [np.full(len(faces), row) for row, faces in enumerate(face_sets)]
            + [np.zeros(0, dtype=int)]
        )
        columns = np.concatenate(
            [np.asarray(faces, dtype=int) for faces in face_sets]
            + [np.zeros(0, dtype=int)]
        )
        incidence = csr_matrix(
            (np.ones(len(rows)), (rows, columns)), shape=(len(face_sets), size)
        )
        incidence.sum_duplicates()
        incidence.data[:] = 1.0
        return incidence

    @property
    def body_of_part(self) -> dict[int, str]:
        """
        :return: The body paired with each part that was paired.
        """
        return {match.part_id: match.body for match in self.matched}

    @property
    def part_of_body(self) -> dict[str, int]:
        """
        :return: The part paired with each body that was paired.
        """
        return {match.body: match.part_id for match in self.matched}


# %% what each part was given


@dataclass(frozen=True)
class JudgedClass(JsonRecord):
    """
    The class a run gave one found part whose class is judged.
    """

    part_id: int
    """
    The part.
    """

    part_name: str
    """
    The name its annotator gave it.
    """

    true_class: str
    """
    The class the ground truth says it is.
    """

    predicted_class: str | None
    """
    The class the run gave its body, or nothing where it gave none.
    """

    projected_class: str | None
    """
    The class of the ontology the run's answer stands for.
    """

    shared_ancestry: int
    """
    How many classes of the ontology both the answer and the truth are.
    """

    predicted_ancestry: int
    """
    How many classes of the ontology the answer is.
    """

    true_ancestry: int
    """
    How many classes of the ontology the truth is.
    """

    @property
    def correct(self) -> bool:
        """
        :return: Whether the answer stands for exactly the true class.
        """
        return self.projected_class == self.true_class


@dataclass(frozen=True)
class ClassConfusion:
    """
    How often parts of one annotated name and one true class were answered one way.
    """

    part_name: str
    """
    The name the parts' annotators gave them.
    """

    true_class: str
    """
    The class they are.
    """

    projected_class: str | None
    """
    The class of the ontology the run's answer stands for.
    """

    count: int
    """
    How many parts were answered so.
    """


@dataclass(frozen=True)
class ClassAgreement(JsonRecord):
    """
    How the classes a run gave agree with the ground truth.
    """

    judged: list[JudgedClass] = field(default_factory=list)
    """
    Every found part whose class is judged.
    """

    classified_parts: int = 0
    """
    How many parts' classes are judged, found or not.
    """

    @classmethod
    def between(
        cls,
        parts: Sequence[GroundTruthPart],
        correspondence: FaceCorrespondence,
        predicted: dict[str, str | None],
        lineage: ClassLineage,
    ) -> ClassAgreement:
        """
        :param parts: The annotated parts.
        :param correspondence: Which body is which part.
        :param predicted: The class the run gave each body.
        :param lineage: What each class derives from.
        :return: Every found part's class, judged.
        :raises ClassNotInOntologyError: If the ground truth names a class the run's
            ontology does not hold.
        """
        classified = [part for part in parts if part.semantic_class is not None]
        for part in classified:
            if part.semantic_class not in lineage.bases:
                raise ClassNotInOntologyError(class_name=part.semantic_class)
        body_of = correspondence.body_of_part
        judged = []
        for part in classified:
            if part.part_id not in body_of:
                continue
            answer = predicted.get(body_of[part.part_id])
            answer_ancestry = lineage.ancestry(answer)
            true_ancestry = lineage.ancestry(part.semantic_class)
            judged.append(
                JudgedClass(
                    part_id=part.part_id,
                    part_name=part.part_name,
                    true_class=part.semantic_class,
                    predicted_class=answer,
                    projected_class=lineage.projected(answer),
                    shared_ancestry=len(answer_ancestry & true_ancestry),
                    predicted_ancestry=len(answer_ancestry),
                    true_ancestry=len(true_ancestry),
                )
            )
        return cls(judged=judged, classified_parts=len(classified))

    def combined_with(self, other: ClassAgreement) -> ClassAgreement:
        """
        :param other: The agreement of other scenes or runs.
        :return: Both, as one.
        """
        return ClassAgreement(
            judged=self.judged + other.judged,
            classified_parts=self.classified_parts + other.classified_parts,
        )

    @property
    def accuracy(self) -> Share:
        """
        :return: The found parts given exactly their class.
        """
        return Share(sum(one.correct for one in self.judged), len(self.judged))

    @property
    def recognised(self) -> Share:
        """
        :return: The parts found and given exactly their class, out of every part whose
            class is judged.
        """
        return Share(sum(one.correct for one in self.judged), self.classified_parts)

    @property
    def unprojectable(self) -> int:
        """
        :return: How many answers stand for no single class of the ontology.
        """
        return sum(
            1
            for one in self.judged
            if one.predicted_class is not None and one.projected_class is None
        )

    @property
    def hierarchical_precision(self) -> Share:
        """
        :return: Of the classes the answers are, those the truths are too.
        """
        return Share(
            sum(one.shared_ancestry for one in self.judged),
            sum(one.predicted_ancestry for one in self.judged),
        )

    @property
    def hierarchical_recall(self) -> Share:
        """
        :return: Of the classes the truths are, those the answers are too.
        """
        return Share(
            sum(one.shared_ancestry for one in self.judged),
            sum(one.true_ancestry for one in self.judged),
        )

    def confusions(self) -> list[ClassConfusion]:
        """
        :return: How each annotated name and true class was answered, commonest first.
        """
        counted = Counter(
            (one.part_name, one.true_class, one.projected_class) for one in self.judged
        )
        return [
            ClassConfusion(
                part_name=part_name,
                true_class=true_class,
                projected_class=projected_class,
                count=count,
            )
            for (part_name, true_class, projected_class), count in sorted(
                counted.items(), key=lambda one: (-one[1], str(one[0]))
            )
        ]


# %% where each part hangs


class ClassVerdict(StrEnum):
    """
    What the ground truth makes of the class a run gave one annotated part.
    """

    RIGHT = "given exactly its class"
    """
    The part was found and given exactly the class it is.
    """

    WRONG = "given another class, or none"
    """
    The part's class is judged, and the run did not give it that class, found or not.
    """

    NOT_JUDGED = "its class is not judged"
    """
    The protocol judges no class for the part.
    """


@dataclass(frozen=True)
class JudgedAttachment(JsonRecord):
    """
    Where one annotated part hangs in the part hierarchy, and where the run hung it.

    A run hangs a part from the nearest body above it that is an annotated part, past
    any body between that is none.
    """

    part_id: int
    """
    The part.
    """

    relation: GroundTruthRelation | None
    """
    The relation of the part hierarchy holding the part, or nothing where no annotated
    part holds it.
    """

    predicted_whole: int | None
    """
    The annotated part the run hung it from, or nothing where it was not found or hangs
    from no annotated part.
    """

    true_ancestors: list[int]
    """
    Every annotated part it lies below in the part hierarchy, nearest first.
    """

    predicted_ancestors: list[int]
    """
    Every annotated part the run hung it below, nearest first.
    """

    part_class: ClassVerdict
    """
    What the ground truth makes of the class the run gave the part.
    """

    true_whole_class: ClassVerdict | None
    """
    What it makes of the class the run gave the part's whole, where it has one.
    """

    predicted_whole_class: ClassVerdict | None
    """
    What it makes of the class the run gave the part it hung this one from, where it
    hung it from one.
    """

    @property
    def correct(self) -> bool:
        """
        :return: Whether the run hung the part from exactly its whole.
        """
        return self.relation is not None and self.predicted_whole == self.relation.whole

    @property
    def shared_ancestors(self) -> int:
        """
        :return: How many parts the run hung it below that it does lie below.
        """
        return len(set(self.true_ancestors) & set(self.predicted_ancestors))

    @property
    def classes_right(self) -> bool:
        """
        :return: Whether the run hung the part from exactly its whole and gave both
            exactly their class.
        """
        return (
            self.correct
            and self.part_class is ClassVerdict.RIGHT
            and self.true_whole_class is ClassVerdict.RIGHT
        )


@dataclass(frozen=True)
class LinkAgreement(JsonRecord):
    """
    How the tree a run built agrees with the part hierarchy, link by link.
    """

    attachments: list[JudgedAttachment] = field(default_factory=list)
    """
    Every annotated part, and where the run hung it.
    """

    @classmethod
    def between(
        cls,
        ground_truth: SceneGroundTruth,
        correspondence: FaceCorrespondence,
        parents: dict[str, str | None],
        classes: ClassAgreement,
    ) -> LinkAgreement:
        """
        :param ground_truth: The scene's ground truth.
        :param correspondence: Which body is which part.
        :param parents: The body each body of the run hangs below.
        :param classes: How the classes the run gave agree with the ground truth.
        :return: Every annotated part's place, judged.
        """
        part_of, body_of = correspondence.part_of_body, correspondence.body_of_part
        holding = {relation.part: relation for relation in ground_truth.relations}
        verdicts = {
            part.part_id: ClassVerdict.NOT_JUDGED
            if part.semantic_class is None
            else ClassVerdict.WRONG
            for part in ground_truth.parts
        }
        for one in classes.judged:
            if one.correct:
                verdicts[one.part_id] = ClassVerdict.RIGHT
        attachments = []
        for part in ground_truth.parts:
            true_ancestors = []
            holder = holding.get(part.part_id)
            while holder is not None:
                true_ancestors.append(holder.whole)
                holder = holding.get(holder.whole)
            predicted_ancestors = []
            above = parents[body_of[part.part_id]] if part.part_id in body_of else None
            while above is not None:
                if above in part_of:
                    predicted_ancestors.append(part_of[above])
                above = parents[above]
            relation = holding.get(part.part_id)
            predicted_whole = predicted_ancestors[0] if predicted_ancestors else None
            attachments.append(
                JudgedAttachment(
                    part_id=part.part_id,
                    relation=relation,
                    predicted_whole=predicted_whole,
                    true_ancestors=true_ancestors,
                    predicted_ancestors=predicted_ancestors,
                    part_class=verdicts[part.part_id],
                    true_whole_class=(
                        None if relation is None else verdicts[relation.whole]
                    ),
                    predicted_whole_class=(
                        None if predicted_whole is None else verdicts[predicted_whole]
                    ),
                )
            )
        return cls(attachments=attachments)

    def combined_with(self, other: LinkAgreement) -> LinkAgreement:
        """
        :param other: The agreement of other scenes or runs.
        :return: Both, as one.
        """
        return LinkAgreement(attachments=self.attachments + other.attachments)

    @property
    def precision(self) -> Share:
        """
        :return: Of the parts the run hung from another, those it hung from their whole.
        """
        return Share(
            sum(one.correct for one in self.attachments),
            sum(one.predicted_whole is not None for one in self.attachments),
        )

    @property
    def recall(self) -> Share:
        """
        :return: Of the relations of the part hierarchy, those the run made.
        """
        return Share(
            sum(one.correct for one in self.attachments),
            sum(one.relation is not None for one in self.attachments),
        )

    @property
    def reachable(self) -> Share:
        """
        :return: Of the relations of the part hierarchy, those the ontology lets a run
            make, which is the most of them any run can make.
        """
        return Share(
            len(self._reachable),
            sum(one.relation is not None for one in self.attachments),
        )

    @property
    def reachable_recall(self) -> Share:
        """
        :return: Of the relations the ontology lets a run make, those the run made.
        """
        return Share(sum(one.correct for one in self._reachable), len(self._reachable))

    @property
    def ancestor_precision(self) -> Share:
        """
        :return: Of the parts the run hung each part below, those it does lie below.
        """
        return Share(
            sum(one.shared_ancestors for one in self.attachments),
            sum(len(one.predicted_ancestors) for one in self.attachments),
        )

    @property
    def ancestor_recall(self) -> Share:
        """
        :return: Of the parts each part lies below, those the run hung it below.
        """
        return Share(
            sum(one.shared_ancestors for one in self.attachments),
            sum(len(one.true_ancestors) for one in self.attachments),
        )

    @property
    def triplet_precision(self) -> Share:
        """
        :return: Of the parts the run hung from another, both classes judged, those hung
            from their whole with both given exactly their class.
        """
        judged = [
            one
            for one in self.attachments
            if one.part_class is not ClassVerdict.NOT_JUDGED
            and one.predicted_whole_class not in (None, ClassVerdict.NOT_JUDGED)
        ]
        return Share(sum(one.classes_right for one in judged), len(judged))

    @property
    def triplet_recall(self) -> Share:
        """
        :return: Of the relations of the part hierarchy, both classes judged, those the
            run made with both ends given exactly their class.
        """
        judged = [
            one
            for one in self.attachments
            if one.part_class is not ClassVerdict.NOT_JUDGED
            and one.true_whole_class not in (None, ClassVerdict.NOT_JUDGED)
        ]
        return Share(sum(one.classes_right for one in judged), len(judged))

    @property
    def _reachable(self) -> list[JudgedAttachment]:
        """
        :return: The parts held by a relation the ontology lets a run make.
        """
        return [
            one
            for one in self.attachments
            if one.relation is not None and one.relation.expressibility.reachable
        ]


# %% one run, judged


class ScoreFile(StrEnum):
    """
    Where a run's score is written.
    """

    SCENE = "articulate3d_score.json"
    """
    The score of one run, in the run's own directory.
    """


@dataclass(frozen=True)
class SceneScore(JsonRecord):
    """
    One run of one scene, judged against the scene's ground truth.
    """

    run: str
    """
    The run judged.
    """

    scene_id: str
    """
    The scene it ran on.
    """

    correspondence: FaceCorrespondence
    """
    Which body is which part.
    """

    classes: ClassAgreement
    """
    How the classes it gave agree with the ground truth.
    """

    links: LinkAgreement
    """
    How the tree it built agrees with the part hierarchy.
    """

    @classmethod
    def between(
        cls,
        run_name: str,
        ground_truth: SceneGroundTruth,
        bodies: dict[str, np.ndarray],
        graph: EvaluationGraph,
        lineage: ClassLineage,
        minimum_overlap: float = 0.5,
    ) -> SceneScore:
        """
        :param run_name: The run judged.
        :param ground_truth: The scene's ground truth.
        :param bodies: The faces of each body the run made, by name.
        :param graph: The classes the run decided on, and where each body hangs.
        :param lineage: What each class the run could answer with derives from.
        :param minimum_overlap: How much of their faces a part and a body must share to
            be paired.
        :return: The run, judged.
        """
        correspondence = FaceCorrespondence.between(
            ground_truth.parts, bodies, minimum_overlap
        )
        classes = ClassAgreement.between(
            ground_truth.parts,
            correspondence,
            {node.name: node.predicted_class for node in graph.nodes},
            lineage,
        )
        return cls(
            run=run_name,
            scene_id=ground_truth.scene_id,
            correspondence=correspondence,
            classes=classes,
            links=LinkAgreement.between(
                ground_truth,
                correspondence,
                {node.name: node.parent for node in graph.nodes},
                classes,
            ),
        )

    @classmethod
    def of(
        cls, run: Run, ground_truth_directory: Path, minimum_overlap: float = 0.5
    ) -> SceneScore:
        """
        :param run: A finished run of a converted scene.
        :param ground_truth_directory: The directory holding every scene's ground truth.
        :param minimum_overlap: How much of their faces a part and a body must share to
            be paired.
        :return: The run, judged.
        """
        scene_id = ConvertedScene.beside(run.scene_directory).scene_id
        faces = np.load(run.path(RunFile.SPLIT_FACES))
        return cls.between(
            run_name=run.directory.name,
            ground_truth=SceneGroundTruth.read(
                GroundTruthFile.SUFFIX.path(ground_truth_directory, scene_id)
            ),
            bodies={name: faces[name] for name in faces.files},
            graph=run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph),
            lineage=ClassLineage.of_run(run),
            minimum_overlap=minimum_overlap,
        )


# %% many runs, judged together


@dataclass(frozen=True)
class ScoreOverRuns(JsonRecord):
    """
    Many runs judged together, every count added up over them.
    """

    runs: list[str] = field(default_factory=list)
    """
    The runs added up.
    """

    parts: int = 0
    """
    How many annotated parts there were.
    """

    matched: int = 0
    """
    How many of them a body was paired with.
    """

    unmatched_bodies: int = 0
    """
    How many bodies on annotated faces were paired with no part.
    """

    outside_bodies: int = 0
    """
    How many bodies lay outside the annotation and were not judged.
    """

    fragmented_parts: int = 0
    """
    How many parts were split into several bodies.
    """

    merged_bodies: int = 0
    """
    How many bodies covered several parts.
    """

    classes: ClassAgreement = field(default_factory=ClassAgreement)
    """
    How the classes agree, over every run.
    """

    links: LinkAgreement = field(default_factory=LinkAgreement)
    """
    How the trees agree with the part hierarchy, over every run.
    """

    @classmethod
    def over(cls, scores: Sequence[SceneScore]) -> ScoreOverRuns:
        """
        :param scores: The runs to add up.
        :return: Their counts, added up.
        """
        summary = cls()
        for score in scores:
            matching = score.correspondence
            summary = ScoreOverRuns(
                runs=summary.runs + [score.run],
                parts=summary.parts
                + len(matching.matched)
                + len(matching.unmatched_parts),
                matched=summary.matched + len(matching.matched),
                unmatched_bodies=summary.unmatched_bodies
                + len(matching.unmatched_bodies),
                outside_bodies=summary.outside_bodies + len(matching.outside_bodies),
                fragmented_parts=summary.fragmented_parts
                + len(matching.fragmented_parts),
                merged_bodies=summary.merged_bodies + len(matching.merged_bodies),
                classes=summary.classes.combined_with(score.classes),
                links=summary.links.combined_with(score.links),
            )
        return summary

    @property
    def found(self) -> Share:
        """
        :return: The parts a body was paired with.
        """
        return Share(self.matched, self.parts)

    @property
    def matched_bodies(self) -> Share:
        """
        :return: Of the bodies judged, those paired with a part.
        """
        return Share(self.matched, self.matched + self.unmatched_bodies)

    def lines(self) -> list[str]:
        """
        :return: The headline numbers, one per line.
        """
        classes, links = self.classes, self.links
        link_f1 = links.precision.harmonic_mean_with(links.recall)
        triplet_f1 = links.triplet_precision.harmonic_mean_with(links.triplet_recall)
        return [
            f"{len(self.runs)} runs, {self.parts} annotated parts",
            f"parts found          {self._shown(self.found)}",
            f"bodies that are parts {self._shown(self.matched_bodies)}",
            f"fragmented parts {self.fragmented_parts}, merged bodies "
            f"{self.merged_bodies}, bodies outside {self.outside_bodies}",
            f"class accuracy       {self._shown(classes.accuracy)}",
            f"classes recognised   {self._shown(classes.recognised)}",
            f"hierarchical P / R   {self._shown(classes.hierarchical_precision)} / "
            f"{self._shown(classes.hierarchical_recall)}",
            f"answers standing for no class {classes.unprojectable}",
            f"link P / R           {self._shown(links.precision)} / "
            f"{self._shown(links.recall)}, F1 {self._number(link_f1)}",
            f"links reachable      {self._shown(links.reachable)}",
            f"reachable recall     {self._shown(links.reachable_recall)}",
            f"ancestor P / R       {self._shown(links.ancestor_precision)} / "
            f"{self._shown(links.ancestor_recall)}",
            f"triplet P / R        {self._shown(links.triplet_precision)} / "
            f"{self._shown(links.triplet_recall)}, F1 {self._number(triplet_f1)}",
        ]

    @staticmethod
    def _shown(share: Share) -> str:
        """
        :param share: A share.
        :return: It as a number and the counts it is taken from.
        """
        return f"{ScoreOverRuns._number(share.value)} ({share.count}/{share.total})"

    @staticmethod
    def _number(value: float | None) -> str:
        """
        :param value: A score, or nothing where it is undefined.
        :return: It as a number.
        """
        return "n/a" if value is None else f"{value:.3f}"


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for judging runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Finished runs' directories")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="The directory holding every scene's ground truth",
    )
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=0.5,
        help="How much of their faces a part and a body must share to be paired",
    )
    parser.add_argument(
        "--summary", type=Path, help="Where to write the runs' counts added up"
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Judge finished runs and write what each, and all of them together, came to.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every score is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    scores = []
    for directory in parsed.runs:
        run = Run(directory=directory)
        score = SceneScore.of(run, parsed.ground_truth, parsed.minimum_overlap)
        (run.directory / ScoreFile.SCENE).write_text(
            json.dumps(score.to_json(), indent=2)
        )
        scores.append(score)
    summary = ScoreOverRuns.over(scores)
    for line in summary.lines():
        logger.info(line)
    if parsed.summary is not None:
        parsed.summary.write_text(json.dumps(summary.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
