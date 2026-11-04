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
- **relations**: whether each mount the run made is a relation of the part hierarchy,
  and how many of those relations it made.

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
    Expressibility,
    GroundTruthFile,
    GroundTruthPart,
    GroundTruthRelation,
    SceneGroundTruth,
)
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.graph import EvaluationEdge, EvaluationGraph
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


# %% what each mount was


class MountVerdict(StrEnum):
    """
    What the ground truth makes of one mount a run made.
    """

    HELD = "the ground truth holds the part in that whole"
    """
    Both ends are found parts, and the part hierarchy holds the one directly in the other.
    """

    NOT_HELD = "the ground truth does not hold the part in that whole"
    """
    Both ends are found parts, and the part hierarchy does not hold the one directly in
    the other.
    """

    UNMATCHED_END = "an end is no annotated part"
    """
    An end lies on annotated faces but was paired with no part, so the mount cannot be a
    relation of the part hierarchy.
    """

    OUTSIDE = "an end lies outside the annotation"
    """
    An end lies on faces the ground truth does not cover, so the mount is not judged.
    """


@dataclass(frozen=True)
class JudgedMount(JsonRecord):
    """
    One part-whole mount a run made, and what the ground truth makes of it.
    """

    whole: str
    """
    The body the part was mounted into.
    """

    part: str
    """
    The body mounted.
    """

    field_name: str
    """
    The field the run mounted it through.
    """

    verdict: MountVerdict
    """
    What the ground truth makes of it.
    """

    truth: GroundTruthRelation | None = None
    """
    The relation of the part hierarchy it is, where it is one.
    """

    @property
    def same_field(self) -> bool:
        """
        :return: Whether it is a relation of the part hierarchy, mounted through the field
            the ontology holds that relation in.
        """
        return self.truth is not None and self.truth.field_name == self.field_name


@dataclass(frozen=True)
class RelationAgreement(JsonRecord):
    """
    How the part-whole mounts a run made agree with the part hierarchy.
    """

    mounts: list[JudgedMount] = field(default_factory=list)
    """
    Every part-whole mount the run made.
    """

    true_relations: list[GroundTruthRelation] = field(default_factory=list)
    """
    Every relation of the part hierarchy.
    """

    @classmethod
    def between(
        cls,
        relations: Sequence[GroundTruthRelation],
        correspondence: FaceCorrespondence,
        edges: Sequence[EvaluationEdge],
    ) -> RelationAgreement:
        """
        :param relations: The relations of the part hierarchy.
        :param correspondence: Which body is which part.
        :param edges: The relations the run tried to make.
        :return: Every part-whole mount the run made, judged.
        """
        part_of = correspondence.part_of_body
        outside = set(correspondence.outside_bodies)
        truth = {(relation.whole, relation.part): relation for relation in relations}
        mounts = []
        for edge in edges:
            if not edge.accepted:
                continue
            ends = (edge.whole, edge.part)
            if outside & set(ends):
                verdict, held = MountVerdict.OUTSIDE, None
            elif not all(end in part_of for end in ends):
                verdict, held = MountVerdict.UNMATCHED_END, None
            else:
                held = truth.get((part_of[edge.whole], part_of[edge.part]))
                verdict = MountVerdict.NOT_HELD if held is None else MountVerdict.HELD
            mounts.append(
                JudgedMount(
                    whole=edge.whole,
                    part=edge.part,
                    field_name=edge.field_name,
                    verdict=verdict,
                    truth=held,
                )
            )
        return cls(mounts=mounts, true_relations=list(relations))

    def combined_with(self, other: RelationAgreement) -> RelationAgreement:
        """
        :param other: The agreement of other scenes or runs.
        :return: Both, as one.
        """
        return RelationAgreement(
            mounts=self.mounts + other.mounts,
            true_relations=self.true_relations + other.true_relations,
        )

    @property
    def held(self) -> list[JudgedMount]:
        """
        :return: The mounts that are relations of the part hierarchy.
        """
        return [one for one in self.mounts if one.verdict is MountVerdict.HELD]

    @property
    def precision(self) -> Share:
        """
        :return: Of the mounts judged, those that are relations of the part hierarchy.
        """
        judged = [one for one in self.mounts if one.verdict is not MountVerdict.OUTSIDE]
        return Share(len(self.held), len(judged))

    @property
    def recall(self) -> Share:
        """
        :return: Of the relations of the part hierarchy, those the run made.
        """
        return Share(len(self.held), len(self.true_relations))

    @property
    def expressible_recall(self) -> Share:
        """
        :return: Of the relations the ontology can hold, those the run made.
        """
        return Share(
            sum(
                1
                for one in self.held
                if one.truth.expressibility is Expressibility.EXPRESSIBLE
            ),
            self._counted(Expressibility.EXPRESSIBLE),
        )

    @property
    def expressible(self) -> Share:
        """
        :return: Of the relations of the part hierarchy, those the ontology can hold,
            which is the most of them any run can make.
        """
        return Share(
            self._counted(Expressibility.EXPRESSIBLE), len(self.true_relations)
        )

    @property
    def same_field(self) -> Share:
        """
        :return: Of the mounts that are relations of the part hierarchy, those made
            through the field the ontology holds them in.
        """
        return Share(sum(one.same_field for one in self.held), len(self.held))

    def _counted(self, expressibility: Expressibility) -> int:
        """
        :param expressibility: What the ontology makes of a relation.
        :return: How many relations of the part hierarchy it makes that of.
        """
        return sum(
            1 for one in self.true_relations if one.expressibility is expressibility
        )


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

    relations: RelationAgreement
    """
    How the mounts it made agree with the part hierarchy.
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
        :param graph: The classes and mounts the run decided on.
        :param lineage: What each class the run could answer with derives from.
        :param minimum_overlap: How much of their faces a part and a body must share to
            be paired.
        :return: The run, judged.
        """
        correspondence = FaceCorrespondence.between(
            ground_truth.parts, bodies, minimum_overlap
        )
        return cls(
            run=run_name,
            scene_id=ground_truth.scene_id,
            correspondence=correspondence,
            classes=ClassAgreement.between(
                ground_truth.parts,
                correspondence,
                {node.name: node.predicted_class for node in graph.nodes},
                lineage,
            ),
            relations=RelationAgreement.between(
                ground_truth.relations, correspondence, graph.edges
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

    relations: RelationAgreement = field(default_factory=RelationAgreement)
    """
    How the mounts agree, over every run.
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
                relations=summary.relations.combined_with(score.relations),
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
        classes, relations = self.classes, self.relations
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
            f"relation precision   {self._shown(relations.precision)}",
            f"relation recall      {self._shown(relations.recall)}",
            f"expressible recall   {self._shown(relations.expressible_recall)}",
            f"expressible          {self._shown(relations.expressible)}",
            f"through the same field {self._shown(relations.same_field)}",
        ]

    @staticmethod
    def _shown(share: Share) -> str:
        """
        :param share: A share.
        :return: It as a number and the counts it is taken from.
        """
        shown = "n/a" if share.value is None else f"{share.value:.3f}"
        return f"{shown} ({share.count}/{share.total})"


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
