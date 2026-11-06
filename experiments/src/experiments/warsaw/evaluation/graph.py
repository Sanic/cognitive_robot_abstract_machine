"""
A database-independent snapshot of the semantic graph produced by a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.size import ObjectSize
from experiments.warsaw.pipeline.records import Classifications, SplitRecord

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% graph nodes and edges


@dataclass(frozen=True)
class EvaluationNode(JsonRecord):
    """
    One reconstructed body and the semantic class assigned to it.
    """

    name: str
    """
    The stable body name used by all pipeline artefacts.
    """

    input_label: str
    """
    The rough semantic label supplied with the reconstruction.
    """

    predicted_class: str | None
    """
    The SDT class selected by the pipeline, if any.
    """

    faces: int
    """
    How many mesh faces survived the split for this body.
    """

    body_id: str | None
    """
    The database identity of the underlying body, when persisted.
    """

    annotation_applied: bool
    """
    Whether the selected class was instantiated in the final SDT.
    """

    parent: str | None
    """
    The nearest body of the split this body hangs below in the final world, or nothing
    where it hangs below none.
    """

    size: ObjectSize | None = None
    """
    How big the body is, where a run's geometry has been measured.
    """

    @property
    def classes(self) -> list[str]:
        """
        :return: The classes this body stands as in a comparison, which is the one it was
            classified as, or none where the run never answered for it.
        """
        if self.predicted_class is None:
            return []
        return [self.predicted_class]


# %% one run's portable graph


@dataclass(frozen=True)
class EvaluationGraph(JsonRecord):
    """
    All reconstructed objects, their classes, and where each hangs in the world.
    """

    nodes: list[EvaluationNode] = field(default_factory=list)
    """
    Every body made by the split, including bodies left unannotated.
    """

    source_world_id: int | None = None
    """
    The database ID of the flat split world.
    """

    annotated_world_id: int | None = None
    """
    The database ID of the final annotated world.
    """

    @classmethod
    def from_run_products(
        cls,
        *,
        split: SplitRecord,
        classifications: Classifications,
        annotated_names: set[str],
        parents: dict[str, str | None],
    ) -> EvaluationGraph:
        """
        Build a graph from files and outcomes already produced by the pipeline.

        :param split: The bodies the scene was cut into, and the worlds it was written
            as.
        :param classifications: The class selected for every body the model answered.
        :param annotated_names: Bodies whose annotations were instantiated successfully.
        :param parents: The body each body of the split hangs below in the final world,
            as :meth:`parents_in` finds it.
        :return: A portable graph that does not depend on the run's database schema.
        """
        answers = {
            answer.name: answer
            for answer in classifications.bodies
            if answer.name is not None
        }
        nodes = [
            EvaluationNode(
                name=body.name,
                input_label=body.label,
                predicted_class=(
                    answers[body.name].class_name if body.name in answers else None
                ),
                faces=body.faces,
                body_id=body.body_id,
                annotation_applied=body.name in annotated_names,
                parent=parents[body.name],
            )
            for body in split.bodies
        ]
        return cls(
            nodes=nodes,
            source_world_id=split.world_id,
            annotated_world_id=split.annotated_world_id,
        )

    @staticmethod
    def parents_in(world: World, names: set[str]) -> dict[str, str | None]:
        """
        :param world: The world a run built.
        :param names: The bodies of its split.
        :return: For each of those bodies, the nearest of them it hangs below, or nothing
            where it hangs below none.
        """
        parents = {}
        for body in world.bodies:
            name = str(body.name.name)
            if name not in names:
                continue
            above = body.parent_kinematic_structure_entity
            while above is not None and str(above.name.name) not in names:
                above = above.parent_kinematic_structure_entity
            parents[name] = None if above is None else str(above.name.name)
        return parents
