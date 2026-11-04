"""
A database-independent snapshot of a modelled world, for comparing a run against it.

A hand-modelled world holds far more than the mesh export of it does: which body a class
was put on, which parts were mounted into which whole, and through which field. Comparing
a run against exported labels alone throws that away, so the ground truth is read out of
the world itself and written in the same shape the run writes its own graph in.

The two graphs agree on how a relation is named -- what it means and the field realizing
it -- because both read it from the ontology rather than spelling it out.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.world_mesh_exporter import (
    GeometrySource,
    WorldMeshExtractor,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    mounted_relations_of,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    WorldEntity,
)
from typing_extensions import List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.size import ObjectSize
from experiments.warsaw.exceptions import (
    CorrectedEntityNotInGraphError,
    GroundTruthAlreadyCorrectedError,
    RelationHasNoEntityError,
    WorldProviderNotFoundError,
)

# %% ground truth a person supplied


@dataclass(frozen=True)
class ClassCorrection(JsonRecord):
    """
    The class a person gave an entity the modelled world does not name correctly.

    A world only carries what was modelled into it, and its classes may be inferred by
    rules that need evidence the model happens to lack. What a person knows about the
    scene reaches the comparison through this, and is recorded beside what the world
    said rather than in place of it, so an audit of the rules can still tell the two
    apart.
    """

    entity: str
    """
    The entity being given a class.
    """

    semantic_class: str
    """
    The class it stands as in the comparison.
    """

    reason: str
    """
    Why the world does not name it, so the decision can be reviewed rather than trusted.
    """


# %% one modelled world's nodes and edges


@dataclass(frozen=True)
class GroundTruthNode(JsonRecord):
    """
    One entity of a modelled world and the classes put on it.
    """

    name: str
    """
    The entity's name, which identifies it everywhere in this snapshot.
    """

    source_id: str
    """
    The entity's identity in the world it was read from.
    """

    parent: Optional[str]
    """
    The entity it hangs from, or nothing for the world's root.
    """

    semantic_classes: List[str]
    """
    The classes rooted at this entity, empty where the world names it nothing.
    """

    faces: int
    """
    How many faces the entity's own geometry has, zero where it has none.
    """

    world_transform: List[List[float]]
    """
    The entity's pose in the world's root frame.
    """

    bounds: Optional[List[List[float]]]
    """
    The lowest and highest corner of the entity's geometry in the world's root frame, or
    nothing where it has no geometry.
    """

    size: Optional[ObjectSize] = None
    """
    How big it is, measured so a reconstruction in another frame can be compared with
    it.
    """

    correction: Optional[ClassCorrection] = None
    """
    The class a person gave this entity, where the world does not name it correctly.
    """

    @property
    def classes(self) -> List[str]:
        """
        :return: The classes this entity stands as in a comparison, which is what a
            person decided where they decided anything and what the world says
            otherwise.
        """
        if self.correction is None:
            return self.semantic_classes
        return [self.correction.semantic_class]


@dataclass(frozen=True)
class GroundTruthEdge(JsonRecord):
    """
    One relation the modelled world holds between two of its entities.
    """

    whole: str
    """
    The entity at the holding end of the relation.
    """

    part: str
    """
    The entity at the held end of the relation.
    """

    relation: str
    """
    The semantic relation: part, contains, or supports.
    """

    field_name: str
    """
    The ontology field the relation is held in.
    """


@dataclass(frozen=True)
class GroundTruthGraph(JsonRecord):
    """
    Everything a modelled world says, in the shape a run's own graph is written in.
    """

    scene: str
    """
    What the world was built from.
    """

    frame: str
    """
    The frame every transform in this snapshot is expressed in.
    """

    geometry_source: str
    """
    Which of each entity's shape collections its geometry was taken from.
    """

    nodes: List[GroundTruthNode] = field(default_factory=list)
    """
    Every entity of the world in name order, including those carrying no class.
    """

    edges: List[GroundTruthEdge] = field(default_factory=list)
    """
    Every relation the world's annotations hold, in a stable order.
    """

    corrections: Optional[GroundTruthCorrections] = None
    """
    The ground truth a person supplied, where any was.
    """

    source_commit: Optional[str] = None
    """
    The commit of the rules that named this world, since a world carrying inferred
    classes says something different once those rules change.
    """

    source_dirty: Optional[bool] = None
    """
    Whether that commit had uncommitted changes, which the commit alone does not
    reproduce.
    """

    @classmethod
    def from_world(
        cls,
        world: World,
        *,
        scene: str,
        geometry_source: GeometrySource = (
            GeometrySource.VISUAL_WITH_COLLISION_FALLBACK
        ),
    ) -> GroundTruthGraph:
        """
        Read a world's entities, classes, and relations into a portable graph.

        :param world: The modelled world to read.
        :param scene: What the world was built from, recorded with the result.
        :param geometry_source: Which shapes an entity's geometry is taken from.
        :return: A graph that does not depend on the world staying loaded, written in a
            stable order so two exports of one world can be compared as they are.
        :raises RelationHasNoEntityError: If a relation reaches something the world
            gives no entity to, which this snapshot has no way to name.
        """
        snapshot = WorldMeshExtractor(geometry_source=geometry_source).extract(world)
        class_by_annotation_id = {
            annotation.identifier: annotation.type_name
            for annotation in snapshot.semantic_annotations
        }
        name_by_body_id = {body.identifier: body.name for body in snapshot.body_meshes}
        nodes = [
            GroundTruthNode(
                name=body.name,
                source_id=str(body.source_identifier),
                parent=name_by_body_id.get(body.parent_identifier),
                semantic_classes=[
                    class_by_annotation_id[annotation_id]
                    for annotation_id in body.direct_semantic_annotation_identifiers
                ],
                faces=0 if body.local_mesh is None else len(body.local_mesh.faces),
                world_transform=body.world_transform.tolist(),
                bounds=(
                    None if body.local_mesh is None else body.world_mesh.bounds.tolist()
                ),
                size=ObjectSize.of(body.local_mesh),
            )
            for body in snapshot.body_meshes
        ]
        nodes.extend(cls._region_nodes(world))
        return cls(
            scene=scene,
            frame=str(world.root.name),
            geometry_source=geometry_source.value,
            nodes=sorted(nodes, key=lambda node: node.name),
            edges=sorted(
                cls._edges(world),
                key=lambda edge: (
                    edge.whole,
                    edge.part,
                    edge.relation,
                    edge.field_name,
                ),
            ),
        )

    @staticmethod
    def _region_nodes(world: World) -> List[GroundTruthNode]:
        """
        Describe the world's regions, which carry a pose but no shapes.
        """
        return [
            GroundTruthNode(
                name=str(region.name),
                source_id=str(region.id),
                parent=str(region.parent_kinematic_structure_entity.name),
                semantic_classes=[
                    type(annotation).__name__
                    for annotation in world.semantic_annotations
                    if isinstance(annotation, HasRootKinematicStructureEntity)
                    and annotation.root is region
                ],
                faces=0,
                world_transform=world.compute_forward_kinematics_np(
                    world.root, region
                ).tolist(),
                bounds=None,
            )
            for region in world.regions
        ]

    @staticmethod
    def _edges(world: World) -> List[GroundTruthEdge]:
        """
        Read every relation the world's annotations currently hold.
        """
        return [
            GroundTruthEdge(
                whole=entity_name_of(annotation),
                part=entity_name_of(relation.target),
                relation=relation.kind.value,
                field_name=relation.field_name,
            )
            for annotation in world.semantic_annotations
            for relation in mounted_relations_of(annotation)
        ]


# %% applying ground truth a person supplied


@dataclass(frozen=True)
class RelationCorrection(JsonRecord):
    """
    A relation a person added, which the modelled world does not hold itself.

    Supplying an entity's class without the relations it stands in would leave a cabinet
    whose doors are named but not attached to it, costing a run precision on hierarchy
    it got right.
    """

    edge: GroundTruthEdge
    """
    The relation to add.
    """

    reason: str
    """
    Why the world does not hold it, so the decision can be reviewed.
    """


@dataclass(frozen=True)
class GroundTruthCorrections(JsonRecord):
    """
    Everything a person supplies about one scene's ground truth.

    This is versioned and read beside the world rather than written into it, because the
    reasoning that names a world is also used by the pipeline being measured: correcting
    the rules to make ground truth come out right would change the system under test in
    the same move.
    """

    scene: str
    """
    The scene these corrections were written for.
    """

    classes: List[ClassCorrection] = field(default_factory=list)
    """
    The entities a person gave a class the world does not name them with.
    """

    relations: List[RelationCorrection] = field(default_factory=list)
    """
    The relations a person added to the ones the world holds.
    """

    @classmethod
    def read(cls, path: Path) -> GroundTruthCorrections:
        """
        :param path: The overlay to read.
        :return: What it holds.
        """
        return cls.from_json(json.loads(Path(path).read_text()))

    def applied_to(self, graph: GroundTruthGraph) -> GroundTruthGraph:
        """
        Supplement a graph with what a person decided about the scene.

        What the world said is kept on every node it said it about, so ground truth a
        person supplied stays distinguishable from ground truth the world carried.

        :param graph: The graph read from the modelled world.
        :return: The same graph with the corrections applied and recorded.
        :raises CorrectedEntityNotInGraphError: If a correction names an entity the
            graph does not hold.
        :raises GroundTruthAlreadyCorrectedError: If the graph already carries an
            overlay, whose corrections this one would silently drop.
        """
        if graph.corrections is not None:
            raise GroundTruthAlreadyCorrectedError(
                scene=graph.scene, already_applied=graph.corrections.scene
            )
        correction_by_entity = {
            correction.entity: correction for correction in self.classes
        }
        names = {node.name for node in graph.nodes}
        missing = sorted(set(correction_by_entity) - names)
        if missing:
            raise CorrectedEntityNotInGraphError(scene=self.scene, entities=missing)
        return replace(
            graph,
            nodes=[
                replace(node, correction=correction_by_entity.get(node.name))
                for node in graph.nodes
            ],
            edges=graph.edges + [added.edge for added in self.relations],
            corrections=self,
        )


# %% naming the entity a relation reaches


def entity_name_of(held: WorldEntity) -> str:
    """
    Name the entity something occupies in the world.

    An annotation is named by the entity it is rooted at, because that is what carries
    its geometry and what a reconstructed body can correspond to. Annotation names
    themselves do not identify anything: an inferred world calls every handle ``Handle``.

    :param held: An annotation or an entity at one end of a relation.
    :return: The name of the entity it occupies.
    :raises RelationHasNoEntityError: If it occupies no entity.
    """
    if isinstance(held, KinematicStructureEntity):
        return str(held.name)
    if isinstance(held, HasRootKinematicStructureEntity):
        return str(held.root.name)
    raise RelationHasNoEntityError(held_type=type(held).__name__)


# %% reading a modelled world


def world_from_provider(reference: str) -> World:
    """
    Build a world from something that knows how to build one.

    A modelled world such as the kitchen is written as Python rather than as a file, and
    names its own classes as it builds, so nothing has to be inferred afterwards.

    :param reference: What builds it, as ``module.path:ClassName``.
    :return: The world it builds.
    :raises WorldProviderNotFoundError: If nothing of that name builds a world.
    """
    module_name, separator, provider_name = reference.partition(":")
    if not (module_name and separator and provider_name):
        raise WorldProviderNotFoundError(
            reference=reference, problem="not module:Class"
        )
    module = importlib.import_module(module_name)
    provider = vars(module).get(provider_name)
    if provider is None:
        raise WorldProviderNotFoundError(
            reference=reference, problem=f"{module_name} holds no {provider_name}"
        )
    world = provider().create_world()
    if not isinstance(world, World):
        raise WorldProviderNotFoundError(
            reference=reference, problem="create_world() did not return a World"
        )
    return world


def world_from_urdf(path: Path, *, infer_semantics: bool = True) -> World:
    """
    Load a world from a URDF file and name what it holds.

    A URDF describes bodies and joints and says nothing about what any of them is, so
    the classes have to be inferred before the world can serve as semantic ground truth.

    :param path: The URDF file to load.
    :param infer_semantics: Whether to run the reasoner that puts classes on bodies.
    :return: The loaded world.
    :raises FileNotFoundError: If the path is not a file.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {resolved}")
    world = URDFParser.from_file(str(resolved)).create_world()
    if infer_semantics:
        WorldReasoner(world).infer_semantic_annotations()
    return world
