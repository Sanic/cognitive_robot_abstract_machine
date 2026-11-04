"""
A world drawn as one interactive graph of its entities and the relations between them.

    WorldGraph.from_world(world).open_page(Path("world_graph.html"))

The page shows bodies and regions joined by their connections, and each semantic
annotation joined to whatever its fields hold, so a drawer can be followed to the cabinet
holding it and to the body it is rooted in. It can be zoomed, searched by name or class,
and cut down to the annotations alone.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.code_generation.generator import CodeGenerator
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Region,
    SemanticAnnotation,
    WorldEntityWithID,
)
from typing_extensions import Any

GRAPH_ELEMENT_ID = "world-graph"
"""
The id of the element the page carries its graph data in.
"""

PAGE_TEMPLATE = "world_graph.html.jinja"
"""
The template the page is written from.
"""

VIS_NETWORK_SCRIPT = (
    "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"
)
"""
The graph drawing library the page loads, so opening it needs a network connection.
"""

# %% nodes and edges


class NodeKind(StrEnum):
    """
    What a node in the graph stands for.
    """

    BODY = "body"
    REGION = "region"
    SEMANTIC_ANNOTATION = "semantic_annotation"


class EdgeKind(StrEnum):
    """
    What an edge in the graph stands for.
    """

    CONNECTION = "connection"
    """
    A kinematic connection, from its parent to its child.
    """

    ANNOTATION_REFERENCE = "annotation_reference"
    """
    A field of an annotation holding another annotation.
    """

    ANNOTATED_ENTITY = "annotated_entity"
    """
    A field of an annotation holding a body or region.
    """


@dataclass(frozen=True)
class WorldGraphNode:
    """
    One entity or annotation of the world.
    """

    identifier: str
    """
    The id of the entity or annotation.
    """

    label: str
    """
    Its name.
    """

    kind: NodeKind
    """
    What it is.
    """

    classes: list[str]
    """
    Its class followed by the classes it inherits from within the world's ontology.
    """


@dataclass(frozen=True)
class WorldGraphEdge:
    """
    One relation between two nodes.
    """

    source: str
    """
    The identifier of the node the relation starts at.
    """

    target: str
    """
    The identifier of the node the relation points to.
    """

    label: str
    """
    The connection's class, or the name of the annotation field holding the target.
    """

    kind: EdgeKind
    """
    What the relation is.
    """


# %% the graph


@dataclass
class WorldGraph:
    """
    The entities and annotations of a world and the relations between them.
    """

    nodes_by_identifier: dict[str, WorldGraphNode] = field(default_factory=dict)
    """
    Every node, by its identifier.
    """

    edges: list[WorldGraphEdge] = field(default_factory=list)
    """
    Every relation.
    """

    introspector: DataclassOnlyIntrospector = field(
        default_factory=DataclassOnlyIntrospector
    )
    """
    Finds the public fields of an annotation.
    """

    @classmethod
    def from_world(cls, world: World) -> WorldGraph:
        """
        :param world: The world to draw.
        :return: Its entities, connections, annotations and what the annotations hold.
        """
        graph = cls()
        for entity in world.kinematic_structure_entities:
            graph.add_entity(entity)
        for connection in world.connections:
            graph.edges.append(
                WorldGraphEdge(
                    source=str(connection.parent.id),
                    target=str(connection.child.id),
                    label=type(connection).__name__,
                    kind=EdgeKind.CONNECTION,
                )
            )
        for annotation in world.semantic_annotations:
            graph.add_annotation(annotation)
        return graph

    @property
    def nodes(self) -> list[WorldGraphNode]:
        """
        :return: Every node.
        """
        return list(self.nodes_by_identifier.values())

    def edges_of_kind(self, kind: EdgeKind) -> list[WorldGraphEdge]:
        """
        :param kind: The kind of relation wanted.
        :return: The relations of that kind.
        """
        return [edge for edge in self.edges if edge.kind == kind]

    def add_entity(self, entity: KinematicStructureEntity) -> None:
        """
        Add a body or region as a node, unless it is already one.

        :param entity: The body or region.
        """
        kind = NodeKind.REGION if isinstance(entity, Region) else NodeKind.BODY
        self.add_node(entity, kind, KinematicStructureEntity)

    def add_annotation(self, annotation: SemanticAnnotation) -> None:
        """
        Add an annotation as a node with an edge to everything its fields hold, and add
        the annotations it holds the same way.

        :param annotation: The annotation.
        """
        if str(annotation.id) in self.nodes_by_identifier:
            return
        self.add_node(annotation, NodeKind.SEMANTIC_ANNOTATION, SemanticAnnotation)
        for discovered in self.introspector.discover(type(annotation)):
            for held in self.values_held_by(vars(annotation)[discovered.public_name]):
                self.add_held(annotation, discovered.public_name, held)

    def add_held(
        self, annotation: SemanticAnnotation, field_name: str, held: Any
    ) -> None:
        """
        Add an edge from an annotation to one thing a field of it holds, if that is an
        annotation, body or region.

        :param annotation: The annotation holding it.
        :param field_name: The field it is held in.
        :param held: What the field holds.
        """
        if isinstance(held, SemanticAnnotation):
            self.add_annotation(held)
            kind = EdgeKind.ANNOTATION_REFERENCE
        elif isinstance(held, KinematicStructureEntity):
            self.add_entity(held)
            kind = EdgeKind.ANNOTATED_ENTITY
        else:
            return
        self.edges.append(
            WorldGraphEdge(
                source=str(annotation.id),
                target=str(held.id),
                label=field_name,
                kind=kind,
            )
        )

    @staticmethod
    def values_held_by(value: Any) -> list[Any]:
        """
        :param value: What a field holds.
        :return: The things in it, if it is a collection, or the value itself.
        """
        if isinstance(value, CollectionKind.ELEMENT_SEQUENCE.value):
            return list(value)
        return [value]

    def add_node(
        self, entity: WorldEntityWithID, kind: NodeKind, ontology_root: type
    ) -> None:
        """
        Add a node for an entity or annotation, unless it is already one.

        :param entity: The entity or annotation.
        :param kind: What it is.
        :param ontology_root: The most general class its listed classes may be.
        """
        identifier = str(entity.id)
        if identifier in self.nodes_by_identifier:
            return
        self.nodes_by_identifier[identifier] = WorldGraphNode(
            identifier=identifier,
            label=str(entity.name.name),
            kind=kind,
            classes=[
                one.__name__
                for one in type(entity).__mro__
                if issubclass(one, ontology_root)
            ],
        )

    # %% the page

    def to_json(self) -> dict[str, Any]:
        """
        :return: The nodes and edges as the page reads them.
        """
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }

    def page(self) -> str:
        """
        :return: A self-contained HTML page drawing this graph.
        """
        return CodeGenerator(
            template_directory=str(Path(__file__).resolve().parent / "templates")
        ).render(
            PAGE_TEMPLATE,
            graph_json=self.json_safe_inside_a_script(),
            graph_element_id=GRAPH_ELEMENT_ID,
            vis_network_script=VIS_NETWORK_SCRIPT,
            node_kind=NodeKind,
            edge_kind=EdgeKind,
        )

    def json_safe_inside_a_script(self) -> str:
        """
        :return: The graph as JSON that no name in it can end the script element early.
        """
        return (
            json.dumps(self.to_json())
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )

    def write_page(self, path: Path) -> Path:
        """
        :param path: Where to write the page.
        :return: The written page, as an absolute path.
        """
        path = Path(path).resolve()
        path.write_text(self.page())
        return path

    def open_page(self, path: Path) -> Path:
        """
        Write the page and open it in a browser.

        :param path: Where to write the page.
        :return: The written page, as an absolute path.
        """
        written = self.write_page(path)
        webbrowser.open(written.as_uri())
        return written
