"""
A world drawn as one graph of its entities and the relations between them.

What has to hold is that the page tells the truth about the world: every body and
annotation appears once, every connection joins the bodies it joins, and every field of
an annotation that holds something in the world is drawn to what it holds, under that
field's name.
"""

from __future__ import annotations

import json
import re

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
    Furniture,
    Handle,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.warsaw.world_graph import (
    GRAPH_ELEMENT_ID,
    EdgeKind,
    NodeKind,
    WorldGraph,
    WorldGraphEdge,
)

# %% a cabinet holding a drawer with a handle


def cabinet_world(handle_name: str = "handle") -> World:
    """
    :param handle_name: What the handle's body is called.
    :return: A cabinet whose drawer carries a handle, each on a body of its own.
    """
    world = World.create_with_root_body("root")
    cabinet_body, drawer_body, handle_body = (
        Body(name=PrefixedName(name)) for name in ("cabinet", "drawer", handle_name)
    )
    with world.modify_world():
        world.add_connection(FixedConnection(parent=world.root, child=cabinet_body))
        world.add_connection(FixedConnection(parent=cabinet_body, child=drawer_body))
        world.add_connection(FixedConnection(parent=drawer_body, child=handle_body))
        world.add_semantic_annotation_recursively(
            Cabinet(
                root=cabinet_body,
                drawers=[
                    Drawer(root=drawer_body, handles=[Handle(root=handle_body)]),
                ],
            )
        )
    return world


def the_one(world: World, annotation_type: type):
    """
    :param world: The world to look in.
    :param annotation_type: The class to look for.
    :return: The single annotation of exactly that class.
    """
    [annotation] = [
        one for one in world.semantic_annotations if type(one) is annotation_type
    ]
    return annotation


# %% what the graph holds


def test_every_entity_and_annotation_is_a_node_exactly_once():
    """
    The handle is reached both from the world and through the drawer holding it, and is
    still one node.
    """
    world = cabinet_world()

    graph = WorldGraph.from_world(world)

    expected = [str(one.id) for one in world.kinematic_structure_entities] + [
        str(one.id) for one in world.semantic_annotations
    ]
    assert sorted(node.identifier for node in graph.nodes) == sorted(expected)


def test_nodes_say_which_kind_of_thing_they_are():
    world = cabinet_world()

    kinds = {node.identifier: node.kind for node in WorldGraph.from_world(world).nodes}

    assert kinds[str(world.root.id)] == NodeKind.BODY
    assert kinds[str(the_one(world, Drawer).id)] == NodeKind.SEMANTIC_ANNOTATION


def test_every_connection_joins_its_parent_to_its_child():
    world = cabinet_world()

    edges = WorldGraph.from_world(world).edges_of_kind(EdgeKind.CONNECTION)

    assert sorted(edges, key=str) == sorted(
        (
            WorldGraphEdge(
                source=str(connection.parent.id),
                target=str(connection.child.id),
                label=type(connection).__name__,
                kind=EdgeKind.CONNECTION,
            )
            for connection in world.connections
        ),
        key=str,
    )


def test_a_drawer_held_by_a_cabinet_is_drawn_under_the_field_holding_it():
    world = cabinet_world()
    cabinet, drawer = the_one(world, Cabinet), the_one(world, Drawer)

    edges = WorldGraph.from_world(world).edges_of_kind(EdgeKind.ANNOTATION_REFERENCE)

    assert (
        WorldGraphEdge(
            source=str(cabinet.id),
            target=str(drawer.id),
            label="drawers",
            kind=EdgeKind.ANNOTATION_REFERENCE,
        )
        in edges
    )


def test_an_annotation_is_drawn_to_the_body_it_is_rooted_in():
    world = cabinet_world()
    drawer = the_one(world, Drawer)

    edges = WorldGraph.from_world(world).edges_of_kind(EdgeKind.ANNOTATED_ENTITY)

    assert [edge for edge in edges if edge.source == str(drawer.id)] == [
        WorldGraphEdge(
            source=str(drawer.id),
            target=str(drawer.root.id),
            label="root",
            kind=EdgeKind.ANNOTATED_ENTITY,
        )
    ]


def test_an_annotation_can_be_found_by_what_it_is_a_kind_of():
    """
    Searching for furniture has to find the cabinet, which is only called a Cabinet.
    """
    world = cabinet_world()
    cabinet = the_one(world, Cabinet)

    [node] = [
        node
        for node in WorldGraph.from_world(world).nodes
        if node.identifier == str(cabinet.id)
    ]

    assert node.classes[0] == Cabinet.__name__
    assert Furniture.__name__ in node.classes


# %% the page


def graph_embedded_in(page: str) -> dict:
    """
    :param page: The written page.
    :return: The graph data the page carries.
    """
    [embedded] = re.findall(
        rf'<script id="{GRAPH_ELEMENT_ID}" type="application/json">(.*?)</script>',
        page,
        flags=re.DOTALL,
    )
    return json.loads(embedded)


def test_the_page_carries_the_whole_graph(tmp_path):
    graph = WorldGraph.from_world(cabinet_world())

    written = graph.write_page(tmp_path / "world_graph.html")

    assert graph_embedded_in(written.read_text()) == graph.to_json()


def test_a_name_that_looks_like_markup_does_not_end_the_graph_data_early(tmp_path):
    graph = WorldGraph.from_world(cabinet_world(handle_name="</script><b>handle"))

    written = graph.write_page(tmp_path / "world_graph.html")

    assert graph_embedded_in(written.read_text()) == graph.to_json()


def test_opening_the_page_shows_the_written_file_in_a_browser(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    written = WorldGraph.from_world(cabinet_world()).open_page(
        tmp_path / "world_graph.html"
    )

    assert opened == [written.as_uri()]
