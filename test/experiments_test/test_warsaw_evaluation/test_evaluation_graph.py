"""
Portable graph snapshots of the SDT produced by a pipeline run.
"""

from __future__ import annotations

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    Classifications,
    RefusedMount,
    SplitBody,
    SplitRecord,
)
from experiments.warsaw.scene_split import Pairing

# %% preserving the bodies and where they hang


def test_graph_snapshot_keeps_every_body_its_class_and_where_it_hangs() -> None:
    """
    Evaluation reads what each body was classified as and which body it hangs below.
    """
    accepted = Pairing(
        whole="cabinet_1",
        part="drawer_1",
        field_name="drawers",
    )
    refused = Pairing(
        whole="drawer_1",
        part="handle_1",
        field_name="handles",
    )
    split = SplitRecord(
        scene="apartment.glb",
        bodies=[
            SplitBody(name="cabinet_1", label="cabinet", faces=100, body_id="b1"),
            SplitBody(name="drawer_1", label="drawer", faces=40, body_id="b2"),
            SplitBody(name="handle_1", label="handle", faces=10, body_id="b3"),
        ],
        pairings=[accepted, refused],
        refused=[RefusedMount(pairing=refused, reason="class does not admit part")],
        world_id=7,
        annotated_world_id=8,
    )
    classifications = Classifications(
        scene="apartment.glb",
        model="scripted/model",
        bodies=[
            BodyAnswer(name="cabinet_1", label="cabinet", class_name="Cabinet"),
            BodyAnswer(name="drawer_1", label="drawer", class_name="Drawer"),
            BodyAnswer(name="handle_1", label="handle", class_name="Handle"),
        ],
    )

    graph = EvaluationGraph.from_run_products(
        split=split,
        classifications=classifications,
        annotated_names={"cabinet_1", "drawer_1"},
        parents={"cabinet_1": None, "drawer_1": "cabinet_1", "handle_1": None},
    )

    assert graph.source_world_id == 7
    assert graph.annotated_world_id == 8
    assert [node.name for node in graph.nodes] == [
        "cabinet_1",
        "drawer_1",
        "handle_1",
    ]
    assert graph.nodes[0].predicted_class == "Cabinet"
    assert graph.nodes[0].annotation_applied is True
    assert graph.nodes[2].annotation_applied is False
    assert [node.parent for node in graph.nodes] == [None, "cabinet_1", None]
    assert EvaluationGraph.from_json(graph.to_json()) == graph


def test_unclassified_body_remains_visible_in_snapshot() -> None:
    """
    A missed annotation is measurable rather than disappearing from evaluation.
    """
    split = SplitRecord(
        scene="apartment.glb",
        bodies=[SplitBody(name="unknown_1", label="object", faces=12, body_id="b1")],
    )

    graph = EvaluationGraph.from_run_products(
        split=split,
        classifications=Classifications(scene="apartment.glb", model="scripted/model"),
        annotated_names=set(),
        parents={"unknown_1": None},
    )

    assert graph.nodes[0].predicted_class is None
    assert graph.nodes[0].annotation_applied is False


# %% where each body hangs


def test_a_body_hangs_from_the_nearest_body_of_the_split_above_it():
    """
    What a mount puts between a part and its whole, such as a region, is not a body the
    run judged, so it is passed over.
    """
    world = World.create_with_root_body("root")
    cabinet, between, drawer = (
        Body(name=PrefixedName(name)) for name in ("cabinet_1", "region", "drawer_1")
    )
    with world.modify_world():
        for parent, child in (
            (world.root, cabinet),
            (cabinet, between),
            (between, drawer),
        ):
            world.add_connection(FixedConnection(parent=parent, child=child))

    parents = EvaluationGraph.parents_in(world, {"cabinet_1", "drawer_1"})

    assert parents == {"cabinet_1": None, "drawer_1": "cabinet_1"}
