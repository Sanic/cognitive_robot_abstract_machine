"""
Portable graph snapshots of the SDT produced by a pipeline run.
"""

from __future__ import annotations

from dataclasses import replace

from semantic_digital_twin.semantic_annotations.part_whole import field_holding
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.evaluation.graph import (
    EvaluationEdge,
    EvaluationGraph,
    EvaluationNode,
)
from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    Classifications,
    RefusedMount,
    SplitBody,
    SplitRecord,
)
from experiments.warsaw.scene_split import Pairing

# %% preserving nodes and relation outcomes


def test_graph_snapshot_keeps_accepted_and_refused_relations() -> None:
    """
    Evaluation can distinguish a proposed edge from one present in the final SDT.
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
    assert graph.edges[0].accepted is True
    assert graph.edges[1].accepted is False
    assert graph.edges[1].refusal_reason == "class does not admit part"
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
    )

    assert graph.nodes[0].predicted_class is None
    assert graph.nodes[0].annotation_applied is False


# %% naming the field a mount went through


def graph_with_an_unnamed_mount() -> EvaluationGraph:
    """
    Build a graph whose relation was mounted without naming a field, as a run's is.
    """
    return EvaluationGraph(
        nodes=[
            EvaluationNode(
                name="cabinet_5",
                input_label="cabinet",
                predicted_class="Cabinet",
                faces=100,
                body_id=None,
                annotation_applied=True,
            ),
            EvaluationNode(
                name="drawer_2",
                input_label="drawer",
                predicted_class="Drawer",
                faces=50,
                body_id=None,
                annotation_applied=True,
            ),
        ],
        edges=[
            EvaluationEdge(
                whole="cabinet_5",
                part="drawer_2",
                field_name="",
                accepted=True,
            )
        ],
    )


def test_a_mount_carried_out_without_a_field_is_told_which_it_used():
    """
    ``add()`` routes a part by its type and records nothing, but the field it routes to
    follows from the two classes, so it is recovered rather than lost.
    """
    resolved = graph_with_an_unnamed_mount().with_fields_resolved(
        annotation_classes(SemanticAnnotation)
    )

    assert resolved.edges[0].field_name == field_holding(Cabinet, Drawer).field_name


def test_a_field_the_run_did_record_is_left_alone():
    """
    What the run wrote is what the run did, and re-deriving it could only disagree.
    """
    graph = replace(
        graph_with_an_unnamed_mount(),
        edges=[
            replace(graph_with_an_unnamed_mount().edges[0], field_name="a_named_field")
        ],
    )

    resolved = graph.with_fields_resolved(annotation_classes(SemanticAnnotation))

    assert resolved.edges[0].field_name == "a_named_field"


def test_a_relation_whose_classes_cannot_say_keeps_having_no_field():
    """
    Guessing a field for a body the run never classified would invent a relation the run
    did not build.
    """
    graph = graph_with_an_unnamed_mount()
    graph = replace(
        graph,
        nodes=[replace(graph.nodes[0], predicted_class=None), graph.nodes[1]],
    )

    resolved = graph.with_fields_resolved(annotation_classes(SemanticAnnotation))

    assert resolved.edges[0].field_name == ""
