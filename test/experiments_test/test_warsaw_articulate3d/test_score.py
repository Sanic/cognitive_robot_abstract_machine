"""
Judging a run of an Articulate3D scene against the scene's ground truth.

The runs here are written by hand: which faces each body has, what each was classified
as, and what was mounted into what. The perfect run cuts out exactly the annotated
parts, classifies each as the ground truth does and mounts each part into its whole.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from semantic_digital_twin.semantic_annotations.part_whole import field_holding
from semantic_digital_twin.semantic_annotations.taxonomy_export import MountKind

from experiments.warsaw.articulate3d.convert import SceneConversion
from experiments.warsaw.articulate3d.ground_truth import GroundTruthFile
from experiments.warsaw.articulate3d.score import (
    ClassLineage,
    MountVerdict,
    SceneScore,
    ScoreFile,
    ScoreSummary,
    Share,
    main,
)
from experiments.warsaw.evaluation.graph import (
    EvaluationEdge,
    EvaluationGraph,
    EvaluationNode,
)
from experiments.warsaw.pipeline.run import Run, RunFile

from .scenes import kitchen_corner

# %% a run, written by hand


def node(name, predicted_class):
    """
    :return: A body the run classified.
    """
    return EvaluationNode(
        name=name,
        input_label=name.rsplit("_", 1)[0],
        predicted_class=predicted_class,
        faces=0,
        body_id=None,
        annotation_applied=predicted_class is not None,
    )


def mount(whole, part, field_name="", relation=MountKind.PART, accepted=True):
    """
    :return: A mount the run made.
    """
    return EvaluationEdge(
        whole=whole,
        part=part,
        relation=relation,
        field_name=field_name,
        accepted=accepted,
    )


@pytest.fixture
def lineage(taxonomy):
    """
    :return: What the ontology's classes derive from, with nothing generated.
    """
    return ClassLineage.from_taxonomy(taxonomy, {})


@pytest.fixture
def bodies():
    """
    :return: The faces of every body of the perfect run, one body per annotated part.
    """
    return {
        "cabinet_1": np.array([0, 1, 2, 3]),
        "door_1": np.array([4, 5]),
        "handle_1": np.array([6]),
        "drawer_1": np.array([7, 8]),
        "handle_2": np.array([9]),
        "switch_base_1": np.array([10, 11]),
        "control_1": np.array([12]),
    }


@pytest.fixture
def graph(ontology):
    """
    :return: The perfect run's classes and mounts.
    """

    def field_of(whole, part):
        return field_holding(ontology[whole], ontology[part]).field_name

    return EvaluationGraph(
        nodes=[
            node("cabinet_1", "Cabinet"),
            node("door_1", "Door"),
            node("handle_1", "Handle"),
            node("drawer_1", "Drawer"),
            node("handle_2", "Handle"),
            node("switch_base_1", "ElectricalDevice"),
            node("control_1", None),
        ],
        edges=[
            mount("cabinet_1", "door_1", field_of("Cabinet", "Door")),
            mount("door_1", "handle_1", field_of("Door", "Handle")),
            mount("cabinet_1", "drawer_1", field_of("Cabinet", "Drawer")),
            mount("drawer_1", "handle_2", field_of("Drawer", "Handle")),
            mount("switch_base_1", "control_1"),
        ],
    )


def judged(ground_truth, bodies, graph, lineage):
    """
    :return: The run, judged.
    """
    return SceneScore.between(
        run_name="run",
        ground_truth=ground_truth,
        bodies=bodies,
        graph=graph,
        lineage=lineage,
    )


def with_class(graph, body, predicted_class):
    """
    :return: The same run, with one body classified otherwise.
    """
    return replace(
        graph,
        nodes=[
            node(one.name, predicted_class) if one.name == body else one
            for one in graph.nodes
        ],
    )


# %% which body is which part


def test_a_body_made_of_exactly_a_parts_faces_is_that_part(
    ground_truth, bodies, graph, lineage
):
    score = judged(ground_truth, bodies, graph, lineage)

    assert score.correspondence.part_of_body == {
        name: part.part_id for name, part in zip(bodies, ground_truth.parts)
    }


def test_a_part_cut_in_two_keeps_one_half_and_is_counted_fragmented(
    ground_truth, bodies, graph, lineage
):
    bodies = {**bodies, "cabinet_1": np.array([0, 1]), "cabinet_2": np.array([2, 3])}

    correspondence = judged(ground_truth, bodies, graph, lineage).correspondence

    assert correspondence.fragmented_parts == [1]
    assert correspondence.unmatched_bodies == ["cabinet_2"]


def test_a_body_covering_two_parts_is_paired_with_one_and_counted_merged(
    ground_truth, bodies, graph, lineage
):
    bodies = {**bodies, "drawer_1": np.array([7, 8, 9])}
    del bodies["handle_2"]

    correspondence = judged(ground_truth, bodies, graph, lineage).correspondence

    assert correspondence.merged_bodies == ["drawer_1"]
    assert correspondence.unmatched_parts == [5]


def test_a_body_on_faces_nobody_annotated_is_left_out_rather_than_counted_wrong(
    ground_truth, bodies, graph, lineage
):
    bodies = {**bodies, "shelf_1": np.array([14, 15])}

    correspondence = judged(ground_truth, bodies, graph, lineage).correspondence

    assert correspondence.outside_bodies == ["shelf_1"]
    assert correspondence.unmatched_bodies == []


# %% classes


def test_the_perfect_run_gives_every_found_part_its_class(
    ground_truth, bodies, graph, lineage
):
    classes = judged(ground_truth, bodies, graph, lineage).classes
    classified = [one for one in ground_truth.parts if one.semantic_class is not None]

    assert classes.accuracy == Share(len(classified), len(classified))


def test_a_misclassified_part_is_still_found_and_counted_as_misclassified(
    ground_truth, bodies, graph, lineage
):
    score = judged(ground_truth, bodies, with_class(graph, "drawer_1", "Door"), lineage)

    assert score.correspondence.body_of_part[4] == "drawer_1"
    [drawer] = [one for one in score.classes.judged if one.part_id == 4]
    assert not drawer.correct


def test_an_answer_on_the_right_branch_earns_part_of_the_hierarchy(
    ground_truth, bodies, graph, lineage
):
    classes = judged(
        ground_truth, bodies, with_class(graph, "drawer_1", "Furniture"), lineage
    ).classes

    [drawer] = [one for one in classes.judged if one.part_id == 4]
    answered, true = lineage.ancestry("Furniture"), lineage.ancestry("Drawer")
    assert (
        drawer.shared_ancestry,
        drawer.predicted_ancestry,
        drawer.true_ancestry,
    ) == (
        len(answered & true),
        len(answered),
        len(true),
    )


def test_a_generated_class_is_judged_as_the_class_it_derives_from(
    ground_truth, bodies, graph, taxonomy
):
    lineage = ClassLineage.from_taxonomy(
        taxonomy, {"SlidingBox": ["Drawer", "HasHandle"]}
    )

    classes = judged(
        ground_truth, bodies, with_class(graph, "drawer_1", "SlidingBox"), lineage
    ).classes

    [drawer] = [one for one in classes.judged if one.part_id == 4]
    assert drawer.correct


def test_a_part_not_found_counts_against_recognition_but_not_accuracy(
    ground_truth, bodies, graph, lineage
):
    del bodies["door_1"]

    classes = judged(ground_truth, bodies, graph, lineage).classes

    assert classes.accuracy.count == classes.accuracy.total
    assert classes.recognised == Share(
        classes.accuracy.count, classes.accuracy.total + 1
    )


# %% the lineage of a class


def test_a_class_of_the_ontology_stands_for_itself(lineage):
    assert lineage.projected("Drawer") == "Drawer"


def test_a_generated_class_with_mixins_stands_for_its_one_class(taxonomy):
    lineage = ClassLineage.from_taxonomy(
        taxonomy, {"SlidingBox": ["Drawer", "HasHandle"]}
    )

    assert lineage.projected("SlidingBox") == "Drawer"


def test_a_generated_class_of_no_class_or_of_two_stands_for_none(taxonomy):
    lineage = ClassLineage.from_taxonomy(
        taxonomy,
        {"Gizmo": [taxonomy["root_name"]], "Hybrid": ["Table", "Cabinet"]},
    )

    assert lineage.projected("Gizmo") is None
    assert lineage.projected("Hybrid") is None
    assert lineage.ancestry("Hybrid") == lineage.ancestry("Table") | lineage.ancestry(
        "Cabinet"
    )


def test_ancestry_holds_neither_mixins_nor_the_root(lineage, taxonomy):
    mixins = {one["name"] for one in taxonomy["classes"] if one.get("mixin")}

    ancestry = lineage.ancestry("Drawer")

    assert "Drawer" in ancestry
    assert ancestry.isdisjoint(mixins | {taxonomy["root_name"]})


# %% relations


def test_the_perfect_run_makes_every_relation_and_nothing_else(
    ground_truth, bodies, graph, lineage
):
    relations = judged(ground_truth, bodies, graph, lineage).relations

    assert relations.precision == Share(len(graph.edges), len(graph.edges))
    assert relations.recall == Share(
        len(ground_truth.relations), len(ground_truth.relations)
    )


def test_hanging_every_part_off_the_outermost_whole_is_mostly_wrong(
    ground_truth, bodies, graph, lineage
):
    flat = replace(
        graph,
        edges=[mount("cabinet_1", name) for name in bodies if name != "cabinet_1"],
    )

    relations = judged(ground_truth, bodies, flat, lineage).relations

    assert relations.precision == Share(2, len(flat.edges))
    assert relations.recall == Share(2, len(ground_truth.relations))


def test_a_run_mounting_nothing_has_no_precision_rather_than_a_perfect_one(
    ground_truth, bodies, graph, lineage
):
    relations = judged(
        ground_truth, bodies, replace(graph, edges=[]), lineage
    ).relations

    assert relations.precision.value is None
    assert relations.recall == Share(0, len(ground_truth.relations))


def test_a_mount_reaching_outside_the_annotation_is_not_judged(
    ground_truth, bodies, graph, lineage
):
    bodies = {**bodies, "shelf_1": np.array([14, 15])}
    graph = replace(graph, edges=graph.edges + [mount("cabinet_1", "shelf_1")])

    relations = judged(ground_truth, bodies, graph, lineage).relations

    assert relations.mounts[-1].verdict is MountVerdict.OUTSIDE
    assert relations.precision.total == len(graph.edges) - 1


def test_refused_mounts_and_mounts_of_other_kinds_are_not_part_whole_mounts(
    ground_truth, bodies, graph, lineage
):
    graph = replace(
        graph,
        edges=graph.edges
        + [
            mount("cabinet_1", "control_1", accepted=False),
            mount("cabinet_1", "control_1", relation=MountKind.CONTAINS),
        ],
    )

    relations = judged(ground_truth, bodies, graph, lineage).relations

    assert len(relations.mounts) == len(graph.edges) - 2


def test_the_ontology_bounds_what_a_run_can_make(ground_truth, bodies, graph, lineage):
    relations = judged(ground_truth, bodies, graph, lineage).relations
    expressible = [one for one in ground_truth.relations if one.field_name is not None]

    assert relations.expressible == Share(len(expressible), len(ground_truth.relations))
    assert relations.same_field == Share(len(expressible), len(graph.edges))


# %% many runs together


def test_a_summary_adds_every_count_up(ground_truth, bodies, graph, lineage):
    score = judged(ground_truth, bodies, graph, lineage)

    summary = ScoreSummary.over([score, replace(score, run="other")])

    assert summary.parts == 2 * len(ground_truth.parts)
    assert summary.relations.recall == Share(
        2 * score.relations.recall.count, 2 * score.relations.recall.total
    )


# %% a run directory


@pytest.fixture
def finished_run(tmp_path, bodies, graph, taxonomy, ground_truth):
    """
    :return: A run of the converted scene, its files written as a run writes them, and
        the directory holding the scene's ground truth.
    """
    scene_directory = tmp_path / "scene"
    SceneConversion().write(kitchen_corner(), scene_directory)
    run = Run(directory=tmp_path / "run")
    run.directory.mkdir()
    run.write_json(
        RunFile.PROVENANCE, {"settings": {"scene_directory": str(scene_directory)}}
    )
    run.write_json(RunFile.TAXONOMY, taxonomy)
    run.write_record(RunFile.EVALUATION_GRAPH, graph)
    np.savez(run.path(RunFile.SPLIT_FACES), **bodies)
    ground_truth_directory = tmp_path / "ground_truth"
    ground_truth_directory.mkdir()
    ground_truth.write(
        GroundTruthFile.SUFFIX.path(ground_truth_directory, ground_truth.scene_id)
    )
    return run, ground_truth_directory


def test_a_run_directory_is_judged_against_its_scenes_ground_truth(
    finished_run, ground_truth, bodies, graph, lineage
):
    run, ground_truth_directory = finished_run

    assert SceneScore.of(run, ground_truth_directory) == SceneScore.between(
        run_name=run.directory.name,
        ground_truth=ground_truth,
        bodies=bodies,
        graph=graph,
        lineage=lineage,
    )


def test_the_command_writes_each_runs_score_and_the_summary(finished_run, tmp_path):
    run, ground_truth_directory = finished_run

    main(
        [
            str(run.directory),
            "--ground-truth",
            str(ground_truth_directory),
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    )

    written = SceneScore.from_json(
        json.loads((run.directory / ScoreFile.SCENE).read_text())
    )
    assert written == SceneScore.of(run, ground_truth_directory)
    assert ScoreSummary.from_json(
        json.loads((tmp_path / "summary.json").read_text())
    ) == ScoreSummary.over([written])
