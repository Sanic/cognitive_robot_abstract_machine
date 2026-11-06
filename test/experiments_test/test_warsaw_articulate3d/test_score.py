"""
Judging a run of an Articulate3D scene against the scene's ground truth.

The runs here are written by hand: which faces each body has, what each was classified
as, and which body each hangs below. The perfect run cuts out exactly the annotated
parts, classifies each as the ground truth does and hangs each part from its whole.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.warsaw.articulate3d.convert import SceneConversion
from experiments.warsaw.articulate3d.ground_truth import Expressibility, GroundTruthFile
from experiments.warsaw.articulate3d.score import (
    ClassLineage,
    SceneScore,
    ScoreFile,
    ScoreOverRuns,
    Share,
    main,
)
from experiments.warsaw.evaluation.graph import EvaluationGraph, EvaluationNode
from experiments.warsaw.pipeline.run import Run, RunFile

from .scenes import kitchen_corner

# %% a run, written by hand


def node(name, predicted_class, parent=None):
    """
    :return: A body the run classified, hanging below *parent*.
    """
    return EvaluationNode(
        name=name,
        input_label=name.rsplit("_", 1)[0],
        predicted_class=predicted_class,
        faces=0,
        body_id=None,
        annotation_applied=predicted_class is not None,
        parent=parent,
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
def graph():
    """
    :return: The perfect run's classes, and where each body hangs.
    """
    return EvaluationGraph(
        nodes=[
            node("cabinet_1", "Cabinet"),
            node("door_1", "Door", "cabinet_1"),
            node("handle_1", "Handle", "door_1"),
            node("drawer_1", "Drawer", "cabinet_1"),
            node("handle_2", "Handle", "drawer_1"),
            node("switch_base_1", "ElectricalDevice"),
            node("control_1", None, "switch_base_1"),
        ]
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
            replace(one, predicted_class=predicted_class) if one.name == body else one
            for one in graph.nodes
        ],
    )


def with_parents(graph, parents):
    """
    :return: The same run, with the bodies *parents* names hung otherwise.
    """
    return replace(
        graph,
        nodes=[
            replace(one, parent=parents[one.name]) if one.name in parents else one
            for one in graph.nodes
        ],
    )


def attachment_of(links, part_id):
    """
    :return: Where the run hung one annotated part.
    """
    [attachment] = [one for one in links.attachments if one.part_id == part_id]
    return attachment


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


# %% links


def test_the_perfect_run_makes_every_link_and_nothing_else(
    ground_truth, bodies, graph, lineage
):
    links = judged(ground_truth, bodies, graph, lineage).links

    assert links.precision == Share(
        len(ground_truth.relations), len(ground_truth.relations)
    )
    assert links.recall == links.precision


def test_hanging_every_part_off_the_outermost_whole_is_mostly_wrong(
    ground_truth, bodies, graph, lineage
):
    flat = with_parents(
        graph, {name: "cabinet_1" for name in bodies if name != "cabinet_1"}
    )

    links = judged(ground_truth, bodies, flat, lineage).links

    assert links.precision == Share(2, len(bodies) - 1)
    assert links.recall == Share(2, len(ground_truth.relations))


def test_a_run_hanging_nothing_has_no_precision_rather_than_a_perfect_one(
    ground_truth, bodies, graph, lineage
):
    loose = with_parents(graph, {name: None for name in bodies})

    links = judged(ground_truth, bodies, loose, lineage).links

    assert links.precision.value is None
    assert links.recall == Share(0, len(ground_truth.relations))


def test_a_part_hung_below_a_body_that_is_no_part_hangs_from_the_part_above_that(
    ground_truth, bodies, graph, lineage
):
    """
    A body the run cut that matches no part says nothing about the part hierarchy, so
    what lies past it is judged.
    """
    bodies = {**bodies, "cabinet_1": np.array([0, 1, 2]), "panel_1": np.array([3])}
    graph = replace(graph, nodes=graph.nodes + [node("panel_1", None, "cabinet_1")])
    graph = with_parents(graph, {"door_1": "panel_1"})

    links = judged(ground_truth, bodies, graph, lineage).links

    assert attachment_of(links, 2).correct


def test_a_part_the_annotation_hangs_from_nothing_is_wrong_when_hung_from_a_part(
    ground_truth, bodies, graph, lineage
):
    graph = with_parents(graph, {"switch_base_1": "cabinet_1"})

    links = judged(ground_truth, bodies, graph, lineage).links

    assert links.precision == Share(
        len(ground_truth.relations), len(ground_truth.relations) + 1
    )


def test_a_part_hung_from_its_whole_s_whole_is_wrong_but_on_the_right_branch(
    ground_truth, bodies, graph, lineage
):
    """
    Annotators hang a handle from its door or from a plate on it, and the ontology
    allows both, so a run choosing the other still placed the handle on the right
    object.
    """
    graph = with_parents(graph, {"handle_1": "cabinet_1"})

    handle = attachment_of(judged(ground_truth, bodies, graph, lineage).links, 3)

    assert not handle.correct
    assert (handle.predicted_ancestors, handle.true_ancestors) == ([1], [2, 1])
    assert handle.shared_ancestors == 1


def test_a_link_with_a_misclassified_end_is_right_but_not_as_a_triplet(
    ground_truth, bodies, graph, lineage
):
    links = judged(
        ground_truth, bodies, with_class(graph, "drawer_1", "Door"), lineage
    ).links
    with_both_classes_judged = [
        one
        for one in ground_truth.relations
        if one.expressibility is not Expressibility.UNDETERMINED
    ]
    through_the_drawer = [
        one for one in with_both_classes_judged if 4 in (one.whole, one.part)
    ]

    assert links.recall == Share(
        len(ground_truth.relations), len(ground_truth.relations)
    )
    assert links.triplet_recall == Share(
        len(with_both_classes_judged) - len(through_the_drawer),
        len(with_both_classes_judged),
    )
    assert links.triplet_precision == links.triplet_recall


def test_the_ontology_bounds_what_a_run_can_make(ground_truth, bodies, graph, lineage):
    links = judged(ground_truth, bodies, graph, lineage).links
    reachable = [one for one in ground_truth.relations if one.field_name is not None]

    assert links.reachable == Share(len(reachable), len(ground_truth.relations))
    assert links.reachable_recall == Share(len(reachable), len(reachable))


def test_a_relation_held_the_other_way_round_by_a_carrying_part_is_reachable(
    ground_truth, bodies, graph, lineage
):
    """
    Mounted through a field whose part carries its whole, the whole hangs from the part,
    which is the link the annotation makes.
    """
    [first, *others] = ground_truth.relations
    ground_truth = replace(
        ground_truth,
        relations=[replace(first, expressibility=Expressibility.CARRIED), *others],
    )

    links = judged(ground_truth, bodies, graph, lineage).links
    held_by_a_field = [
        one for one in ground_truth.relations if one.field_name is not None
    ]

    assert links.reachable == Share(len(held_by_a_field), len(ground_truth.relations))


# %% many runs together


def test_a_summary_adds_every_count_up(ground_truth, bodies, graph, lineage):
    score = judged(ground_truth, bodies, graph, lineage)

    summary = ScoreOverRuns.over([score, replace(score, run="other")])

    assert summary.parts == 2 * len(ground_truth.parts)
    assert summary.links.recall == Share(
        2 * score.links.recall.count, 2 * score.links.recall.total
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


def test_a_scene_not_from_articulate3d_is_judged_against_the_ground_truth_of_its_name(
    finished_run, ground_truth, tmp_path
):
    """
    A scan of our own has no record of a conversion; what it is called is its
    directory's name.
    """
    run, ground_truth_directory = finished_run
    own_scene = tmp_path / "our_own_kitchen"
    own_scene.mkdir()
    run.write_json(
        RunFile.PROVENANCE, {"settings": {"scene_directory": str(own_scene)}}
    )
    replace(ground_truth, scene_id=own_scene.name).write(
        GroundTruthFile.SUFFIX.path(ground_truth_directory, own_scene.name)
    )

    assert SceneScore.of(run, ground_truth_directory).scene_id == own_scene.name


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
    assert ScoreOverRuns.from_json(
        json.loads((tmp_path / "summary.json").read_text())
    ) == ScoreOverRuns.over([written])
