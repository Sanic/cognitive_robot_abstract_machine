"""
The annotation rules: which scenes are left out, and how the others are corrected.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.warsaw.articulate3d.protocol import EvaluationProtocol
from experiments.warsaw.articulate3d.rules import (
    AmbiguousWhole,
    AnnotatedScene,
    AnnotationRules,
    ArticulationContradictsHierarchy,
    FaceCountMisrecorded,
    InstanceNumberCorrection,
    MeshChanged,
    MeshMissing,
    MeshSize,
    PartNameCorrection,
    RuleFindings,
    SpacingCorrection,
)

from .scenes import (
    SceneName,
    articulation,
    kitchen_corner_parts,
    part,
    row_of_squares,
)

# %% the scene the rules are applied to


def kitchen_corner_annotation(**changes) -> AnnotatedScene:
    """
    :param changes: Fields to give other values than the scene's.
    :return: The kitchen corner as the rules see it, its mesh the one annotated.
    """
    mesh = row_of_squares(8)
    size = MeshSize(vertices=len(mesh.vertices), faces=len(mesh.faces))
    return replace(
        AnnotatedScene(
            scene_id=SceneName.KITCHEN_CORNER,
            parts=kitchen_corner_parts(),
            articulations=[],
            recorded_mesh_size=size,
            mesh_size=size,
        ),
        **changes,
    )


def scene_with_labels(*written: str) -> AnnotatedScene:
    """
    :param written: The labels of the scene's parts, numbered from one.
    :return: The kitchen corner with only those parts.
    """
    return kitchen_corner_annotation(
        parts=[part(number, label, [0]) for number, label in enumerate(written, 1)]
    )


def labels(scene: AnnotatedScene) -> dict[int, str]:
    """
    :return: The label of every part of the scene, by part.
    """
    return {one.part_id: one.label for one in scene.parts}


# %% reading a scene


def test_a_scene_is_read_with_the_size_of_its_mesh(dataset):
    mesh = row_of_squares(8)
    dataset.write_scene(SceneName.KITCHEN_CORNER, kitchen_corner_parts(), mesh)

    read = AnnotatedScene.read(
        SceneName.KITCHEN_CORNER,
        dataset.articulate3d_directory,
        dataset.scannetpp_directory,
    )

    assert read.mesh_size == MeshSize(
        vertices=len(mesh.vertices), faces=len(mesh.faces)
    )


def test_a_scene_without_a_mesh_is_read_without_its_size(dataset):
    dataset.write_scene(
        SceneName.KITCHEN_CORNER,
        kitchen_corner_parts(),
        mesh=None,
        recorded_mesh_size=MeshSize(vertices=48, faces=16),
    )

    read = AnnotatedScene.read(
        SceneName.KITCHEN_CORNER,
        dataset.articulate3d_directory,
        dataset.scannetpp_directory,
    )

    assert read.mesh_size is None


# %% scenes whose mesh does not fit


def test_a_scene_without_a_mesh_is_left_out():
    assert MeshMissing().finding(kitchen_corner_annotation(mesh_size=None)) is not None


def test_a_scene_with_its_mesh_is_kept_by_every_mesh_rule():
    scene = kitchen_corner_annotation()

    assert [
        rule.finding(scene)
        for rule in (MeshMissing(), MeshChanged(), FaceCountMisrecorded())
    ] == [None, None, None]


def test_a_mesh_of_other_vertices_than_annotated_is_a_changed_mesh():
    scene = kitchen_corner_annotation()
    changed = replace(
        scene,
        mesh_size=MeshSize(
            vertices=scene.recorded_mesh_size.vertices + 1,
            faces=scene.recorded_mesh_size.faces + 1,
        ),
    )

    assert [
        rule.finding(changed) is not None
        for rule in (MeshChanged(), FaceCountMisrecorded())
    ] == [True, False]


def test_a_wrong_number_of_faces_on_the_annotated_mesh_is_misrecorded():
    scene = kitchen_corner_annotation()
    misrecorded = replace(
        scene,
        recorded_mesh_size=replace(
            scene.recorded_mesh_size, faces=scene.recorded_mesh_size.faces // 2
        ),
    )

    assert [
        rule.finding(misrecorded) is not None
        for rule in (MeshChanged(), FaceCountMisrecorded())
    ] == [False, True]


# %% articulations that contradict the part hierarchy


@pytest.mark.parametrize(
    "base_part_ids",
    [[2], [6, 2]],
    ids=["its parent", "its parent among others"],
)
def test_a_part_moving_relative_to_its_parent_fits(base_part_ids):
    scene = kitchen_corner_annotation(articulations=[articulation(3, base_part_ids)])

    assert ArticulationContradictsHierarchy().finding(scene) is None


@pytest.mark.parametrize(
    "moving",
    [
        articulation(3, [4]),
        articulation(10, [2]),
        articulation(3, [10]),
        articulation(6, [1]),
    ],
    ids=[
        "base not its parent",
        "part not annotated",
        "base not annotated",
        "top of the hierarchy with a base",
    ],
)
def test_an_articulation_against_the_hierarchy_leaves_the_scene_out(moving):
    scene = kitchen_corner_annotation(articulations=[moving])

    assert ArticulationContradictsHierarchy().finding(scene) is not None


# %% wholes the hierarchy does not settle


def test_parts_below_a_place_two_parts_share_leave_the_scene_out():
    scene = scene_with_labels("door.1.1.2_1", "hinge.1.1.2_1", "handle.1.1.2_1.3_1")

    assert AmbiguousWhole().finding(scene) is not None


def test_two_parts_sharing_a_place_with_nothing_below_keep_the_scene():
    scene = scene_with_labels("cabinet.1.1", "door.1.1.2_1", "hinge.1.1.2_1")

    assert AmbiguousWhole().finding(scene) is None


# %% correcting part names


@pytest.mark.parametrize(
    "rule, written, meant",
    [
        (SpacingCorrection(), "switch  base .1.1", "switch base.1.1"),
        (InstanceNumberCorrection(), "box_1.1.1", "box.1.1"),
        (InstanceNumberCorrection(), "hinge1.1.1.2_1", "hinge.1.1.2_1"),
        (
            PartNameCorrection(
                written="hindge", meant="hinge", mistake="a spelling mistake"
            ),
            "hindge.1.1.2_1",
            "hinge.1.1.2_1",
        ),
    ],
    ids=[
        "spacing",
        "instance number",
        "instance number without underscore",
        "spelling",
    ],
)
def test_a_name_is_corrected_keeping_its_place_in_the_hierarchy(rule, written, meant):
    corrected = rule.correct(scene_with_labels(written, "door.2.1"))

    assert labels(corrected.scene) == {1: meant, 2: "door.2.1"}


@pytest.mark.parametrize(
    "rule",
    [
        SpacingCorrection(),
        InstanceNumberCorrection(),
        PartNameCorrection(
            written="hindge", meant="hinge", mistake="a spelling mistake"
        ),
    ],
    ids=["spacing", "instance number", "spelling"],
)
def test_a_name_correction_leaves_a_scene_of_correct_names_alone(rule):
    assert rule.correct(scene_with_labels("hinge.1.1.2_1", "light switch")) is None


def test_a_spelling_correction_corrects_only_the_name_it_names():
    corrected = PartNameCorrection(
        written="handle f", meant="handle", mistake="a spelling mistake"
    ).correct(scene_with_labels("handle f.1.1", "handle fix.2.1"))

    assert labels(corrected.scene) == {1: "handle.1.1", 2: "handle fix.2.1"}


# %% applying every rule


def rules() -> AnnotationRules:
    """
    :return: Every exclusion, and one correction of a part name.
    """
    return AnnotationRules(
        exclusions=[
            MeshMissing(),
            MeshChanged(),
            FaceCountMisrecorded(),
            ArticulationContradictsHierarchy(),
        ],
        corrections=[
            PartNameCorrection(
                written="unknown", meant="handle", mistake="a spelling mistake"
            )
        ],
    )


def test_a_scene_kept_is_corrected_and_says_by_what():
    corrected_alone = rules().corrections[0].correct(kitchen_corner_annotation())

    ruled = rules().apply(kitchen_corner_annotation())

    assert (
        ruled.excluded,
        labels(ruled.scene),
        [one.rule for one in ruled.corrections],
    ) == (False, labels(corrected_alone.scene), rules().corrections)


def test_a_scene_left_out_says_by_which_rules_and_is_not_corrected():
    scene = kitchen_corner_annotation(mesh_size=None)

    ruled = rules().apply(scene)

    assert (
        ruled.excluded,
        labels(ruled.scene),
        [one.rule for one in ruled.exclusions],
        ruled.corrections,
    ) == (True, labels(scene), [MeshMissing()], [])


def test_the_findings_of_many_scenes_are_collected():
    kept = rules().apply(kitchen_corner_annotation())
    left_out = rules().apply(
        kitchen_corner_annotation(scene_id="hallway", mesh_size=None)
    )

    findings = RuleFindings.of([kept, left_out])

    assert (findings.exclusions, findings.corrections) == (
        left_out.exclusions,
        kept.corrections,
    )


def test_a_protocol_reads_its_rules_back_as_they_were_written(tmp_path):
    protocol = EvaluationProtocol(annotation_rules=rules())
    path = tmp_path / "protocol.json"

    protocol.write(path)

    assert EvaluationProtocol.read(path) == protocol
