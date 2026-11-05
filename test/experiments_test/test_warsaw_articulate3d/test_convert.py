"""
Writing an Articulate3D scene as a scene the Warsaw pipeline reads.

What is checked is what the pipeline reads back: one segment per part, made of that
part's faces, with nothing of the part hierarchy reaching it.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.warsaw.articulate3d.convert import (
    ConvertedScene,
    LabelDisclosure,
    LabelOverlap,
    NameSpelling,
    SceneConversion,
    main,
)
from experiments.warsaw.articulate3d.exceptions import (
    ConvertedLabelsCollideError,
    SceneExcludedError,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol
from experiments.warsaw.articulate3d.rules import (
    AnnotationRules,
    MeshChanged,
    MeshSize,
    PartNameCorrection,
)
from experiments.warsaw.habitat.convert import label_property
from experiments.warsaw.world_loader.scene import SceneFrame, SourceFrame, WarsawScene

from .scenes import (
    SceneName,
    kitchen_corner,
    kitchen_corner_parts,
    part,
    row_of_squares,
)

# %% what the pipeline reads back


@pytest.fixture
def scene():
    """
    :return: The scene to convert.
    """
    return kitchen_corner()


def segments_read_back(directory):
    """
    :param directory: A converted scene.
    :return: The faces of each segment the pipeline reads from it, by segment name.
    """
    return {
        str(segment.name): sorted(segment.face_indices.tolist())
        for segment in WarsawScene.from_directory(directory).segments()
    }


def test_every_part_with_faces_is_read_back_as_a_segment_of_its_faces(scene, tmp_path):
    converted = SceneConversion().write(scene, tmp_path)

    by_part = {one.part_id: one for one in scene.parts}
    assert segments_read_back(tmp_path) == {
        one.segment: sorted(by_part[one.part_id].face_indices.tolist())
        for one in converted.parts
    }


def test_a_part_without_faces_is_recorded_rather_than_written(scene, tmp_path):
    converted = SceneConversion().write(scene, tmp_path)

    assert converted.faceless == [
        one.part_id for one in scene.parts if not len(one.face_indices)
    ]
    assert len(converted.parts) + len(converted.faceless) == len(scene.parts)


def test_labels_are_the_annotated_names_without_their_place_in_the_hierarchy(
    scene, tmp_path
):
    SceneConversion().write(scene, tmp_path)

    assert set(WarsawScene.from_directory(tmp_path).face_labels) == {
        label_property(one.part_name) for one in scene.parts if len(one.face_indices)
    }


def test_neutral_labels_name_no_part(scene, tmp_path):
    SceneConversion(disclosure=LabelDisclosure.NEUTRAL).write(scene, tmp_path)

    labels = set(WarsawScene.from_directory(tmp_path).face_labels)
    names = {label_property(one.part_name) for one in scene.parts}
    assert labels.isdisjoint(names)
    assert len(labels) == len(
        {one.part_name for one in scene.parts if len(one.face_indices)}
    )


def test_a_label_covering_the_parts_below_it_covers_their_faces(scene, tmp_path):
    converted = SceneConversion(overlap=LabelOverlap.SUBTREE).write(scene, tmp_path)

    [cabinet] = [one for one in converted.parts if one.label == "cabinet.1.1"]
    below = [
        one
        for one in scene.parts
        if one.hierarchy_path is not None and one.hierarchy_path.startswith("1.1.")
    ]
    assert segments_read_back(tmp_path)[cabinet.segment] == sorted(
        [0, 1, 2, 3] + [face for one in below for face in one.face_indices.tolist()]
    )


def test_two_names_written_as_one_label_are_refused(tmp_path):
    scene = kitchen_corner()
    scene.parts.append(part(10, "switch_base.4.1", [14]))

    with pytest.raises(ConvertedLabelsCollideError):
        SceneConversion().write(scene, tmp_path)


# %% the scan itself


def test_every_face_is_kept_in_its_place(scene, tmp_path):
    SceneConversion().write(scene, tmp_path)

    read = WarsawScene.from_directory(tmp_path).mesh
    source = scene.scannetpp_scene.mesh
    np.testing.assert_allclose(read.vertices[read.faces], source.vertices[source.faces])


def test_neighbouring_parts_meet_along_an_edge_once_written(scene, tmp_path):
    SceneConversion().write(scene, tmp_path)

    read = WarsawScene.from_directory(tmp_path).mesh
    assert len(read.face_adjacency) == len(read.faces) - 1


def test_the_scan_is_read_standing_as_it_was_written(scene, tmp_path):
    SceneConversion().write(scene, tmp_path)

    assert SceneFrame.beside(tmp_path).source is SourceFrame.UPRIGHT


# %% the record of where it came from


def test_the_record_reads_back_as_it_was_written(scene, tmp_path):
    converted = SceneConversion(
        disclosure=LabelDisclosure.NEUTRAL, overlap=LabelOverlap.SUBTREE, seed=3
    ).write(scene, tmp_path)

    assert ConvertedScene.beside(tmp_path) == converted


def test_the_same_seed_writes_the_same_scene(scene, tmp_path):
    first = SceneConversion(seed=5).write(scene, tmp_path / "first")
    second = SceneConversion(seed=5).write(scene, tmp_path / "second")

    assert first.parts == second.parts


# %% converting by the protocol's rules


def convert_kitchen_corner(dataset, tmp_path, rules, *options):
    """
    :param rules: The annotation rules of the protocol the scene is converted by.
    :param options: Further command-line options of the conversion.
    :return: The directory the scene is converted into.
    """
    protocol_path = tmp_path / "protocol.json"
    EvaluationProtocol(annotation_rules=rules).write(protocol_path)
    output = tmp_path / "converted"
    main(
        [
            "--dataset",
            str(dataset.articulate3d_directory),
            "--scannetpp",
            str(dataset.scannetpp_directory),
            "--protocol",
            str(protocol_path),
            "--scene",
            SceneName.KITCHEN_CORNER,
            "--output",
            str(output),
            *options,
        ]
    )
    return output


def test_a_scene_is_converted_as_the_rules_correct_it(dataset, tmp_path):
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    correction = PartNameCorrection(
        written="unknown", meant="handle", mistake="a spelling mistake"
    )

    output = convert_kitchen_corner(
        dataset,
        tmp_path,
        AnnotationRules(corrections=[correction]),
        "--spelling",
        NameSpelling.CORRECTED,
    )

    assert [
        one.label for one in ConvertedScene.beside(output).parts if one.part_id == 9
    ] == ["handle.3.1"]


def test_a_name_keeps_its_spelling_mistake_unless_asked_otherwise(dataset, tmp_path):
    """
    The pipeline is to cope with names as its labels come; the ground truth is still
    built from the corrected names.
    """
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    correction = PartNameCorrection(
        written="unknown", meant="handle", mistake="a spelling mistake"
    )

    output = convert_kitchen_corner(
        dataset, tmp_path, AnnotationRules(corrections=[correction])
    )

    converted = ConvertedScene.beside(output)
    assert [one.label for one in converted.parts if one.part_id == 9] == ["unknown.3.1"]
    assert converted.spelling is NameSpelling.AS_WRITTEN


def test_a_scene_the_rules_leave_out_is_not_converted(dataset, tmp_path):
    mesh = row_of_squares(8)
    dataset.write_scene(
        SceneName.KITCHEN_CORNER,
        kitchen_corner_parts(),
        mesh,
        recorded_mesh_size=MeshSize(
            vertices=len(mesh.vertices) + 1, faces=len(mesh.faces)
        ),
    )

    with pytest.raises(SceneExcludedError):
        convert_kitchen_corner(
            dataset, tmp_path, AnnotationRules(exclusions=[MeshChanged()])
        )
