"""
Listing the part names the dataset's annotators used.
"""

from __future__ import annotations

from experiments.warsaw.articulate3d.labels import PartNameCensus, PartNameUse, main
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol
from experiments.warsaw.articulate3d.rules import (
    AnnotatedScene,
    AnnotationRules,
    MeshMissing,
    MeshSize,
)

from .scenes import (
    SceneName,
    kitchen_corner_parts,
    kitchen_corner_protocol,
    part,
    row_of_squares,
)

# %% counting the names


def test_every_name_is_counted_by_parts_and_by_scenes(dataset):
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    dataset.write_scene("hallway", [part(1, "handle.1.1", [0])], row_of_squares(1))

    census = PartNameCensus.of(
        [
            AnnotatedScene.read(
                scene_id, dataset.articulate3d_directory, dataset.scannetpp_directory
            )
            for scene_id in [SceneName.KITCHEN_CORNER, "hallway"]
        ]
    )

    assert census.uses[0] == PartNameUse(part_name="handle", parts=3, scenes=2)
    assert len(census.uses) == len({one.part_name for one in kitchen_corner_parts()})


def command_line(dataset, protocol_path):
    """
    :return: The arguments listing the dataset's names into a protocol.
    """
    return [
        "--dataset",
        str(dataset.articulate3d_directory),
        "--scannetpp",
        str(dataset.scannetpp_directory),
        "--protocol",
        str(protocol_path),
    ]


def test_scenes_the_rules_leave_out_are_not_counted(dataset, tmp_path):
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    dataset.write_scene(
        "hallway",
        [part(1, "light switch.1.1", [0])],
        mesh=None,
        recorded_mesh_size=MeshSize(vertices=3, faces=1),
    )
    protocol_path = tmp_path / "protocol.json"
    EvaluationProtocol(
        annotation_rules=AnnotationRules(exclusions=[MeshMissing()])
    ).write(protocol_path)

    main(command_line(dataset, protocol_path))

    assert [
        one.part_name for one in EvaluationProtocol.read(protocol_path).part_names
    ] == sorted({one.part_name for one in kitchen_corner_parts()})


# %% adding them to a protocol


def test_the_names_are_added_to_a_protocol_keeping_its_rulings(dataset, tmp_path):
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    dataset.write_scene(
        "hallway", [part(1, "light switch.1.1", [0])], row_of_squares(1)
    )
    protocol_path = tmp_path / "protocol.json"
    kitchen_corner_protocol().write(protocol_path)

    main(command_line(dataset, protocol_path))

    assert EvaluationProtocol.read(protocol_path) == kitchen_corner_protocol().listing(
        ["light switch"]
    )


def test_a_missing_protocol_is_made_with_every_name_undecided(dataset, tmp_path):
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    protocol_path = tmp_path / "protocol.json"

    main(command_line(dataset, protocol_path))

    assert EvaluationProtocol.read(protocol_path) == EvaluationProtocol().listing(
        one.part_name for one in kitchen_corner_parts()
    )
