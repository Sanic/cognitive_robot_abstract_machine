"""
Listing the part names the dataset's annotators used.
"""

from __future__ import annotations

from experiments.warsaw.articulate3d.labels import PartNameCensus, PartNameUse, main
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol

from .scenes import (
    SceneName,
    kitchen_corner_parts,
    kitchen_corner_protocol,
    part,
    write_parts_file,
)

# %% counting the names


def test_every_name_is_counted_by_parts_and_by_scenes(tmp_path):
    write_parts_file(tmp_path, SceneName.KITCHEN_CORNER, kitchen_corner_parts())
    write_parts_file(tmp_path, "hallway", [part(1, "handle.1.1", [0])])

    census = PartNameCensus.of(tmp_path, [SceneName.KITCHEN_CORNER, "hallway"])

    assert census.uses[0] == PartNameUse(part_name="handle", parts=3, scenes=2)
    assert len(census.uses) == len({one.part_name for one in kitchen_corner_parts()})


# %% adding them to a protocol


def test_the_names_are_added_to_a_protocol_keeping_its_rulings(tmp_path):
    write_parts_file(tmp_path, SceneName.KITCHEN_CORNER, kitchen_corner_parts())
    write_parts_file(tmp_path, "hallway", [part(1, "light switch.1.1", [0])])
    protocol_path = tmp_path / "protocol.json"
    kitchen_corner_protocol().write(protocol_path)

    main(["--dataset", str(tmp_path), "--protocol", str(protocol_path)])

    assert EvaluationProtocol.read(protocol_path) == kitchen_corner_protocol().listing(
        ["light switch"]
    )


def test_a_missing_protocol_is_made_with_every_name_undecided(tmp_path):
    write_parts_file(tmp_path, SceneName.KITCHEN_CORNER, kitchen_corner_parts())
    protocol_path = tmp_path / "protocol.json"

    main(["--dataset", str(tmp_path), "--protocol", str(protocol_path)])

    assert EvaluationProtocol.read(protocol_path) == EvaluationProtocol().listing(
        one.part_name for one in kitchen_corner_parts()
    )
