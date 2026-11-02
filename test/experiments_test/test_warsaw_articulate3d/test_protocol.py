"""
The protocol: what a person decided about the dataset's names before any run was judged.
"""

from __future__ import annotations

import pytest

from experiments.warsaw.articulate3d.exceptions import (
    ClassNotInOntologyError,
    PartNameNotRuledError,
    PartNameUndecidedError,
)
from experiments.warsaw.articulate3d.protocol import (
    EvaluationProtocol,
    PartNameMeaning,
    PartNameRuling,
)

from .scenes import SceneName, kitchen_corner_protocol

# %% listing the names a dataset uses


def test_listing_adds_every_name_not_listed_yet_as_undecided():
    listed = EvaluationProtocol().listing(["light switch", "door", "door"])

    assert listed.part_names == [
        PartNameMeaning(part_name="door"),
        PartNameMeaning(part_name="light switch"),
    ]


def test_listing_keeps_every_ruling_already_made():
    protocol = kitchen_corner_protocol()

    listed = protocol.listing(["door", "light switch"])

    assert protocol.meaning_of("door") == listed.meaning_of("door")
    assert listed.part_names[
        [one.part_name for one in listed.part_names].index("light switch")
    ] == PartNameMeaning(part_name="light switch")


# %% what a name means


def test_a_name_the_protocol_does_not_list_is_refused():
    with pytest.raises(PartNameNotRuledError):
        kitchen_corner_protocol().meaning_of("light switch")


def test_a_name_left_undecided_is_refused():
    with pytest.raises(PartNameUndecidedError):
        EvaluationProtocol().listing(["door"]).meaning_of("door")


def test_a_name_the_protocol_ignores_is_not_covered():
    protocol = kitchen_corner_protocol()

    assert not protocol.covers("unknown")
    assert protocol.covers("switch base")


def test_a_part_takes_the_class_its_name_means():
    protocol = kitchen_corner_protocol()

    assert (
        protocol.class_of(SceneName.KITCHEN_CORNER, 2, "door")
        == protocol.meaning_of("door").semantic_class
    )


def test_a_part_of_an_ambiguous_name_has_no_class_judged():
    assert (
        kitchen_corner_protocol().class_of(SceneName.KITCHEN_CORNER, 7, "control")
        is None
    )


def test_a_part_classified_on_sight_takes_that_class_whatever_its_name_means():
    protocol = kitchen_corner_protocol()
    [classified] = protocol.part_classes

    assert (
        protocol.class_of(classified.scene_id, classified.part_id, "switch base")
        == classified.semantic_class
    )


# %% the protocol against the ontology


def test_a_protocol_naming_a_class_the_ontology_lacks_is_refused():
    with pytest.raises(ClassNotInOntologyError):
        kitchen_corner_protocol().check_classes({"Cabinet", "Door"})


def test_a_protocol_naming_only_classes_the_ontology_holds_passes():
    protocol = kitchen_corner_protocol()
    named = {one.semantic_class for one in protocol.part_names} | {
        one.semantic_class for one in protocol.part_classes
    }

    protocol.check_classes(named - {None})


# %% the protocol file


def test_a_protocol_reads_back_as_it_was_written(tmp_path):
    protocol = kitchen_corner_protocol()

    protocol.write(tmp_path / "protocol.json")

    assert EvaluationProtocol.read(tmp_path / "protocol.json") == protocol


def test_a_ruling_reads_back_as_its_enumeration_member(tmp_path):
    kitchen_corner_protocol().write(tmp_path / "protocol.json")

    read = EvaluationProtocol.read(tmp_path / "protocol.json")

    assert read.meaning_of("unknown").ruling is PartNameRuling.IGNORED
