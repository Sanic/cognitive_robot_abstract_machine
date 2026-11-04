"""
Carrying the decided mounts out: every one is a structural part, held in the field it
was measured for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.warsaw.pipeline.steps.annotate import MountAnnotations
from experiments.warsaw.scene_split import Pairing
from semantic_digital_twin.exceptions import CannotBeAPartOf
from semantic_digital_twin.world import World
from typing_extensions import Any, List

# %% wholes that record or refuse what they are given


@dataclass
class RecordingWhole:
    """
    Something that holds, which records what it was given and in which field.

    Named for the behaviour it exercises rather than for any one annotation class: what
    is checked is the mount, which every class that holds parts offers alike.
    """

    mounted: List[Any] = field(default_factory=list)
    """
    What was mounted, as ``(part, field)``.
    """

    def add(self, part, *, field_name: str = "") -> None:
        """
        :param part: The structural part to hold.
        :param field_name: The field to hold it in.
        """
        self.mounted.append((part, field_name))


@dataclass
class RefusingWhole:
    """
    Something that will not hold what it is offered.
    """

    def add(self, part, *, field_name: str = "") -> None:
        """
        :param part: The part it refuses.
        :raises CannotBeAPartOf: Always.
        """
        raise CannotBeAPartOf(self, part)


# %% carrying a mount out


def test_a_structural_part_is_mounted_into_the_field_it_was_measured_for(tmp_path):
    """
    A whole holds several drawers and several doors, so which field takes this one was
    decided when the candidates were gathered.
    """
    whole, part = RecordingWhole(), object()

    MountAnnotations(directory=tmp_path).mount(
        World(),
        {"a": whole, "b": part},
        [Pairing(whole="a", part="b", field_name="doors")],
    )

    assert whole.mounted == [(part, "doors")]


def test_a_whole_that_will_not_hold_the_part_is_recorded_as_refusing_it(tmp_path):
    """
    The class a body was given may simply not admit the part another step said it holds,
    and that is an answer about this pairing rather than a run that should not have got
    this far.
    """
    pairing = Pairing(whole="a", part="b", field_name="drawers")

    mounted = MountAnnotations(directory=tmp_path).mount(
        World(), {"a": RefusingWhole(), "b": object()}, [pairing]
    )

    assert [refused.pairing for refused in mounted.refused] == [pairing]
