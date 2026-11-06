"""
Checking, in an interpreter of its own, that amended classes hold what they were amended
to hold.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.mixins import HasDoors, HasHandles
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
    Handle,
)

from experiments.warsaw.exceptions import AmendmentNotInForceError
from experiments.warsaw.pipeline.steps.amend.amended_classes_check import (
    AmendedClassesCheck,
    AmendedPair,
)

# %% handing the check over


def test_the_check_survives_the_crossing_into_its_interpreter(tmp_path):
    """
    The interpreter is handed the run's directory and the check's arguments, and has to
    rebuild the very check the step meant.
    """
    check = AmendedClassesCheck(
        pairs=[
            AmendedPair(whole=Cabinet.__name__, mixin=HasDoors.__name__),
            AmendedPair(whole=Drawer.__name__, mixin=HasHandles.__name__),
        ]
    )
    assert AmendedClassesCheck.from_arguments(check.arguments()) == check


# %% what the check finds


def test_classes_holding_their_mixins_pass():
    """
    A class that derives from the mixin it was amended with has the amendment in force.
    """
    AmendedClassesCheck(
        pairs=[AmendedPair(whole=Cabinet.__name__, mixin=HasDoors.__name__)]
    ).carry_out()


def test_a_class_missing_its_mixin_is_reported():
    """
    An amendment that was written but is not in force reads as done and is not, so the
    check names the pair rather than letting the run go on.
    """
    missing = AmendedPair(whole=Handle.__name__, mixin=HasDoors.__name__)
    with pytest.raises(AmendmentNotInForceError) as raised:
        AmendedClassesCheck(
            pairs=[
                AmendedPair(whole=Cabinet.__name__, mixin=HasDoors.__name__),
                missing,
            ]
        ).carry_out()
    assert raised.value.pairs == [missing]
