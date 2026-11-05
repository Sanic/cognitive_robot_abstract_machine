"""
Which bodies their label already names, which are to choose among the classes their
label was left open between, and which are asked what they are.
"""

from __future__ import annotations

import numpy as np

from experiments.warsaw.pipeline.records import LabelAnswer, Vocabulary
from experiments.warsaw.pipeline.steps.classify.step import BodiesByLabel
from experiments.warsaw.world_loader.scene import LabelSegment

# %% a scene of three labels


def segment(label: str, instance: int = 1) -> LabelSegment:
    """
    :return: A body of *label*.
    """
    return LabelSegment(class_name=label, instance=instance, face_indices=np.array([0]))


def vocabulary() -> Vocabulary:
    """
    :return: A drawer read as one class, a base left open between two, and a gadget read
        as nothing the ontology holds.
    """
    return Vocabulary(
        model="",
        scene="",
        labels=[
            LabelAnswer(label="drawer", class_name="Drawer"),
            LabelAnswer(label="base", class_name="Backplate", alternatives=["Handle"]),
            LabelAnswer(label="gadget", class_name=None),
        ],
    )


def names(sorted_out: BodiesByLabel) -> tuple[list[str], list[str], list[str]]:
    """
    :return: The names of the settled, the choosing and the unsettled bodies.
    """
    return (
        [str(one.name) for one in sorted_out.settled],
        [str(one.name) for one in sorted_out.choosing],
        [str(one.name) for one in sorted_out.unsettled],
    )


# %% sorting the bodies


def test_a_body_whose_label_means_one_class_needs_no_question():
    sorted_out = BodiesByLabel.of([segment("drawer")], vocabulary(), settle=True)

    assert names(sorted_out) == (["drawer_1"], [], [])


def test_a_body_whose_label_was_left_open_is_to_choose():
    sorted_out = BodiesByLabel.of([segment("base")], vocabulary(), settle=True)

    assert names(sorted_out) == ([], ["base_1"], [])


def test_a_body_whose_label_means_nothing_is_asked_what_it_is():
    sorted_out = BodiesByLabel.of([segment("gadget")], vocabulary(), settle=True)

    assert names(sorted_out) == ([], [], ["gadget_1"])


def test_without_settling_every_body_is_asked_what_it_is():
    bodies = [segment("drawer"), segment("base"), segment("gadget")]

    sorted_out = BodiesByLabel.of(bodies, vocabulary(), settle=False)

    assert names(sorted_out) == ([], [], ["drawer_1", "base_1", "gadget_1"])


def test_a_body_named_by_its_label_takes_the_class_the_label_was_read_as():
    named = BodiesByLabel.named_by_label(segment("drawer", 4), vocabulary())

    assert (named.name, named.class_name) == (
        "drawer_4",
        vocabulary().answer_for("drawer").class_name,
    )
