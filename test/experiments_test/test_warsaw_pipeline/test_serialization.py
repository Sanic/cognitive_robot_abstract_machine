"""
How a record reaches a file and comes back, now that krrood does the writing.

A record is written by naming its class in the file and letting the dataclass say what
its fields are. Three things about that are easy to lose and are pinned here: a record
read from a model's reply carries no class name, a tuple must not come back a list, and
a member of a string enumeration must come back the member.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    CountedClaimants,
    LabelAnswer,
)
from experiments.warsaw.segment_relations import ClaimedFaces

# %% what a model says, which nothing has stamped


def test_an_answer_a_model_wrote_is_read_as_the_class_it_was_asked_on():
    """
    A reply is a record written by something that has never heard of these classes, so
    it carries no class name.

    The class it is read on is the class it is.
    """
    answer = LabelAnswer.spoken({"class": "Drawer", "confidence": 0.9})
    assert isinstance(answer, LabelAnswer)
    assert answer.class_name == "Drawer"


def test_an_answer_is_read_in_the_words_the_model_was_asked_for(replies):
    """
    Read against a reply a model really sent.

    The prompt asks for ``class``, because that is what the thing is called, and the
    field is ``class_name``, because ``class`` is a keyword. Reading the reply by the
    field's own name finds nothing there, and every label of the run comes back mapped
    to no class at all -- silently, since an answer naming nothing is a legitimate
    answer.
    """
    reply = json.loads((replies / "vocabulary_kitchen_island.json").read_text())
    said = reply["choices"][0]["message"]["content"]
    answer = LabelAnswer.spoken(
        json.loads(said[said.index("{") : said.rindex("}") + 1])
    )
    assert answer.class_name
    assert answer.is_usable


def test_a_body_s_answer_is_read_in_those_words_too():
    """
    The classification step asks for the same key, one object at a time.
    """
    answer = BodyAnswer.spoken({"name": "drawer_1", "class": "Drawer"})
    assert answer.name == "drawer_1"
    assert answer.class_name == "Drawer"


def test_an_answer_naming_a_class_by_hand_is_still_read():
    """
    A mapping written by hand to try something out names the class and nothing else.
    """
    assert LabelAnswer.of("Drawer").class_name == "Drawer"
    assert LabelAnswer.of({"class": "Drawer"}).class_name == "Drawer"


def test_a_record_the_pipeline_wrote_keeps_the_class_that_wrote_it():
    """
    A file names its own class, so reading it needs nothing but the file.
    """
    written = CountedClaimants(claimants=("a", "b"), faces=3).to_json()
    assert written["__json_type__"].endswith("CountedClaimants")


# %% what JSON cannot say on its own


def test_a_tuple_does_not_come_back_a_list():
    """
    The claimants are put in a set and used as a dictionary key, and a list is neither
    hashable nor equal to the tuple it was written from.
    """
    counted = CountedClaimants(claimants=("cabinet_8", "drawer_5"), faces=3)
    read = CountedClaimants.from_json(json.loads(json.dumps(counted.to_json())))
    assert read == counted
    assert isinstance(read.claimants, tuple)
    assert {read.claimants} == {("cabinet_8", "drawer_5")}


# %% the claimants, counted


def test_a_measured_group_of_claimants_is_counted_without_its_faces():
    """
    A group carries which faces are contested, which no file can hold; the record holds
    how many there are.
    """
    group = ClaimedFaces(names=("cabinet_8", "drawer_5"), faces=np.arange(7))
    counted = CountedClaimants.of(group)
    assert isinstance(counted, CountedClaimants)
    assert counted.claimants == group.names
    assert counted.faces == 7
