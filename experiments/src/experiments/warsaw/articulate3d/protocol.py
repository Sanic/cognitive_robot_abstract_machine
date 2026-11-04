"""
What a person decided about Articulate3D before any run was judged against it.

Articulate3D's parts are named by whoever annotated the scene, so one kind of object comes
as ``switch`` in one scene and ``light switch`` or ``control`` in another, and one name
covers different kinds of object in different scenes. A ground truth read straight off
those names would measure how a run agrees with the annotators' wording. The protocol is
where that wording is turned into classes of the ontology, once for the whole dataset:

- every part name gets a ruling, which says what it means wherever it occurs;
- a part a person classified on sight overrides what its name means;
- where a whole could hold a part through several fields, the protocol says which;
- its annotation rules say which scenes are left out and which parts are corrected.

The protocol is written and frozen before any run on the evaluated scenes is looked at.
Nothing in it may be derived from what a run produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from typing_extensions import Iterable

from experiments.warsaw.articulate3d.exceptions import (
    ClassNotInOntologyError,
    PartNameNotRuledError,
    PartNameUndecidedError,
)
from experiments.warsaw.articulate3d.rules import AnnotationRules
from experiments.warsaw.bases import JsonRecord

# %% what one part name means


class PartNameRuling(StrEnum):
    """
    What a part name means wherever an annotator used it.
    """

    CLASS = "class"
    """
    The name means one class of the ontology.
    """

    AMBIGUOUS = "ambiguous"
    """
    The name covers objects of different classes, so its parts are judged as objects and
    as ends of relations, but their class is not judged.
    """

    IGNORED = "ignored"
    """
    The name marks nothing the evaluation covers, so its parts are left out of it
    altogether.
    """

    UNDECIDED = "undecided"
    """
    Nobody has decided yet. A ground truth cannot be built while a name it needs is
    undecided.
    """


@dataclass(frozen=True)
class PartNameMeaning(JsonRecord):
    """
    The ruling on one part name.
    """

    part_name: str
    """
    The name as annotators wrote it, without its position in the part hierarchy.
    """

    ruling: PartNameRuling = PartNameRuling.UNDECIDED
    """
    What the name means.
    """

    semantic_class: str | None = None
    """
    The class of the ontology the name means, where the ruling is that it means one.
    """


# %% what a person saw a part to be


@dataclass(frozen=True)
class PartClass(JsonRecord):
    """
    The class a person gave one part by looking at it rather than at its name.
    """

    scene_id: str
    """
    The scene the part is annotated in.
    """

    part_id: int
    """
    The part, as Articulate3D numbers the parts of a scene.
    """

    semantic_class: str | None = None
    """
    The class the part is, or nothing where the people classifying it could not settle on
    one.
    """


# %% which field holds a part


@dataclass(frozen=True)
class FieldChoice(JsonRecord):
    """
    The field through which a whole of one class holds a part of another, where the
    ontology admits several.
    """

    whole_class: str
    """
    The class of the whole.
    """

    part_class: str
    """
    The class of the part.
    """

    field_name: str
    """
    The field that holds it.
    """


# %% the whole protocol


@dataclass(frozen=True)
class EvaluationProtocol(JsonRecord):
    """
    Everything decided by hand about how Articulate3D's annotations become ground truth.
    """

    part_names: list[PartNameMeaning] = field(default_factory=list)
    """
    The ruling on every part name the dataset uses.
    """

    part_classes: list[PartClass] = field(default_factory=list)
    """
    The parts a person classified on sight, which overrides what their names mean.
    """

    field_choices: list[FieldChoice] = field(default_factory=list)
    """
    The field each ambiguous pair of classes is held through.
    """

    annotation_rules: AnnotationRules = field(default_factory=AnnotationRules)
    """
    Which scenes are left out, and how the annotation of the others is corrected.
    """

    @classmethod
    def read(cls, path: Path) -> EvaluationProtocol:
        """
        :param path: The protocol file.
        :return: What it holds.
        """
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path: Path) -> None:
        """
        :param path: Where to write the protocol.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    def listing(self, part_names: Iterable[str]) -> EvaluationProtocol:
        """
        :param part_names: Part names a dataset uses.
        :return: This protocol with every one of them it does not list yet added as
            undecided, every ruling it holds kept, and the names in name order.
        """
        ruled = {one.part_name: one for one in self.part_names}
        return replace(
            self,
            part_names=[
                ruled.get(part_name, PartNameMeaning(part_name=part_name))
                for part_name in sorted(set(ruled) | set(part_names))
            ],
        )

    def meaning_of(self, part_name: str) -> PartNameMeaning:
        """
        :param part_name: A part name as annotators wrote it.
        :return: The ruling on it.
        :raises PartNameNotRuledError: If the protocol does not list the name.
        :raises PartNameUndecidedError: If it lists the name without a ruling.
        """
        meaning = next(
            (one for one in self.part_names if one.part_name == part_name), None
        )
        if meaning is None:
            raise PartNameNotRuledError(part_name=part_name)
        if meaning.ruling is PartNameRuling.UNDECIDED:
            raise PartNameUndecidedError(part_name=part_name)
        return meaning

    def covers(self, part_name: str) -> bool:
        """
        :param part_name: A part name as annotators wrote it.
        :return: Whether parts of that name are part of the evaluation.
        """
        return self.meaning_of(part_name).ruling is not PartNameRuling.IGNORED

    def class_of(self, scene_id: str, part_id: int, part_name: str) -> str | None:
        """
        :param scene_id: The scene the part is annotated in.
        :param part_id: The part.
        :param part_name: The name its annotator gave it.
        :return: The class the part is judged against, or nothing where its class is not
            judged.
        """
        classified = next(
            (
                one
                for one in self.part_classes
                if one.scene_id == scene_id and one.part_id == part_id
            ),
            None,
        )
        if classified is not None:
            return classified.semantic_class
        return self.meaning_of(part_name).semantic_class

    def field_choice(self, whole_class: str, part_class: str) -> str | None:
        """
        :param whole_class: The class of a whole.
        :param part_class: The class of a part.
        :return: The field the protocol chose for the pair, or nothing where it chose none.
        """
        return next(
            (
                one.field_name
                for one in self.field_choices
                if one.whole_class == whole_class and one.part_class == part_class
            ),
            None,
        )

    def check_classes(self, ontology: set[str]) -> None:
        """
        Make sure every class the protocol names is one the ontology holds.

        :param ontology: The names of the ontology's classes.
        :raises ClassNotInOntologyError: For the first class it does not hold.
        """
        named = (
            [one.semantic_class for one in self.part_names]
            + [one.semantic_class for one in self.part_classes]
            + [one.whole_class for one in self.field_choices]
            + [one.part_class for one in self.field_choices]
        )
        for class_name in named:
            if class_name is not None and class_name not in ontology:
                raise ClassNotInOntologyError(class_name=class_name)
