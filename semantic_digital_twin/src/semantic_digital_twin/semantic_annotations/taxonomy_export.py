"""
Export the semantic annotation taxonomy in the form a model has to read it: every class
with its place in the hierarchy and the parts it can hold. Only part-whole relations are
reported; what merely rests on or inside something is no part of it.
"""

from __future__ import annotations

import importlib

import inspect
from abc import ABC
import json
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing_extensions import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

from semantic_digital_twin.semantic_annotations.part_whole import (
    part_whole_fields,
)

# %% the vocabulary a mount is described in


MIXIN_MODULE_SUFFIX = "semantic_annotations.mixins"
"""
Where the classes that exist to contribute relations live.
"""

INFRASTRUCTURE_FIELDS = frozenset(
    {"root", "name", "id", "simulator_additional_properties"}
)
"""
Fields every annotation carries, which say nothing about what it is.

``root`` in particular is declared by every annotation, so reporting it tells a model
nothing that distinguishes one class from another.
"""


@dataclass
class SemanticRelation:
    """
    One part an annotation class can hold, as a model is told about it.
    """

    field_name: str
    """
    The field holding the related annotation.
    """

    target: str
    """
    The name of the type the field accepts.
    """

    holds_many: bool
    """
    Whether the field holds several of them rather than one.
    """

    target_class: Type
    """
    The type the field accepts, so that what may be mounted into it can be asked rather
    than compared by name. Left out of the exported form, which names it.
    """

    removes_geometry: bool = False
    """
    Whether mounting here cuts the part's volume out of the whole's geometry.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The relation as JSON-ready data.
        """
        rendered = {
            "field": self.field_name,
            "target": self.target,
            "many": self.holds_many,
        }
        if self.removes_geometry:
            rendered["removes_geometry"] = True
        return rendered


# %% what relations a class can stand in


def _own_field_names(annotation_class: Type) -> Set[str]:
    """
    :param annotation_class: The class to inspect.
    :return: The names of the fields that class declares itself, rather than inherits.
    """
    if not is_dataclass(annotation_class):
        return set()
    inherited = {
        base_field.name
        for base in annotation_class.__mro__[1:]
        if is_dataclass(base)
        for base_field in dataclass_fields(base)
    }
    return {
        own_field.name for own_field in dataclass_fields(annotation_class)
    } - inherited


def relations_of(annotation_class: Type) -> List[SemanticRelation]:
    """
    Report every relation an annotation class can stand in.

    :param annotation_class: The class to inspect.
    :return: Its part-whole relations.
    """
    return [
        SemanticRelation(
            field_name=part_whole_relationship_field.field_name,
            target=part_whole_relationship_field.part.__name__,
            holds_many=part_whole_relationship_field.holds_many,
            removes_geometry=part_whole_relationship_field.removes_part_geometry_from_whole,
            target_class=part_whole_relationship_field.part,
        )
        for part_whole_relationship_field in part_whole_fields(annotation_class)
    ]


def describe_class(annotation_class: Type) -> str:
    """
    Write out what a class is and what it can hold, as a model reads it.

    :param annotation_class: The class to describe.
    :return: Its declaration and one line per relation it can stand in.
    """
    bases = ", ".join(base.__name__ for base in annotation_class.__bases__)
    lines = [f"{annotation_class.__name__}({bases})"]
    for relation in relations_of(annotation_class):
        many = " (many)" if relation.holds_many else ""
        cuts = (
            " -- mounting here cuts the part out of the whole"
            if relation.removes_geometry
            else ""
        )
        lines.append(f"  part {relation.field_name} -> {relation.target}{many}{cuts}")
    return "\n".join(lines)


def declares_itself_a_category(annotation_class: Type) -> bool:
    """
    Say whether a class says of itself that it is a category.

    Python calls a class abstract only where it carries an unimplemented abstract method,
    and an ontology's categories carry none: ``Furniture`` declares ``ABC`` and is
    instantiable all the same. Declaring ``ABC`` directly is how this ontology says a
    class names a kind of thing rather than a thing.

    :param annotation_class: The class to judge.
    :return: Whether it declares itself a category.
    """
    return ABC in annotation_class.__bases__


def names_a_category(
    annotation_class: Type, categories_are_answers: bool = False
) -> bool:
    """
    Say whether a class is no answer about an object.

    Two different things make a class unanswerable and only one of them is a choice. A
    class carrying an unimplemented abstract method cannot be instantiated at all, so it
    is never an answer. A class that merely *declares* itself a category can be
    instantiated, and whether a run should accept one is a judgement: refusing them is
    what makes a run compose ``Stool`` instead of answering ``Furniture``, and it costs
    every body whose answer it refuses.

    Read in one place, so that what a model is told about a class and what the annotate
    step will accept for one cannot drift apart.

    :param annotation_class: The class to judge.
    :param categories_are_answers: Whether a class declaring itself a category may still
        be given to an object.
    :return: Whether it is no answer about an object.
    """
    if inspect.isabstract(annotation_class):
        return True
    return not categories_are_answers and declares_itself_a_category(annotation_class)


def _summary_of(annotation_class: Type) -> Optional[str]:
    """
    A dataclass without a docstring is given one listing its signature, which says
    nothing a model needs and costs more than everything else in the node, so it counts
    as having none.

    :param annotation_class: The class to describe.
    :return: The first sentence of its docstring, or None when it has none of its own.
    """
    documentation = annotation_class.__dict__.get("__doc__")
    if not documentation or documentation.startswith(f"{annotation_class.__name__}("):
        return None
    return " ".join(documentation.split()).split(". ")[0].rstrip(".") or None


# %% finding the classes there are


ONTOLOGY_MODULES = (
    "semantic_digital_twin.semantic_annotations.mixins",
    "semantic_digital_twin.semantic_annotations.semantic_annotations",
)
"""
The modules the taxonomy is declared in, and the only place a class being *the ontology's*
can be told from a class merely deriving from its root.
"""


def load_annotation_modules() -> None:
    """
    Import the modules declaring the taxonomy.

    The hierarchy is discovered through ``__subclasses__``, which only ever reports
    classes Python has already imported, so a taxonomy exported before these modules are
    loaded comes back empty rather than wrong -- which is worse, because it looks like a
    taxonomy.

    Classes generated at run time are picked up as well, once whoever generated them has
    imported them.
    """
    for name in ONTOLOGY_MODULES:
        importlib.import_module(name)


def _walk_subclasses(root_class: Type) -> Iterator[Type]:
    """
    :param root_class: The class to walk below.
    :return: Every class deriving from it, each once, in breadth-first order.
    """
    seen = {root_class}
    frontier = [root_class]
    while frontier:
        current = frontier.pop(0)
        for subclass in current.__subclasses__():
            if subclass in seen:
                continue
            seen.add(subclass)
            frontier.append(subclass)
            yield subclass


def annotation_classes(root_class: Type) -> Dict[str, Type]:
    """
    :param root_class: The root of the hierarchy, normally ``SemanticAnnotation``.
    :return: Every class below it, by name, so a name read back from a file or a model
        can be turned into the class it stands for.
    """
    load_annotation_modules()
    return {
        annotation_class.__name__: annotation_class
        for annotation_class in _walk_subclasses(root_class)
    }


def declared_annotation_classes(root_class: Type) -> Dict[str, Type]:
    """
    The classes the ontology itself declares, as against every class below the root.

    :func:`annotation_classes` reports whatever this process has imported or composed, and
    :func:`compose_class` registers a proposed class below the root for the rest of the
    process. Asked which classes the ontology *has*, that answer says yes to a class this
    very process invented a moment ago.

    Walked rather than filtered out of :func:`annotation_classes`, which is keyed by name
    and so keeps only one class per name. A composed class deriving from a deep one is
    reached after the class it shadows, so filtering afterwards drops the ontology's own
    class instead of the composed one.

    :param root_class: The root of the hierarchy, normally ``SemanticAnnotation``.
    :return: Those of its subclasses declared in the ontology's own modules, by name.
    """
    load_annotation_modules()
    return {
        annotation_class.__name__: annotation_class
        for annotation_class in _walk_subclasses(root_class)
        if annotation_class.__module__ in ONTOLOGY_MODULES
    }


# %% building and writing the taxonomy


def build_taxonomy(
    root_class: Type,
    include_summaries: bool = False,
    categories_are_answers: bool = False,
) -> Dict[str, Any]:
    """
    Describe the annotation taxonomy below a root class.

    Classes are reported flat, each naming its own bases, because the hierarchy is not a
    tree: a cabinet is furniture *and* a thing with doors *and* a thing with drawers, and
    it is the second and third of those that say what it can hold.

    :param root_class: The root of the hierarchy, normally ``SemanticAnnotation``.
    :param include_summaries: Whether to describe each class by its docstring. Off by
        default: only some classes carry one, so including them would tell a model more
        about the documented classes than about the others for no reason but that
        someone happened to write a sentence.
    :return: The taxonomy as JSON-ready data.
    """
    load_annotation_modules()
    annotation_classes = list(_walk_subclasses(root_class))
    known = {annotation_class.__name__ for annotation_class in annotation_classes} | {
        root_class.__name__
    }

    classes = []
    mixins = []
    for annotation_class in annotation_classes:
        relations = relations_of(annotation_class)
        node: Dict[str, Any] = {
            "name": annotation_class.__name__,
            "bases": [
                base.__name__
                for base in annotation_class.__bases__
                if base.__name__ in known
            ],
        }
        if names_a_category(annotation_class, categories_are_answers):
            # A category is not something a room holds, and one that could be instantiated
            # is worse than one that could not: offering it as an answer is offering a
            # coarser answer than the ontology can carry, which a run then asserts.
            node["abstract"] = True
        if annotation_class.__module__.endswith(MIXIN_MODULE_SUFFIX):
            # A mixin is a base to build with, not a thing standing in a room. Saying so
            # is the difference between a model deriving a ceiling from HasRootBody and
            # answering that a ceiling *is* one.
            node["mixin"] = True
        summary = _summary_of(annotation_class) if include_summaries else None
        if summary:
            node["summary"] = summary
        if relations:
            node["relations"] = [relation.to_json() for relation in relations]
        classes.append(node)

        # A mixin is worth offering as a building block only for the relations it
        # introduces; the ones it inherits come with its own bases anyway.
        own_fields = _own_field_names(annotation_class)
        introduced = [
            relation for relation in relations if relation.field_name in own_fields
        ]
        if introduced and annotation_class.__module__.endswith(MIXIN_MODULE_SUFFIX):
            mixins.append(
                {
                    "name": annotation_class.__name__,
                    "introduces": [relation.to_json() for relation in introduced],
                }
            )

    return {
        "root_name": root_class.__name__,
        "note": (
            "Relations say what parts a class can hold. "
            "A class marked 'abstract' "
            "cannot be given to an object: it is a category, so name one of its "
            "subclasses, or compose a new subclass of it where none of them fits. A "
            "class marked 'mixin' exists "
            "to be built with rather than to name something in a room. A new class is "
            "composed by naming a superclass and any of the mixins below."
        ),
        "classes": classes,
        "part_whole_mixins": mixins,
    }


def export_taxonomy(
    root_class: Type,
    output_path: Path,
    include_summaries: bool = False,
    categories_are_answers: bool = False,
) -> Dict[str, Any]:
    """
    Write the taxonomy of a hierarchy to a JSON file.

    :param root_class: The root of the hierarchy.
    :param output_path: Where to write it.
    :param include_summaries: See :func:`build_taxonomy`.
    :param categories_are_answers: See :func:`names_a_category`.
    :return: The taxonomy that was written.
    """
    taxonomy = build_taxonomy(
        root_class,
        include_summaries=include_summaries,
        categories_are_answers=categories_are_answers,
    )
    Path(output_path).write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    return taxonomy


# %% composing a class from a proposal


def compose_class(name: str, superclass: Type, mixins: Sequence[Type] = ()) -> Type:
    """
    Build the class a proposal names, so the ontology can be asked about it.

    A label with no class in the taxonomy is answered by naming a superclass and the
    mixins to compose it from, and it is those that decide what the class admits: a
    ``KitchenIsland(Furniture)`` can hold nothing, while a
    ``KitchenIsland(Furniture, HasDrawers, HasDoors)`` can hold both drawers and doors.
    Composing the class here answers that before anything is generated or persisted.

    ..note:: The result is a real subclass of its bases, so ``__subclasses__`` reports it
        and :func:`build_taxonomy` includes it from here on. Export the taxonomy before
        composing anything that is only a proposal.

    :param name: The name the proposal gives the class.
    :param superclass: The class it derives from.
    :param mixins: The mixins contributing relations to it.
    :return: The composed dataclass.
    :raises TypeError: If the bases cannot be combined into a class, which is itself an
        answer about the proposal.
    """
    composed = type(name, in_base_order([superclass, *mixins]), {"__annotations__": {}})
    return dataclass(composed)


def in_base_order(bases: Sequence[Type]) -> tuple:
    """
    Order bases so that Python can combine them.

    A class may not name a base before one of that base's own subclasses, and a proposal
    naming ``HasRootBody`` with ``IsStorageSpace`` -- which already derives from it --
    names them the wrong way round. Which of two classes derives from the other is not
    something the proposal decides, so it is settled here rather than sent back as a
    problem with the answer.

    :param bases: The classes to derive from, as they were named.
    :return: The same classes, each once, every one before any class it derives from.
    """
    remaining = list(dict.fromkeys(bases))
    ordered = []
    while remaining:
        most_derived = next(
            candidate
            for candidate in remaining
            if not any(
                other is not candidate and issubclass(other, candidate)
                for other in remaining
            )
        )
        remaining.remove(most_derived)
        ordered.append(most_derived)
    return tuple(ordered)


def admissible_mounts(
    one_class: Type, other_class: Type
) -> List[Tuple[Type, SemanticRelation]]:
    """
    Report every way one of two classes could hold the other as a part, in either
    direction. What comes back is what the world would accept, not what is the case.

    :param one_class: One of the annotation classes.
    :param other_class: The other annotation class.
    :return: Each admissible mount, as the class that would hold and the relation it
        would be held in.
    """
    return [
        (whole, relation)
        for whole, part in ((one_class, other_class), (other_class, one_class))
        for relation in relations_of(whole)
        if isinstance(part, type) and issubclass(part, relation.target_class)
    ]
