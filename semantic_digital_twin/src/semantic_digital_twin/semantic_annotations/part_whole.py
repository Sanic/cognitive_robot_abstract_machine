"""
Metadata for the part-whole relation between semantic annotations.

This module holds only the vocabulary of the relation, so both the annotation mixins
that declare part-whole fields and the specification API that fills them can depend on
it without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import List, Optional, Type

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.patterns.field_metadata import FieldMetadata

# %% relation metadata


@dataclass
class IsPartWholeRelationship(FieldMetadata):
    """
    Marks a field as holding a structural *part* of its owner (the part-whole relation).

    The relation is signalled by the presence of an instance of this class in the
    field's ``metadata`` mapping (attach it with :meth:`~FieldMetadata.as_dict`), and
    the instance describes how mounting a part into that field affects the whole.
    """

    removes_part_geometry_from_whole: bool = False
    """
    Whether mounting a part into this field removes the part's volume from the whole's
    collision and visual geometry.

    This is a property of the relation rather than of the part: the same
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.EntryWay`
    cuts the wall it is an aperture of, but not the door whose passage it marks.
    """

    part_carries_whole: bool = False
    """
    Whether a part mounted into this field carries the whole: the whole hangs from the
    part, which takes the whole's place in the kinematic structure, rather than the part
    from the whole.

    This holds where what a thing is part of differs from what it is attached to, such
    as a telephone whose handset hangs from its cable.
    """


# %% relation queries


@dataclass
class PartWholeField:
    """
    One field through which an annotation class holds a structural part.
    """

    whole: Type
    """
    The annotation class holding the part.
    """

    field_name: str
    """
    The field the part is held in, as :meth:`PartWholeRelationship.add` names it.
    """

    part: Type
    """
    The type of part the field accepts.
    """

    removes_part_geometry_from_whole: bool
    """
    Whether mounting a part here cuts the part's volume out of the whole's geometry.
    """

    part_carries_whole: bool
    """
    Whether a part mounted here carries the whole, rather than hanging from it.
    """

    holds_many: bool
    """
    Whether the field holds several parts rather than a single one.
    """


def part_whole_fields(annotation_class: Type) -> List[PartWholeField]:
    """
    Report the structural parts an annotation class can hold.

    This reads the same metadata :meth:`PartWholeRelationship.add` routes by, so what it
    reports and what a mount accepts cannot drift apart.

    :param annotation_class: The annotation class to inspect.
    :return: One entry per field of that class carrying a part-whole relationship.
    """
    part_whole_relationship_fields = []
    for wrapped_field in WrappedClass(annotation_class).fields_with_metadata(
        IsPartWholeRelationship
    ):
        relationship = IsPartWholeRelationship.of_wrapped_field(wrapped_field)
        part_whole_relationship_fields.append(
            PartWholeField(
                whole=annotation_class,
                field_name=wrapped_field.field.name,
                part=wrapped_field.type_endpoint,
                removes_part_geometry_from_whole=relationship.removes_part_geometry_from_whole,
                part_carries_whole=relationship.part_carries_whole,
                holds_many=wrapped_field.is_many_to_many_relationship,
            )
        )
    return part_whole_relationship_fields


def field_holding(whole: Type, part: Type) -> Optional[PartWholeField]:
    """
    Report the one field through which a whole holds a part.

    This is the field :meth:`PartWholeRelationship.add` routes to when it is given no
    field to use, so a mount that was carried out without naming one can be told
    afterwards which field it went through.

    :param whole: The annotation class holding the part.
    :param part: The annotation class being held.
    :return: That field, or ``None`` where no field accepts the part or more than one
        does -- the case ``add`` refuses as ambiguous rather than choosing.
    """
    matching = fields_accepting(whole, part)
    if len(matching) != 1:
        return None
    return matching[0]


def fields_accepting(whole: Type, part: Type) -> List[PartWholeField]:
    """
    Report every field through which a whole could hold a part.

    :param whole: The annotation class holding the part.
    :param part: The annotation class being held.
    :return: The fields whose element type the part is, none where the whole cannot
        hold it and several where :meth:`PartWholeRelationship.add` would raise
        ``AmbiguousPart``.
    """
    return [
        part_whole_relationship_field
        for part_whole_relationship_field in part_whole_fields(whole)
        if isinstance(part_whole_relationship_field.part, type)
        and issubclass(part, part_whole_relationship_field.part)
    ]


def admissible_relations(one_class: Type, other_class: Type) -> List[PartWholeField]:
    """
    Report the part-whole relations two annotation classes may stand in, in either
    direction.

    An empty result means neither class can hold the other as a structural part, so an
    overlap between them is something other than a part-whole relation.

    ..note:: Several entries for the same direction mean the part matches more than one
        of the whole's fields, which is what makes :meth:`PartWholeRelationship.add`
        raise ``AmbiguousPart`` unless it is told which field to use.

    :param one_class: One of the annotation classes.
    :param other_class: The other annotation class.
    :return: Every admissible relation, each naming which class is the whole and through
        which field. The part is named as the field declares it, which may be a base of
        the class that was asked about.
    """
    return [
        part_whole_relationship_field
        for whole, part in ((one_class, other_class), (other_class, one_class))
        for part_whole_relationship_field in part_whole_fields(whole)
        if isinstance(part, type)
        and isinstance(part_whole_relationship_field.part, type)
        and issubclass(part, part_whole_relationship_field.part)
    ]
