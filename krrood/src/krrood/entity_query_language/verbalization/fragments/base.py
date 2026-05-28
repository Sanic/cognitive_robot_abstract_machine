"""
Fragment tree data model for verbalized output.

The fragment hierarchy forms the output IR that renderers traverse:

* :class:`WordFragment` — plain text (no semantic role).
* :class:`RoleFragment` — text with a :class:`SemanticRole` (drives colour / hyperlinks).
* :class:`PhraseFragment` — inline sequence of fragments joined by a separator.
* :class:`BlockFragment` — named structural block with header + bullet items.

Joining utilities (:func:`join_with`, :func:`oxford_and`) produce
:class:`PhraseFragment` trees from lists of fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.utils import _ensure_plural

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute


@dataclass
class VerbFragment:
    """
    Abstract base for all verbalized output fragments.

    The fragment hierarchy forms a tree:

    * Leaf nodes: :class:`WordFragment`, :class:`RoleFragment`.
    * Inline composition: :class:`PhraseFragment`.
    * Block structure: :class:`BlockFragment`.

    Renderers traverse this tree to produce strings.
    """


@dataclass
class WordFragment(VerbFragment):
    """Plain neutral text with no semantic role: articles, connectives, punctuation."""

    text: str
    """The raw text string (e.g. ``"the"``, ``"and"``, ``","``)."""


@dataclass
class RoleFragment(VerbFragment):
    """
    Text carrying a :class:`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole`
    — drives colour markup and optional source hyperlinking.
    """

    text: str
    """Display text (e.g. ``"Robot"``, ``"is greater than"``)."""

    role: SemanticRole
    """Semantic role determining the colour applied by the formatter."""

    source_ref: Optional[SourceRef] = None
    """Optional reference to the Python class or attribute this fragment represents;
    used by
    :class:`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver`
    to build hyperlinks."""

    @classmethod
    def for_variable(cls, label: str, expression) -> RoleFragment:
        """
        Build a fragment for a
        :class:`~krrood.entity_query_language.core.variable.Variable`,
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`,
        or :class:`~krrood.entity_query_language.query.query.Entity`, linked to its type.

        :param label: Display text (type name or disambiguated label).
        :type label: str
        :param expression: Expression whose ``_type_`` attribute supplies the source reference.
        :returns: :class:`RoleFragment` with :attr:`~SemanticRole.VARIABLE` role.
        :rtype: RoleFragment
        """
        return cls(
            text=label,
            role=SemanticRole.VARIABLE,
            source_ref=SourceRef.for_type(getattr(expression, "_type_", None)),
        )

    @classmethod
    def for_attribute(cls, owner, attribute_name: str, plural: bool = False) -> RoleFragment:
        """
        Build a fragment for an attribute access, linked to its owner class.

        :param owner: Owner class of the attribute (used for source linking).
        :param attribute_name: Canonical attribute name on *owner*.
        :type attribute_name: str
        :param plural: Whether the attribute is pluralized in the display text.
        :type plural: bool
        :returns: :class:`RoleFragment` with :attr:`~SemanticRole.ATTRIBUTE` role.
        :rtype: RoleFragment
        """
        label = attribute_name if not plural else _ensure_plural(attribute_name)
        return cls(
            text=label,
            role=SemanticRole.ATTRIBUTE,
            source_ref=SourceRef.for_attribute(owner, attribute_name),
        )

    @classmethod
    def for_operator(cls, label: str) -> RoleFragment:
        """
        Build a fragment for an operator or copula (no source link).

        :param label: Display text (e.g. ``"is"``, ``"not"``, ``"greater than"``).
        :type label: str
        :returns: :class:`RoleFragment` with :attr:`~SemanticRole.OPERATOR` role.
        :rtype: RoleFragment
        """
        return cls(text=label, role=SemanticRole.OPERATOR)


@dataclass
class PhraseFragment(VerbFragment):
    """An inline sequence of fragments joined by a separator."""

    parts: list[VerbFragment]
    """Ordered list of child fragments."""

    separator: str = " "
    """String inserted between adjacent parts."""


@dataclass
class BlockFragment(VerbFragment):
    """
    A named structural block with an optional header and a list of sub-items.

    * :class:`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer`
      flattens header + items into a single comma-separated prose string.
    * :class:`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer`
      renders the header on one line, then each item as a bullet at the next indent level.
    """

    header: Optional[VerbFragment]
    """Optional lead fragment (e.g. ``"Find Robot"`` or ``"If"``)."""

    items: list[VerbFragment] = field(default_factory=list)
    """Ordered list of sub-item fragments."""


# ── Fragment joining utilities ─────────────────────────────────────────────────


def join_with(parts: list[VerbFragment], separator: VerbFragment) -> VerbFragment:
    """
    Interleave *parts* with *separator* between each adjacent pair.

    :param parts: Fragments to join.
    :type parts: list[VerbFragment]
    :param separator: Separator fragment inserted between adjacent items.
    :type separator: VerbFragment
    :returns: A single fragment (or the sole item when ``len(parts) == 1``).
    :rtype: VerbFragment
    """
    if not parts:
        return WordFragment(text="")
    if len(parts) == 1:
        return parts[0]
    result: list[VerbFragment] = []
    for i, fragment in enumerate(parts):
        result.append(fragment)
        if i < len(parts) - 1:
            result.append(separator)
    return PhraseFragment(parts=result, separator="")


def oxford_and(parts: list[VerbFragment], conjunction: VerbFragment) -> VerbFragment:
    """
    Join *parts* with Oxford-comma style: ``f1, f2, conj f3``.

    :param parts: Fragments to join.
    :type parts: list[VerbFragment]
    :param conjunction: Conjunction fragment (e.g. *"and"*, *"or"*).
    :type conjunction: VerbFragment
    :returns: A single fragment representing the joined sequence.
    :rtype: VerbFragment
    """
    if not parts:
        return WordFragment(text="")
    if len(parts) == 1:
        return parts[0]
    head = parts[:-1]
    tail = parts[-1]
    result: list[VerbFragment] = []
    for fragment in head:
        result.append(fragment)
        result.append(WordFragment(text=", "))
    result.append(PhraseFragment(parts=[conjunction, tail]))
    return PhraseFragment(parts=result, separator="")
