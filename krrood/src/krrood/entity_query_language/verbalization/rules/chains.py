"""
Verbalization of :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
chains (Attribute / Index / Call / FlatVariable).

This module is the single source of truth for chain rendering: the rules below own
both the *decision* (which form) and the *rendering* (the fragment tree).  Boolean
terminal attributes render predicatively (*"<nav> is [not] <attr>"*); everything else
renders as a possessive path (*"the attr of the Root"*), optionally pronominalised
(*"its booking_date"*) when the chain root is the current coreference subject.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Optional

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    chain_root,
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization.rules.query import as_inline_noun
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Prepositions,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


# ── Chain rendering helpers ─────────────────────────────────────────────────────


def verbalize_chain(
    expression: MappedVariable,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
    *,
    negated: bool = False,
) -> VerbFragment:
    """
    Render a :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
    chain.  Boolean terminal attributes use the predicative *"<nav> is [not] <attr>"*
    form; all others use the possessive *"the attr of the Root"* path.

    :param expression: Root node of a MappedVariable chain.
    :param context: Shared verbalization state.
    :param verbalizer: Verbalizer for recursive sub-expression rendering.
    :param negated: Use the negated copula for a boolean terminal attribute.
    :return: Fragment for the chain.
    """
    chain, leaf = walk_chain(expression)
    terminal = chain[-1]
    if isinstance(terminal, Attribute) and terminal._type_ is bool:
        return _verbalize_bool_predicative(chain, leaf, context, verbalizer, negated)
    root_fragment = _chain_root_frag(leaf, context, verbalizer)
    return _verbalize_possessive_path(build_path_parts(chain), root_fragment)


def verbalize_possessive_chain(
    expression: MappedVariable,
    context: VerbalizationContext,
    pronoun_fragment: VerbFragment,
) -> VerbFragment:
    """
    Render a chain rooted at the current coreference subject, replacing the trailing
    *"of the Root"* with a leading possessive pronoun: *"the amount of its
    amount_details"* / *"its booking_date"*.
    """
    chain, _root = walk_chain(expression)
    return _verbalize_pronominal_path(build_path_parts(chain), pronoun_fragment)


def _verbalize_possessive_path(
    parts: list[tuple[str, Optional[SourceRef]]], root_fragment: VerbFragment
) -> VerbFragment:
    """Build a possessive-path phrase fragment: ``"the attr of the Root"``.

    Iterates *parts* in reverse (innermost attribute first) and assembles
    ``"the <inner> of the <outer> of <root>"``.

    :param parts: Ordered pairs of ``(display_name, source_ref)`` from
        :func:`~krrood.entity_query_language.verbalization.chain_utils.build_path_parts`,
        outermost attribute first.
    :param root_fragment: Noun-phrase fragment for the chain root.
    :return: :class:`PhraseFragment` representing the full possessive chain.
    """
    if not parts:
        return root_fragment
    reversed_parts = list(reversed(parts))
    first_name, first_ref = reversed_parts[0]
    fragment_parts: list[VerbFragment] = [
        Articles.THE.as_fragment(),
        _attr_fragment(first_name, first_ref),
    ]
    for attribute_name, attribute_reference in reversed_parts[1:]:
        fragment_parts.extend(
            [
                Prepositions.OF_THE.as_fragment(),
                _attr_fragment(attribute_name, attribute_reference),
            ]
        )
    fragment_parts.extend([Prepositions.OF.as_fragment(), root_fragment])
    return PhraseFragment(parts=fragment_parts)


def _verbalize_pronominal_path(
    parts: list[tuple[str, Optional[SourceRef]]], pronoun_fragment: VerbFragment
) -> VerbFragment:
    """Build a pronominal chain: ``"its attr"`` or ``"the attr of its foo"``.

    Replaces the trailing ``"of the Root"`` with a leading possessive pronoun.
    The pronoun is placed at the position closest to the (elided) root:

    * Single hop: ``"its booking_date"``
    * Multi-hop: ``"the amount of its amount_details"``

    :param parts: Ordered pairs from :func:`build_path_parts`, outermost first.
    :param pronoun_fragment: Pronoun fragment (typically ``Pronouns.ITS.as_fragment()``).
    :return: :class:`PhraseFragment` with the pronoun in the correct position.
    """
    if not parts:
        return pronoun_fragment
    reversed_parts = list(reversed(parts))
    last = len(reversed_parts) - 1
    fragment_parts: list[VerbFragment] = []
    for index, (attribute_name, attribute_reference) in enumerate(reversed_parts):
        attribute_fragment = _attr_fragment(attribute_name, attribute_reference)
        if index == 0 and index != last:
            fragment_parts.extend([Articles.THE.as_fragment(), attribute_fragment])
        elif index == 0:  # single attribute → "its booking_date"
            fragment_parts.extend([pronoun_fragment, attribute_fragment])
        elif (
            index == last
        ):  # attribute adjacent to the (elided) root → "of its amount_details"
            fragment_parts.extend(
                [Prepositions.OF.as_fragment(), pronoun_fragment, attribute_fragment]
            )
        else:
            fragment_parts.extend(
                [Prepositions.OF_THE.as_fragment(), attribute_fragment]
            )
    return PhraseFragment(parts=fragment_parts)


def _attr_fragment(name: str, source_ref: Optional[SourceRef]) -> RoleFragment:
    """Return a :class:`RoleFragment` with :attr:`~SemanticRole.ATTRIBUTE` role.

    :param name: Display name of the attribute (from :func:`build_path_parts`).
    :param source_ref: Optional source reference for hyperlink resolution.
    :return: Role-tagged fragment for the attribute.
    """
    return RoleFragment(text=name, role=SemanticRole.ATTRIBUTE, source_ref=source_ref)


def _chain_root_frag(
    leaf: object, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Build the noun-phrase fragment for the root of a MappedVariable chain.

    Unwraps :class:`~krrood.entity_query_language.query.quantifiers.ResultQuantifier`
    wrappers, then delegates to :func:`as_inline_noun` for
    :class:`~krrood.entity_query_language.query.query.Entity` roots, or to
    ``verbalizer.build`` for all others.

    :param leaf: The chain root expression (returned by :func:`walk_chain`).
    :param context: Shared verbalization state.
    :param verbalizer: Top-level verbalizer for recursive sub-expression building.
    :return: Noun-phrase fragment for the root.
    """
    inner = leaf
    while isinstance(inner, ResultQuantifier):
        inner = inner._child_
    if isinstance(inner, Entity):
        return as_inline_noun(inner, context, verbalizer)
    return verbalizer.build(leaf, context)


def _verbalize_bool_predicative(
    chain: list[MappedVariable],
    leaf: object,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
    negated: bool,
) -> VerbFragment:
    """Build a predicative boolean attribute phrase: ``"<nav> is [not] <attr>"``.

    For chains ending in an int-typed :class:`Index`, produces ordinal navigation
    (e.g. ``"the first tasks of the Robot is completed"``).

    :param chain: Full chain from :func:`walk_chain` (outermost-first, including terminal).
    :param leaf: Chain root expression.
    :param context: Shared verbalization state.
    :param verbalizer: Top-level verbalizer for recursive sub-expression building.
    :param negated: If ``True``, use ``"is not"`` copula.
    :return: :class:`PhraseFragment` for the predicative form.
    """
    root_fragment = _chain_root_frag(leaf, context, verbalizer)
    nav_chain = chain[:-1]

    if not nav_chain:
        nav_fragment = root_fragment
    elif isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
        ordinal = morphology.ordinal(nav_chain[-1]._key_)
        pre_frag = _verbalize_possessive_path(
            build_path_parts(nav_chain[:-1]), root_fragment
        )
        nav_fragment = phrase(
            Articles.THE.as_fragment(),
            word(ordinal),
            Prepositions.OF.as_fragment(),
            pre_frag,
        )
    else:
        nav_fragment = _verbalize_possessive_path(
            build_path_parts(nav_chain), root_fragment
        )

    copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
    terminal = chain[-1]
    return phrase(
        nav_fragment,
        copula,
        RoleFragment.for_attribute(terminal._owner_class_, terminal._attribute_name_),
    )


# ── Rules ───────────────────────────────────────────────────────────────────────


class MappedVariableRule(VerbalizationRule):
    """
    Verbalizes all :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
    chains (Attribute, Index, Call) via :func:`verbalize_chain`.

    :class:`FlatVariableRule` and :class:`PronominalChainRule` are more-specific
    subclasses and take priority via MRO-depth sorting.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for any :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`."""
        return isinstance(expression, MappedVariable)

    @classmethod
    def transform(
        cls,
        expression: MappedVariable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the chain as a possessive or predicative fragment."""
        return verbalize_chain(expression, context, verbalizer)


class PronominalChainRule(MappedVariableRule):
    """
    Verbalizes a chain whose root is the current coreference subject using a possessive
    pronoun: *"the amount of its amount_details"* / *"its booking_date"*.

    Precondition (declarative): the chain has a non-boolean terminal attribute and its
    root is the unambiguous, already-introduced subject — i.e.
    :meth:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.pronoun_for`
    returns a fragment.  Takes priority over :class:`MappedVariableRule` via MRO depth;
    when the precondition fails it falls through to the plain possessive-path form.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for a non-bool chain rooted at the current pronoun-eligible subject."""
        if not isinstance(expression, MappedVariable) or isinstance(
            expression, FlatVariable
        ):
            return False
        chain, root = walk_chain(expression)
        if not chain:
            return False
        terminal = chain[-1]
        if isinstance(terminal, Attribute) and terminal._type_ is bool:
            return False
        return context.pronoun_for(root) is not None

    @classmethod
    def transform(
        cls,
        expression: MappedVariable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the chain with a leading *"its"* possessive pronoun."""
        root = chain_root(expression)
        return verbalize_possessive_chain(
            expression, context, context.pronoun_for(root)
        )


class FlatVariableRule(MappedVariableRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable`
    by unwrapping to its child expression.

    :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable` is a transparent
    wrapper added during SetOf expansion.  It is a subclass of
    :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable` and takes
    priority over :class:`MappedVariableRule` via MRO-depth sorting.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable`."""
        return isinstance(expression, FlatVariable)

    @classmethod
    def transform(
        cls,
        expression: FlatVariable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Unwrap to the child expression."""
        return verbalizer.build(expression._child_, context)
