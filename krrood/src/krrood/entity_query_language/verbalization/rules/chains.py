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

from typing import Optional, TYPE_CHECKING

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
from krrood.entity_query_language.verbalization.utils import _ordinal
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
    expr: MappedVariable,
    ctx: VerbalizationContext,
    verbalizer: EQLVerbalizer,
    *,
    negated: bool = False,
) -> VerbFragment:
    """
    Render a :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
    chain.  Boolean terminal attributes use the predicative *"<nav> is [not] <attr>"*
    form; all others use the possessive *"the attr of the Root"* path.

    :param expr: Root node of a MappedVariable chain.
    :param ctx: Shared verbalization state.
    :param verbalizer: Verbalizer for recursive sub-expression rendering.
    :param negated: Use the negated copula for a boolean terminal attribute.
    :returns: Fragment for the chain.
    """
    chain, leaf = walk_chain(expr)
    terminal = chain[-1]
    if isinstance(terminal, Attribute) and terminal._type_ is bool:
        return _bool_attribute_chain(chain, leaf, ctx, verbalizer, negated)
    root_frag = _chain_root_frag(leaf, ctx, verbalizer)
    return _render_path(build_path_parts(chain), root_frag)


def verbalize_possessive_chain(
    expr: MappedVariable, ctx: VerbalizationContext, pronoun_frag: VerbFragment
) -> VerbFragment:
    """
    Render a chain rooted at the current coreference subject, replacing the trailing
    *"of the Root"* with a leading possessive pronoun: *"the amount of its
    amount_details"* / *"its booking_date"*.
    """
    chain, _root = walk_chain(expr)
    return _render_possessive_path(build_path_parts(chain), pronoun_frag)


def _render_path(
    parts: list[tuple[str, Optional[SourceRef]]], root_frag: VerbFragment
) -> VerbFragment:
    if not parts:
        return root_frag
    reversed_parts = list(reversed(parts))
    first_name, first_ref = reversed_parts[0]
    frag_parts: list[VerbFragment] = [
        Articles.THE.as_fragment(),
        RoleFragment(text=first_name, role=SemanticRole.ATTRIBUTE, source_ref=first_ref),
    ]
    for attr_name, attr_ref in reversed_parts[1:]:
        frag_parts.extend(
            [
                Prepositions.OF_THE.as_fragment(),
                RoleFragment(
                    text=attr_name, role=SemanticRole.ATTRIBUTE, source_ref=attr_ref
                ),
            ]
        )
    frag_parts.extend([Prepositions.OF.as_fragment(), root_frag])
    return PhraseFragment(parts=frag_parts)


def _render_possessive_path(
    parts: list[tuple[str, Optional[SourceRef]]], pronoun_frag: VerbFragment
) -> VerbFragment:
    if not parts:
        return pronoun_frag
    reversed_parts = list(reversed(parts))
    last = len(reversed_parts) - 1
    frag_parts: list[VerbFragment] = []
    for idx, (attr_name, attr_ref) in enumerate(reversed_parts):
        attr_frag = RoleFragment(
            text=attr_name, role=SemanticRole.ATTRIBUTE, source_ref=attr_ref
        )
        if idx == 0 and idx != last:
            frag_parts.extend([Articles.THE.as_fragment(), attr_frag])
        elif idx == 0:  # single attribute → "its booking_date"
            frag_parts.extend([pronoun_frag, attr_frag])
        elif idx == last:  # attribute adjacent to the (elided) root → "of its amount_details"
            frag_parts.extend([Prepositions.OF.as_fragment(), pronoun_frag, attr_frag])
        else:
            frag_parts.extend([Prepositions.OF_THE.as_fragment(), attr_frag])
    return PhraseFragment(parts=frag_parts)


def _chain_root_frag(leaf, ctx: VerbalizationContext, verbalizer: EQLVerbalizer) -> VerbFragment:
    """Noun phrase for the root of a chain; unwraps ResultQuantifier wrappers."""
    inner = leaf
    while isinstance(inner, ResultQuantifier):
        inner = inner._child_
    if isinstance(inner, Entity):
        return as_inline_noun(inner, ctx, verbalizer)
    return verbalizer.build(leaf, ctx)


def _navigation_chain(nav_chain: list, root_frag: VerbFragment) -> VerbFragment:
    """Fragment for the navigation path up to (but not including) the terminal attribute."""
    if not nav_chain:
        return root_frag
    if isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
        ordinal = _ordinal(nav_chain[-1]._key_)
        pre_frag = _render_path(build_path_parts(nav_chain[:-1]), root_frag)
        return phrase(
            Articles.THE.as_fragment(),
            word(ordinal),
            Prepositions.OF.as_fragment(),
            pre_frag,
        )
    return _render_path(build_path_parts(nav_chain), root_frag)


def _bool_attribute_chain(
    chain: list, leaf, ctx: VerbalizationContext, verbalizer: EQLVerbalizer, negated: bool
) -> VerbFragment:
    """Produces '<nav-path> is [not] <attr>' for boolean terminal attributes."""
    root_frag = _chain_root_frag(leaf, ctx, verbalizer)
    nav_frag = _navigation_chain(chain[:-1], root_frag)
    copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
    terminal = chain[-1]
    return phrase(
        nav_frag,
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
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for any :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`."""
        return isinstance(expr, MappedVariable)

    @classmethod
    def transform(
        cls,
        expr: MappedVariable,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the chain as a possessive or predicative fragment."""
        return verbalize_chain(expr, ctx, verbalizer)


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
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for a non-bool chain rooted at the current pronoun-eligible subject."""
        if not isinstance(expr, MappedVariable) or isinstance(expr, FlatVariable):
            return False
        chain, root = walk_chain(expr)
        if not chain:
            return False
        terminal = chain[-1]
        if isinstance(terminal, Attribute) and terminal._type_ is bool:
            return False
        return ctx.pronoun_for(root) is not None

    @classmethod
    def transform(
        cls,
        expr: MappedVariable,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the chain with a leading *"its"* possessive pronoun."""
        _chain, root = walk_chain(expr)
        return verbalize_possessive_chain(expr, ctx, ctx.pronoun_for(root))


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
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.core.mapped_variable.FlatVariable`."""
        return isinstance(expr, FlatVariable)

    @classmethod
    def transform(
        cls,
        expr: FlatVariable,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Unwrap to the child expression."""
        return verbalizer.build(expr._child_, ctx)
