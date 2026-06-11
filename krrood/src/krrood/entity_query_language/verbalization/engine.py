"""
The verbalization engine — a single catamorphism over the EQL expression tree.

:func:`fold` is the *only* place the EQL tree is recursed: it dispatches a node to
the most-specific :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule`
(via :func:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.select`)
and applies its ``build``, handing the rule a :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx`
whose ``child`` re-enters the fold.  Rules therefore never recurse by hand.

This is the F-algebra / catamorphism over the source (EQL) algebra; the grammar
is the algebra (Meijer, Fokkinga & Paterson 1991, "Functional Programming with
Bananas, Lenses, Envelopes and Barbed Wire"; Bird & de Moor 1997, "Algebra of
Programming").  Compare the fold over the *output* tree,
:func:`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Optional, Sequence

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.phrase_rule import (
    Ctx,
    PhraseRule,
    select,
)
from krrood.entity_query_language.verbalization.grammar.english import RULES

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext


class UnverbalizableExpressionError(TypeError):
    """No grammar rule covers an EQL construct.

    Raised by :func:`fold` instead of silently degrading the node to its class name — a coverage
    gap is a bug, not bad English.  Add a
    :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule` for the
    construct (in :mod:`~krrood.entity_query_language.verbalization.grammar.english`).
    """


def fold(
    node,
    context: "VerbalizationContext",
    rules: Optional[Sequence[PhraseRule]] = None,
    number: Number = Number.SINGULAR,
) -> VerbFragment:
    """
    Verbalize *node* by dispatching to the matching grammar rule and recursing.

    Order of resolution:

    1. **Binding-override short-circuit** — if ``node._id_`` has a pre-built
       substitute in :attr:`BindingScope.binding_overrides`, return it before any
       dispatch (used for InstantiatedVariable field references).
    2. :func:`select` the most-specific rule and apply its ``build`` with a fresh
       :class:`Ctx` whose ``child`` re-enters :func:`fold`.
    3. **No rule** → raise :class:`UnverbalizableExpressionError` (a coverage gap is a bug,
       not silent degradation to the class name).

    :param node: Any EQL expression.
    :param context: The verbalization context (services + render config).
    :param rules: Grammar to dispatch over; defaults to ``RULES``.
    :return: The fragment for *node*.
    :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
    :raises UnverbalizableExpressionError: when no grammar rule covers *node*.
    """
    rules = RULES if rules is None else rules

    node_id = getattr(node, "_id_", None)
    if node_id is not None:
        override = context.binding.binding_overrides.get(node_id)
        if override is not None:
            return override

    ctx = Ctx(
        child=lambda child_node, number=Number.SINGULAR: fold(
            child_node, context, rules, number=number
        ),
        context=context,
        number=number,
    )

    rule = select(node, rules, ctx)
    if rule is None:
        raise UnverbalizableExpressionError(
            f"No verbalization rule for {type(node).__name__!r} "
            f"(name={getattr(node, '_name_', None)!r}); add a PhraseRule in grammar/english.py."
        )
    return rule.build(node, ctx)
