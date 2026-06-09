"""
Condition **verbalizer** — the single owner of every surface form a condition can take.

A Comparator/condition is said differently depending on where it sits: a standalone
*predicate* (*"x is greater than 5"*), a post-nominal *attribute modifier* on a subject
(the bare *"<attr> op <value>"* that a *"whose …"* envelope wraps), a *range* modifier
(*"<attr> is between lo and hi"*), or the inference *whose-attribute* body (*"<attr> is
<value>"* agreeing in number).  Previously each of these lived in a different consumer
(the comparator rule, the restriction rules, the inference assembler); they are
co-located here so one component owns *how a condition is verbalized* and the consumers
merely ask for a form.

Realisation-only (``planner = None``), holding the per-node
:class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx`; recursion is
via ``self.ctx.child``.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import Any

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.fragments.factory import phrase
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.operator_phrase import (
    comparator_operator,
)


class ConditionVerbalizer(Assembler[Any, None]):
    """Render a condition in a requested surface form (predicate / modifier / …)."""

    def realize(self, node, plan: None = None) -> VerbFragment:
        """Default form — a standalone predicate (the :class:`Assembler` entry point)."""
        return self.predicate(node)

    def predicate(self, comparator, *, negated: bool = False) -> VerbFragment:
        """*"<left> <operator> <right>"* — the standalone comparator form."""
        return phrase(
            self.ctx.child(comparator.left),
            comparator_operator(comparator, self.ctx.context, negated=negated),
            self.ctx.child(comparator.right),
        )
