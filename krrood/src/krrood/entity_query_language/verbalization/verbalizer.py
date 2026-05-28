"""
EQL verbalizer — coordinator and one-shot convenience entry point.

:class:`EQLVerbalizer` dispatches an EQL expression tree to the rule engine and
returns a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
tree.  :func:`verbalize_expression` is the simplest entry point — it returns a plain
English string with no colour markup.

For coloured / hierarchical output use
:class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.context import VerbalizationContext
from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.rule_engine import RuleEngine
from krrood.entity_query_language.verbalization.rules.registry import ALL_RULES
from krrood.entity_query_language.verbalization.utils import _str


@dataclass
class EQLVerbalizer:
    """
    Coordinator that maps an EQL expression tree to a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree.

    Dispatches via a :class:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine` of
    :class:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule` classes.
    Each rule declares its guard in :meth:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.applies`
    and its rendering in :meth:`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule.transform`.
    More-specific subclasses are tried before their parents (MRO-depth priority).

    For simple plain-text output use :func:`verbalize_expression`.
    For coloured / formatted output build a
    :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`.
    """

    _engine: RuleEngine = field(init=False, repr=False)
    """Rule dispatcher; sorts rules by MRO depth before first call."""

    def __post_init__(self) -> None:
        self._engine = RuleEngine(ALL_RULES)

    def build(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> VerbFragment:
        """
        Translate *expression* into a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree.

        A fresh :class:`~krrood.entity_query_language.verbalization.context.VerbalizationContext`
        (with a pre-built disambiguation map) is created when *context* is ``None``.

        :param expression: Any EQL symbolic expression.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state; created automatically when omitted.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext or None
        :returns: Root of the fragment tree representing *expression* in natural language.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        if context is None:
            context = VerbalizationContext.from_expression(expression)
        return self._engine.build(expression, context, self)

    def verbalize(
        self,
        expression: SymbolicExpression,
        context: Optional[VerbalizationContext] = None,
    ) -> str:
        """
        Translate *expression* into a plain-text English string.

        Equivalent to ``_str(self.build(expression, context))`` — no colour markup.
        Prefer :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`
        when colour or hierarchical layout is needed.

        :param expression: Any EQL symbolic expression.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state; created automatically when omitted.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext or None
        :returns: Plain-text natural-language representation of *expression*.
        :rtype: str
        """
        return _str(self.build(expression, context))


_default_verbalizer = EQLVerbalizer()


def verbalize_expression(expression) -> str:
    """
    Verbalize any EQL expression into a human-readable English phrase (plain text).

    This is the simplest entry point: it uses a module-level singleton
    :class:`EQLVerbalizer` with no colour markup.  Call
    :class:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`
    directly for coloured or hierarchical output.

    :param expression: Any EQL expression or :class:`~krrood.entity_query_language.query.query.Query`.
    :returns: Plain-text natural-language string.
    :rtype: str
    """
    if isinstance(expression, Query):
        expression.build()
    return _default_verbalizer.verbalize(expression)
