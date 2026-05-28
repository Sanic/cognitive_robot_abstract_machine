"""
Subject restriction clauses — turning a subject's ``WHERE`` condition into a
post-nominal *"whose <attr> is …"* modifier plus a residual *"such that …"* clause.

The "is this conjunct a groupable subject predicate, and how does it render?" decision
is **declarative**: a small :class:`RestrictionRule` hierarchy, dispatched first-match-wins
by specificity (mirroring the main
:class:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine`).  A folded
conjunct that **no** rule matches becomes residual — groupability lives in the rule
preconditions, not in branching glue.

:class:`RestrictionClauseBuilder` is the thin orchestrator used by the
query-body assembly in :mod:`~krrood.entity_query_language.verbalization.rules.query`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    flatten_operands,
)
from krrood.entity_query_language.verbalization.chain_utils import chain_root, walk_chain
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.operator_phrase import comparator_operator
from krrood.entity_query_language.verbalization.range_fold import (
    build_between,
    fold_range_pairs,
    RangeFold,
)
from krrood.entity_query_language.verbalization.subquery import aggregation_source_root
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


# ── Chain helpers ──────────────────────────────────────────────────────────────


def _single_hop_attr(expr, subject_var):
    """Return the :class:`Attribute` node when *expr* is exactly ``subject_var.<attr>``, else ``None``."""
    if subject_var is None or not isinstance(expr, MappedVariable):
        return None
    chain, root = walk_chain(expr)
    if not (isinstance(root, Variable) and root._id_ == subject_var._id_):
        return None
    if len(chain) != 1 or not isinstance(chain[0], Attribute):
        return None
    return chain[0]


def _references(expr, subject_var) -> bool:
    """Return ``True`` when *expr* mentions *subject_var* (so it is not a clean RHS value)."""
    try:
        return any(
            getattr(v, "_id_", None) == subject_var._id_
            for v in expr._unique_variables_
        )
    except AttributeError:
        return chain_root(expr) is subject_var


# ── Restriction rule hierarchy ──────────────────────────────────────────────────


class RestrictionRule(ABC):
    """
    Declarative rule that recognises a folded conjunct as a groupable subject
    restriction and renders it as a bare post-nominal predicate (no root / pronoun;
    the enclosing *"whose"* supplies the subject).

    A conjunct matched by no rule is residual and stays in a *"such that …"* clause.
    """

    @classmethod
    @abstractmethod
    def applies(cls, item, subject_var, ctx: VerbalizationContext) -> bool:
        """Return ``True`` when *item* is a groupable restriction on *subject_var*."""

    @classmethod
    @abstractmethod
    def render(
        cls, item, subject_var, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """Render *item* as a bare predicate fragment (e.g. *"booking_date is between X and Y"*)."""


class RangeRestrictionRule(RestrictionRule):
    """Matches a :class:`RangeFold` on a single-hop subject attribute → *"<attr> is between lo and hi"*."""

    @classmethod
    def applies(cls, item, subject_var, ctx: VerbalizationContext) -> bool:
        return (
            isinstance(item, RangeFold)
            and _single_hop_attr(item.chain_expr, subject_var) is not None
        )

    @classmethod
    def render(
        cls, item, subject_var, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        attr = _single_hop_attr(item.chain_expr, subject_var)
        left = RoleFragment.for_attribute(attr._owner_class_, attr._attribute_name_)
        return build_between(
            left,
            verbalizer.build(item.lower_expression, ctx),
            verbalizer.build(item.upper_expression, ctx),
            compact=False,
        )


class AttributePredicateRestrictionRule(RestrictionRule):
    """
    Matches a :class:`~krrood.entity_query_language.operators.comparator.Comparator`
    that is a single-hop, non-boolean subject-attribute predicate whose RHS does not
    reference the subject → *"<attr> is greater than 100"* / *"<attr> is equal to <calc>"*.
    """

    @classmethod
    def applies(cls, item, subject_var, ctx: VerbalizationContext) -> bool:
        if not isinstance(item, Comparator):
            return False
        attr = _single_hop_attr(item.left, subject_var)
        if attr is None or attr._type_ is bool:
            return False
        return not _references(item.right, subject_var)

    @classmethod
    def render(
        cls, item, subject_var, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        attr = _single_hop_attr(item.left, subject_var)
        attr_frag = RoleFragment.for_attribute(
            attr._owner_class_, attr._attribute_name_
        )
        op_frag = comparator_operator(item, ctx, compact=False)
        return PhraseFragment(
            parts=[attr_frag, op_frag, verbalizer.build(item.right, ctx)]
        )


RESTRICTION_RULES: List[type] = [
    RangeRestrictionRule,
    AttributePredicateRestrictionRule,
]


# ── Restriction subject resolution ──────────────────────────────────────────────


class RestrictionSubjectRule(ABC):
    """
    Declarative rule resolving **which variable** a query's selection restricts, so the
    selection's ``WHERE`` clause can fold into a post-nominal *"whose …"* modifier on it.

    Dispatched first-match-wins by registry order (mirroring :class:`RestrictionRule`).  A
    selection matched by **no** rule has no groupable subject — its ``WHERE`` clause stays a
    full *"such that …"* clause.
    """

    @classmethod
    @abstractmethod
    def applies(cls, expr, selected_var, ctx: VerbalizationContext) -> bool:
        """Return ``True`` when this rule can name the restriction subject of *expr*."""

    @classmethod
    @abstractmethod
    def subject(cls, expr, selected_var, ctx: VerbalizationContext):
        """Return the :class:`~krrood.entity_query_language.core.variable.Variable` the ``WHERE`` restricts."""


class SelectedVariableSubjectRule(RestrictionSubjectRule):
    """The selection itself is a plain :class:`~krrood.entity_query_language.core.variable.Variable` → it is its own subject."""

    @classmethod
    def applies(cls, expr, selected_var, ctx: VerbalizationContext) -> bool:
        return isinstance(selected_var, Variable)

    @classmethod
    def subject(cls, expr, selected_var, ctx: VerbalizationContext):
        return selected_var


class AggregationSourceSubjectRule(RestrictionSubjectRule):
    """
    The selection is an aggregation over a single source variable's chain
    (e.g. ``max(t.amount_details.amount)`` or ``max(t, key=…)``).

    The aggregation selection renders its child chain root-last
    (*"the average of the amount of the amount_details of a BankTransaction"*), and
    :func:`~krrood.entity_query_language.verbalization.subquery.aggregation_source_root`
    returns exactly that chain root, so the selection ends with the source variable's noun
    and a *"whose …"* modifier attaches to it grammatically.  The ``WHERE`` therefore
    restricts the aggregated entity.
    """

    @classmethod
    def applies(cls, expr, selected_var, ctx: VerbalizationContext) -> bool:
        return (
            isinstance(selected_var, Aggregator)
            and aggregation_source_root(expr) is not None
        )

    @classmethod
    def subject(cls, expr, selected_var, ctx: VerbalizationContext):
        return aggregation_source_root(expr)


RESTRICTION_SUBJECT_RULES: List[type] = [
    SelectedVariableSubjectRule,
    AggregationSourceSubjectRule,
]


def restriction_subject(expr, selected_var, ctx: VerbalizationContext):
    """
    Resolve the variable a selection's ``WHERE`` clause restricts (first match in
    :data:`RESTRICTION_SUBJECT_RULES` wins), or ``None`` when nothing groups.

    :param expr: The :class:`~krrood.entity_query_language.query.query.Entity` being verbalized.
    :param selected_var: Its selected variable (a ``Variable``, ``Aggregator``, …).
    :param ctx: Shared verbalization state.
    :returns: The subject ``Variable`` for *"whose"* folding, or ``None``.
    :rtype: ~krrood.entity_query_language.core.variable.Variable or None
    """
    for rule in RESTRICTION_SUBJECT_RULES:
        if rule.applies(expr, selected_var, ctx):
            return rule.subject(expr, selected_var, ctx)
    return None


# ── Builder ─────────────────────────────────────────────────────────────────────

@dataclass
class RestrictionClauseBuilder:
    """
    Partition a subject's ``WHERE`` condition into a *"whose <grouped>"* modifier and a
    residual condition fragment.
    """
    _verbalizer: EQLVerbalizer
    """The parent verbalizer for recursive sub-expression rendering."""

    def build(
        self, subject_var: Optional[Variable], condition: SymbolicExpression, ctx: VerbalizationContext
    ) -> tuple[Optional[VerbFragment], Optional[VerbFragment]]:
        """
        :param subject_var: The subject the conditions restrict (its single-hop attribute
            predicates become *"whose"* modifiers); may be ``None`` (nothing groups).
        :param condition: The ``WHERE`` condition expression.
        :param ctx: Shared verbalization state. Residual conditions are rendered through the
            main engine, so push *subject_var* as the coreference subject **before** calling
            this when pronouns are wanted in the residual.
        :returns: ``(whose_modifier | None, residual_condition | None)``.  ``whose_modifier``
            already includes the ``whose`` keyword; ``residual_condition`` does **not** include
            a ``such that`` / ``where`` keyword (the caller adds the appropriate one).
        :rtype: tuple
        """
        items = fold_range_pairs(flatten_operands(condition, AND))
        grouped: List[VerbFragment] = []
        residual: List = []
        for item in items:
            rule = self._match(item, subject_var, ctx)
            if rule is None:
                residual.append(item)
            else:
                grouped.append(rule.render(item, subject_var, ctx, self._verbalizer))

        whose_frag = None
        if grouped:
            whose_frag = PhraseFragment(
                parts=[
                    Keywords.WHOSE.as_fragment(),
                    oxford_and(grouped, Conjunctions.AND.as_fragment()),
                ],
            )
        residual_frag = self._render_residual(residual, ctx) if residual else None
        return whose_frag, residual_frag

    @staticmethod
    def _match(item, subject_var, ctx: VerbalizationContext) -> Optional[type]:
        for rule in RESTRICTION_RULES:
            if rule.applies(item, subject_var, ctx):
                return rule
        return None

    def _render_residual(
        self, items: List, ctx: VerbalizationContext
    ) -> VerbFragment:
        parts: List[VerbFragment] = []
        for item in items:
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        self._verbalizer.build(item.chain_expr, ctx),
                        self._verbalizer.build(item.lower_expression, ctx),
                        self._verbalizer.build(item.upper_expression, ctx),
                        compact=ctx.compact_predicates,
                    )
                )
            else:
                parts.append(self._verbalizer.build(item, ctx))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())
