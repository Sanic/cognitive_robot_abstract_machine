from __future__ import annotations

from abc import abstractmethod
from enum import Enum, auto
from typing_extensions import ClassVar, Optional, Type, Union

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    references,
    single_hop_attribute,
    superlative_aggregation,
)
from krrood.entity_query_language.verbalization.grammar.conditions.verbalizer import (
    ConditionVerbalizer,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import RuleContext
from krrood.entity_query_language.verbalization.grammar.framework.specificity import SpecificityRule
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    RangeFold,
)
from krrood.entity_query_language.verbalization.subquery import aggregation_source_root

# ── restriction rules (folded conjunct → fragment + its placement) ───────────


class Placement(Enum):
    """Where a matched restriction's fragment attaches in the query — the taxonomy of
    restriction surface slots."""

    SELECTION_MODIFIER = auto()
    """A post-nominal prepositional phrase on the selection — *"<noun> with the maximum amount"*."""
    WHOSE_GROUP = auto()
    """A bare predicate gathered under one shared *"whose …, and …"* envelope."""


class RestrictionRule(SpecificityRule):
    """
    Recognise a folded conjunct as a subject restriction and render its fragment, declaring
    where that fragment attaches via ``placement``.  A conjunct matched by no rule is
    residual and stays in a *"such that …"* clause.

    Reference: Dale & Reiter (1995) — referring expressions / post-nominal modification.
    """

    placement: ClassVar[Placement]
    """The surface slot this rule's output occupies."""

    @classmethod
    @abstractmethod
    def applies(
        cls, item: Union[SymbolicExpression, RangeFold], subject_variable: Variable
    ) -> bool:
        """
        :param item: A folded ``WHERE`` conjunct.
        :param subject_variable: The variable the restriction is on.
        :return: ``True`` when *item* is a restriction on *subject_variable* this rule renders.
        """

    @classmethod
    @abstractmethod
    def render(
        cls,
        item: Union[SymbolicExpression, RangeFold],
        subject_variable: Variable,
        context: RuleContext,
    ) -> Fragment:
        """
        :param item: The folded conjunct this rule matched.
        :param subject_variable: The variable the restriction is on.
        :param context: The per-node context (recursion and services).
        :return: *item* rendered as the fragment for this rule's placement.
        """


class RangeRestrictionRule(RestrictionRule):
    """A range fold on a single-hop subject attribute → *"<attribute> is between low and high"*.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     an(entity(employee).where(and_(employee.salary > 100, employee.salary < 200)))
    ... )
    'Find an Employee whose salary is between 100, and 200'
    """

    placement = Placement.WHOSE_GROUP

    @classmethod
    def applies(
        cls, item: Union[SymbolicExpression, RangeFold], subject_variable: Variable
    ) -> bool:
        return (
            isinstance(item, RangeFold)
            and single_hop_attribute(item.chain_expression, subject_variable)
            is not None
        )

    @classmethod
    def render(
        cls, item: RangeFold, subject_variable: Variable, context: RuleContext
    ) -> Fragment:
        return ConditionVerbalizer(context).range_modifier(item, subject_variable)


class AttributePredicateRestrictionRule(RestrictionRule):
    """
    A single-hop, non-boolean subject-attribute comparator whose right-hand side does not reference the
    subject → *"<attribute> is greater than 100"* / *"<attribute> is equal to <calc>"*.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """

    placement = Placement.WHOSE_GROUP

    @classmethod
    def applies(
        cls, item: Union[SymbolicExpression, RangeFold], subject_variable: Variable
    ) -> bool:
        if not isinstance(item, Comparator):
            return False
        attribute = single_hop_attribute(item.left, subject_variable)
        if attribute is None or attribute._type_ is bool:
            return False
        if superlative_aggregation(item, subject_variable) is not None:
            return False  # a superlative comparator is SuperlativeRestrictionRule's
        return not references(item.right, subject_variable)

    @classmethod
    def render(
        cls, item: Comparator, subject_variable: Variable, context: RuleContext
    ) -> Fragment:
        return ConditionVerbalizer(context).attribute_modifier(item, subject_variable)


class SuperlativeRestrictionRule(RestrictionRule):
    """
    ``subject.<chain> == max/min(over all <same-type>.<same chain>)`` → the superlative selection
    modifier *"with the maximum/minimum <leaf>"*.

    >>> t1 = variable(BankTransaction, domain=None)
    >>> t2 = variable(BankTransaction, domain=None)
    >>> verbalize_expression(
    ...     the(entity(t1).where(
    ...         t1.amount_details.amount == an(entity(max(t2.amount_details.amount)))
    ...     ))
    ... )
    'Find the unique BankTransaction with the maximum amount'
    """

    placement = Placement.SELECTION_MODIFIER

    @classmethod
    def applies(
        cls, item: Union[SymbolicExpression, RangeFold], subject_variable: Variable
    ) -> bool:
        return superlative_aggregation(item, subject_variable) is not None

    @classmethod
    def render(
        cls, item: Comparator, subject_variable: Variable, context: RuleContext
    ) -> Fragment:
        return ConditionVerbalizer(context).superlative_modifier(item, subject_variable)


def match_restriction(
    item: Union[SymbolicExpression, RangeFold], subject_variable: Variable
) -> Optional[Type[RestrictionRule]]:
    """
    :param item: A folded ``WHERE`` conjunct.
    :param subject_variable: The variable the restriction is on.
    :return: The most-specific applicable restriction rule for *item*, or ``None`` (residual).
    """
    return RestrictionRule.most_applicable(item, subject_variable)


# ── restriction-subject rules (which variable does the WHERE restrict?) ──────


class RestrictionSubjectRule(SpecificityRule):
    """
    Resolve which variable a query's selection restricts, so the selection's ``WHERE`` can fold
    into a post-nominal *"whose …"* modifier on it.  A selection matched by no rule has no
    groupable subject — its ``WHERE`` stays a full *"such that …"* clause.
    """

    @classmethod
    @abstractmethod
    def applies(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> bool:
        """
        :param expression: The query whose restriction subject to resolve.
        :param selected_variable: The query's selected variable.
        :return: ``True`` when this rule can name the restriction subject of *expression*.
        """

    @classmethod
    @abstractmethod
    def subject(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> Optional[Variable]:
        """
        :param expression: The query whose restriction subject to resolve.
        :param selected_variable: The query's selected variable.
        :return: The variable the ``WHERE`` restricts.
        """


class SelectedVariableSubjectRule(RestrictionSubjectRule):
    """The selection is a plain variable → it is its own subject."""

    @classmethod
    def applies(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> bool:
        return isinstance(selected_variable, Variable)

    @classmethod
    def subject(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> Optional[Variable]:
        return selected_variable


class AggregationSourceSubjectRule(RestrictionSubjectRule):
    """
    The selection aggregates over a single source variable's chain (e.g.
    ``max(t.amount_details.amount)``); the ``WHERE`` restricts that aggregated entity,
    whose noun ends the selection so a *"whose …"* modifier attaches grammatically.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     an(entity(max(employee.salary)).where(employee.department == 'Sales'))
    ... )
    "Find the maximum of the salary of an Employee whose department is 'Sales'"
    """

    @classmethod
    def applies(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> bool:
        return (
            isinstance(selected_variable, Aggregator)
            and aggregation_source_root(expression) is not None
        )

    @classmethod
    def subject(
        cls, expression: SymbolicExpression, selected_variable: SymbolicExpression
    ) -> Optional[Variable]:
        return aggregation_source_root(expression)


def restriction_subject(
    expression: SymbolicExpression, selected_variable: SymbolicExpression
) -> Optional[Variable]:
    """
    :param expression: The query whose restriction subject to resolve.
    :param selected_variable: The query's selected variable.
    :return: The variable a selection's ``WHERE`` restricts (most-specific rule wins), or
        ``None``.
    """
    chosen = RestrictionSubjectRule.most_applicable(expression, selected_variable)
    return chosen.subject(expression, selected_variable) if chosen is not None else None
