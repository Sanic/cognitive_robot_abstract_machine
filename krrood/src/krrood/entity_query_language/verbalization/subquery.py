"""
Recognising and inspecting *value subqueries* — nested
:class:`~krrood.entity_query_language.query.query.Entity` queries that denote a
value rather than a top-level request.

A nested ``Entity`` whose selected variable is an
:class:`~krrood.entity_query_language.operators.aggregators.Aggregator` is an
*aggregation value subquery*.  When it carries no ``WHERE`` / ``HAVING`` /
``GROUP BY`` it is *unconstrained* and may be collapsed to a compact aggregate
noun phrase (e.g. *"the maximum amount"*); otherwise its filter is load-bearing
and must be preserved.

These predicates are the single source of truth for this concept, shared by the
disambiguation pre-scan
(:func:`~krrood.entity_query_language.verbalization.context._build_disambiguation_map`)
and the nested-entity renderer
(:func:`~krrood.entity_query_language.verbalization.rules.query.verbalize_nested`).
"""

from __future__ import annotations

from typing import Optional

from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity


def selected_aggregator(entity) -> Optional[Aggregator]:
    """
    Return the :class:`~krrood.entity_query_language.operators.aggregators.Aggregator`
    selected by *entity*, or ``None`` when *entity* is not an
    :class:`~krrood.entity_query_language.query.query.Entity` selecting one.

    Thin verbalization-facing adapter over
    :attr:`~krrood.entity_query_language.query.query.Entity.selected_aggregator`.

    :param entity: Candidate expression.
    :returns: The selected aggregator, or ``None``.
    :rtype: ~krrood.entity_query_language.operators.aggregators.Aggregator or None
    """
    return entity.selected_aggregator if isinstance(entity, Entity) else None


def is_aggregation_subquery(entity) -> bool:
    """
    Return ``True`` when *entity* is an
    :class:`~krrood.entity_query_language.query.query.Entity` whose selected
    variable is an aggregator.

    :param entity: Candidate expression.
    :rtype: bool
    """
    return selected_aggregator(entity) is not None


def is_calculation_value(expression) -> bool:
    """
    Return ``True`` when *expression* denotes a *calculation* — an
    :class:`~krrood.entity_query_language.operators.aggregators.Aggregator` or an
    aggregation sub-query (an :class:`~krrood.entity_query_language.query.query.Entity`
    selecting one), possibly wrapped in
    :class:`~krrood.entity_query_language.query.quantifiers.ResultQuantifier` s.

    Used to decide equality wording: ``==`` against a calculation reads
    *"is equal to"*, while ``==`` against a plain object/value reads *"is"*.

    :param expression: Candidate operand expression.
    :rtype: bool
    """
    inner = expression
    while isinstance(inner, ResultQuantifier):
        inner = inner._child_
    if isinstance(inner, Aggregator):
        return True
    if isinstance(inner, Entity):
        return is_aggregation_subquery(inner)
    return False


def is_constrained_query(entity) -> bool:
    """
    Return ``True`` when *entity* carries a ``WHERE``, ``HAVING``, or non-empty
    ``GROUP BY`` clause.

    :param entity: An :class:`~krrood.entity_query_language.query.query.Entity`.
    :rtype: bool
    """
    return entity.is_constrained


def is_collapsible_aggregation_subquery(entity) -> bool:
    """
    Return ``True`` when *entity* is an unconstrained aggregation subquery — one
    that may be collapsed to a compact aggregate noun phrase because it adds no
    filtering beyond the aggregation itself.

    :param entity: Candidate expression.
    :rtype: bool
    """
    return is_aggregation_subquery(entity) and not is_constrained_query(entity)


def aggregation_leaf_attribute(entity):
    """
    Return the leaf :class:`~krrood.entity_query_language.core.mapped_variable.Attribute`
    of the aggregator's child chain (e.g. the ``amount`` node behind
    ``max(t.amount_details.amount)``), or ``None`` when the aggregator's child is
    not an attribute chain (e.g. ``max(x)`` over a bare variable).

    :param entity: Candidate expression.
    :returns: The leaf attribute node, or ``None``.
    """
    aggregator = selected_aggregator(entity)
    return aggregator._leaf_attribute_ if aggregator is not None else None


def aggregation_source_root(entity) -> Optional[Variable]:
    """
    Return the root :class:`~krrood.entity_query_language.core.variable.Variable`
    that an aggregation subquery aggregates over (e.g. the ``BankTransaction``
    variable behind ``max(t.amount_details.amount)``), or ``None``.

    :param entity: Candidate expression.
    :returns: The chain-root variable of the aggregator's child, or ``None``.
    :rtype: ~krrood.entity_query_language.core.variable.Variable or None
    """
    aggregator = selected_aggregator(entity)
    return aggregator._source_root_ if aggregator is not None else None
