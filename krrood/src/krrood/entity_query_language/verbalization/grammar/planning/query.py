"""
Query **planner** — pure structural analysis of an
:class:`~krrood.entity_query_language.query.query.Entity` / :class:`~krrood.entity_query_language.query.query.SetOf`
into a :class:`QueryPlan` (the *what to say* decisions).

It never builds fragments, mutates the context, or recurses — those are realisation
concerns owned by
:class:`~krrood.entity_query_language.verbalization.grammar.assembly.query.QueryAssembler`.
The plan is the **selection** concern: the selection shape, the definiteness
(``is_the``), the restriction subject and its WHERE partition (grouped *"whose …"*
predicates vs. residual *"such that …"*), and whether the entity is an aggregation
value-subquery.  The trailing clauses (GROUP BY / HAVING / ORDER BY) are owned by their
own components (see :mod:`~krrood.entity_query_language.verbalization.grammar.assembly.clauses`).

Reference: Reiter & Dale (2000) — content/structure determination (microplanning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import Any, List, Optional, Tuple, Type

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    flatten_operands,
)
from krrood.entity_query_language.query.quantifiers import The
from krrood.entity_query_language.query.query import Entity, Query, SetOf
from krrood.entity_query_language.verbalization.grammar.planning.base import Planner
from krrood.entity_query_language.verbalization.grammar.restriction import (
    RestrictionRule,
    match_restriction,
    restriction_subject,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    fold_range_pairs,
)
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_leaf_attribute,
    aggregation_source_root,
    is_aggregation_subquery,
    is_constrained_query,
    selected_aggregator,
)
from krrood.entity_query_language.verbalization.vocabulary.english import FallbackNouns


class SelectionKind(Enum):
    """The structural shape of a query's selection.

    :cvar ENTITY_SELECTOR: The selected variable is itself an Entity (a sub-query selector).
    :cvar EMPTY: No selected variable — the fallback *"entity"* form.
    :cvar SUBJECT: A plain variable / aggregator selection that can carry restrictions.
    :cvar SET_OF: A :class:`~krrood.entity_query_language.query.query.SetOf` tuple selection.
    """

    ENTITY_SELECTOR = auto()
    EMPTY = auto()
    SUBJECT = auto()
    SET_OF = auto()


@dataclass(frozen=True)
class RestrictionPlan:
    """Partition of a subject's WHERE condition into grouped vs. residual conjuncts."""

    grouped: List[Tuple[Type[RestrictionRule], Any]] = field(default_factory=list)
    """``(rule, folded item)`` pairs that fold into *"whose …"* (the rule renders each)."""

    residual: List[Any] = field(default_factory=list)
    """Folded items (``RangeFold`` or raw expression) for the residual *"such that …"*."""

    @property
    def has_grouped(self) -> bool:
        return bool(self.grouped)

    @property
    def has_residual(self) -> bool:
        return bool(self.residual)


@dataclass(frozen=True)
class AggregationData:
    """A collapsed aggregation subquery (*"the maximum amount among …"*)."""

    aggregator: Optional[Aggregator]
    """The selected aggregator."""

    leaf: Optional[MappedVariable]
    """The leaf :class:`Attribute` of the aggregator chain, or ``None``."""

    is_constrained: bool
    """``True`` when the aggregation is constrained by a WHERE/HAVING clause."""

    source: Optional[Variable]
    """The chain-root source variable aggregated over, or ``None``."""


@dataclass(frozen=True)
class QueryPlan:
    """Complete *what to say* decomposition of a query (the plan)."""

    kind: SelectionKind
    is_the: bool
    selected_type: str
    subject: Optional[Variable]
    subject_restriction: Optional[RestrictionPlan]
    where_condition: Optional[SymbolicExpression]
    is_aggregation_subquery: bool
    aggregation_data: Optional[AggregationData]


@dataclass
class QueryPlanner(Planner[Query, QueryPlan]):
    """Decompose an :class:`Entity` / :class:`SetOf` into a :class:`QueryPlan`."""

    def plan(self) -> QueryPlan:
        self.node.build()
        return QueryPlan(
            kind=self._kind(),
            is_the=self._is_the(),
            selected_type=self._selected_type(),
            subject=self._subject(),
            subject_restriction=self._subject_restriction(),
            where_condition=self._where_condition(),
            is_aggregation_subquery=is_aggregation_subquery(self.node),
            aggregation_data=self._aggregation_data(),
        )

    # ── selection shape ──────────────────────────────────────────────────────

    @property
    def _selected(self):
        return getattr(self.node, "selected_variable", None)

    def _kind(self) -> SelectionKind:
        if isinstance(self.node, SetOf):
            return SelectionKind.SET_OF
        var = self._selected
        if isinstance(var, Entity):
            return SelectionKind.ENTITY_SELECTOR
        if var is None:
            return SelectionKind.EMPTY
        return SelectionKind.SUBJECT

    def _is_the(self) -> bool:
        builder = getattr(self.node, "_quantifier_builder_", None)
        return builder is not None and builder.type is The

    def _selected_type(self) -> str:
        var = self._selected
        if var is not None and getattr(var, "_type_", None):
            return var._type_.__name__
        return FallbackNouns.ENTITY.text

    # ── subject restriction (WHERE partition) ────────────────────────────────

    def _subject(self) -> Optional[Variable]:
        if not isinstance(self.node, Entity):
            return None
        return restriction_subject(self.node, self._selected, None)

    def _subject_restriction(self) -> Optional[RestrictionPlan]:
        condition = self._where_condition()
        subject = self._subject()
        if condition is None or subject is None:
            return None
        return self._partition(subject, condition)

    def _partition(self, subject, condition) -> RestrictionPlan:
        """Fold range pairs, then sort conjuncts into grouped (rule-matched) vs. residual."""
        grouped: List[Tuple[Type[RestrictionRule], Any]] = []
        residual: List[Any] = []
        for item in fold_range_pairs(flatten_operands(condition, AND)):
            rule = match_restriction(item, subject, None)
            if rule is None:
                residual.append(item)
            else:
                grouped.append((rule, item))
        return RestrictionPlan(grouped=grouped, residual=residual)

    # ── clauses ──────────────────────────────────────────────────────────────

    def _where_condition(self) -> Optional[SymbolicExpression]:
        where = getattr(self.node, "_where_expression_", None)
        return where.condition if where is not None else None

    # ── aggregation value-subquery ───────────────────────────────────────────

    def _aggregation_data(self) -> Optional[AggregationData]:
        if not is_aggregation_subquery(self.node):
            return None
        return AggregationData(
            aggregator=selected_aggregator(self.node),
            leaf=aggregation_leaf_attribute(self.node),
            is_constrained=is_constrained_query(self.node),
            source=aggregation_source_root(self.node),
        )
