"""
Rule structure analysis for EQL verbalization.

Classifies the parts of an Entity-over-inference query into antecedents (IF) and
a consequent (THEN) so the verbalizer can produce the "If ..., then ..." form.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing_extensions import Any, FrozenSet, List, Optional, Tuple

from krrood.entity_query_language.verbalization import morphology

from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import AND
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.chain_utils import chain_root


class AggregationStatus(Enum):
    """
    Indicates how a consequent binding or antecedent relates to the GROUP BY clause.

    :cvar GROUP_KEY: This expression is one of the ``grouped_by`` key variables.
    :cvar AGGREGATED: Present in the query but not a group key — should appear in plural form.
    :cvar NONE: No grouping context in this query.
    """

    GROUP_KEY = auto()  # this expression is one of the grouped_by keys
    AGGREGATED = auto()  # present but not a group key → plural in output
    NONE = auto()  # no grouping context


@dataclass
class AntecedentInfo:
    """
    Descriptor for one antecedent variable in the IF clause.

    """

    root: Any
    """The underlying :class:`~krrood.entity_query_language.core.variable.Variable`
    or :class:`~krrood.entity_query_language.query.query.Entity` (unwrapped from any
    :class:`~krrood.entity_query_language.query.quantifiers.ResultQuantifier` wrapper)."""

    type_name: str
    """Human-readable Python type name of *root* (e.g. ``"Robot"``)."""

    aggregation_status: AggregationStatus
    """Whether this antecedent is a group key, aggregated, or neither."""

    conditions: List[Any] = field(default_factory=list)
    """All WHERE conditions attributable to this antecedent (its own WHERE clause
    merged with matched outer WHERE conditions)."""


@dataclass
class ConsequentBinding:
    """
    Descriptor for one field binding in the THEN clause.

    """

    field_name: str
    """Python attribute name on the consequent type (e.g. ``"tasks"``)."""

    value_expression: Any
    """EQL expression providing the value for *field_name*."""

    is_plural_field: bool
    """``True`` when *field_name* is already plural (detected via ``inflect.singular_noun``)."""

    aggregation_status: AggregationStatus
    """Whether the value is a group key, aggregated, or neither."""


@dataclass
class RuleStructure:
    """
    Complete decomposition of an inference-rule Entity query.

    Produced by :meth:`RuleAnalyzer.analyze` and consumed by
    :class:`~krrood.entity_query_language.verbalization.rules.inference_rule.InferenceRuleRule`.

    """

    primary_antecedents: List[AntecedentInfo]
    """Antecedents with at least one condition — appear as items in the IF block."""

    secondary_antecedents: List[AntecedentInfo]
    """Antecedents with no conditions — only registered in
    :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.seen`
    for coreference."""

    consequent_type: str
    """Python type name of the inferred variable (e.g. ``"Drawer"``)."""

    consequent_bindings: List[ConsequentBinding]
    """Ordered list of field bindings for the THEN clause."""

    unmatched_conditions: List[Any]
    """Outer WHERE conditions not attributable to any antecedent."""

    group_key_ids: FrozenSet[uuid.UUID]
    """Frozen set of ``_id_`` values of the GROUP BY key variables."""


# ── Module-level helpers (pure domain-analysis utilities) ─────────────────────


def _antecedent_var_id_(antecedent: AntecedentInfo) -> Optional[object]:
    """Return the stable _id_ of the underlying variable for an antecedent."""
    root = antecedent.root
    if isinstance(root, Entity):
        root.build()
        sel = root.selected_variable
        return getattr(sel, "_id_", None)
    return getattr(root, "_id_", None)


def _condition_left_owner_id_(condition) -> Optional[object]:
    """
    Return the _id_ of the root variable on the left-hand side of an equality condition,
    or None if the condition is not a simple attribute equality.
    """
    if not isinstance(condition, Comparator) or condition.operation is not operator.eq:
        return None
    current = chain_root(condition.left)
    while isinstance(current, ResultQuantifier):
        current = current._child_
    return getattr(current, "_id_", None)


class RuleAnalyzer:
    """
    Analyses an :class:`~krrood.entity_query_language.query.query.Entity`-over-inference
    query and returns a :class:`RuleStructure`.

    The analyzer is stateless; a single shared instance is used by
    :class:`~krrood.entity_query_language.verbalization.rules.inference_rule.InferenceRuleRule`.
    """

    def can_handle(self, entity) -> bool:
        """
        Return ``True`` when *entity*'s selected variable is an
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`.

        :param entity: An :class:`~krrood.entity_query_language.query.query.Entity` expression.
        :return: ``True`` if this analyzer can decompose *entity* as an inference rule.
        :rtype: bool
        """
        entity.build()
        return isinstance(entity.selected_variable, InstantiatedVariable)

    def analyze(self, entity) -> RuleStructure:
        """
        Decompose *entity* into a :class:`RuleStructure`.

        Algorithm:

        1. Collect GROUP BY key IDs from the grouped-by expression.
        2. Walk each field binding of the
           :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
           to build :class:`ConsequentBinding` entries and discover antecedent roots.
        3. Attribute outer WHERE conditions to antecedents by matching the left-hand
           root variable ID.
        4. Split antecedents into primary (have conditions) and secondary (none).

        :param entity: An :class:`~krrood.entity_query_language.query.query.Entity` whose
            selected variable is an
            :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`.
        :return: Fully populated :class:`RuleStructure`.
        :rtype: RuleStructure
        :raises AttributeError: If *entity* has not been built or is not an inference-rule query.
        """
        entity.build()
        inferred: InstantiatedVariable = entity.selected_variable
        type_name = getattr(inferred._type_, "__name__", str(inferred._type_))

        # ── Group-key IDs ──────────────────────────────────────────────────────
        grouped_expression = entity._grouped_by_expression_
        group_key_ids: FrozenSet[uuid.UUID] = frozenset()
        if grouped_expression is not None and grouped_expression.variables_to_group_by:
            group_key_ids = frozenset(
                variable._id_ for variable in grouped_expression.variables_to_group_by
            )
        has_grouping = bool(group_key_ids)

        # ── Walk consequent bindings ───────────────────────────────────────────
        seen_root_ids: dict = {}  # root_id → AntecedentInfo
        consequent_bindings: List[ConsequentBinding] = []

        for field_name, child_expression in inferred._child_vars_.items():
            is_plural = morphology.is_plural(field_name)

            if child_expression._id_ in group_key_ids:
                binding_aggregation = AggregationStatus.GROUP_KEY
            elif has_grouping:
                binding_aggregation = AggregationStatus.AGGREGATED
            else:
                binding_aggregation = AggregationStatus.NONE

            consequent_bindings.append(
                ConsequentBinding(
                    field_name=field_name,
                    value_expression=child_expression,
                    is_plural_field=is_plural,
                    aggregation_status=binding_aggregation,
                )
            )

            root = self._find_root(child_expression)
            if root is None or root._id_ in seen_root_ids:
                continue

            root_type_name, own_conditions = self._extract_root_info(root)

            if root._id_ in group_key_ids:
                variable_aggregation = AggregationStatus.GROUP_KEY
            elif has_grouping:
                variable_aggregation = AggregationStatus.AGGREGATED
            else:
                variable_aggregation = AggregationStatus.NONE

            seen_root_ids[root._id_] = AntecedentInfo(
                root=root,
                type_name=root_type_name,
                aggregation_status=variable_aggregation,
                conditions=own_conditions,
            )

        # ── Attribute outer WHERE conditions to antecedents ────────────────────
        where_expression = entity._where_expression_
        extra: List[Any] = []
        if where_expression is not None:
            extra = self._flatten_and(where_expression.condition)

        primary, secondary, unmatched = self._attribute_conditions_(
            list(seen_root_ids.values()), extra
        )

        return RuleStructure(
            primary_antecedents=primary,
            secondary_antecedents=secondary,
            consequent_type=type_name,
            consequent_bindings=consequent_bindings,
            unmatched_conditions=unmatched,
            group_key_ids=group_key_ids,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _attribute_conditions_(
        antecedents: List[AntecedentInfo],
        extra_conditions: List[Any],
    ) -> Tuple[List[AntecedentInfo], List[AntecedentInfo], List[Any]]:
        """
        Distribute extra outer-WHERE conditions to their owning antecedents,
        then classify antecedents as primary (have conditions) or secondary (none).
        Returns (primary, secondary, unmatched).
        """
        id_to_antecedent = {_antecedent_var_id_(a): a for a in antecedents}
        unmatched: List[Any] = []

        for condition in extra_conditions:
            owner_id = _condition_left_owner_id_(condition)
            if owner_id is not None and owner_id in id_to_antecedent:
                id_to_antecedent[owner_id].conditions.append(condition)
            else:
                unmatched.append(condition)

        primary = [a for a in antecedents if a.conditions]
        secondary = [a for a in antecedents if not a.conditions]
        return primary, secondary, unmatched

    @staticmethod
    def _find_root(expression) -> Optional[Any]:
        current = chain_root(expression)
        while isinstance(current, ResultQuantifier):
            current = current._child_
        if isinstance(current, (Variable, Entity)):
            return current
        return None

    @staticmethod
    def _extract_root_info(root) -> Tuple[str, List[Any]]:
        """Return (type_name, own_conditions) for a root Variable or Entity."""
        if isinstance(root, Entity):
            root.build()
            var = root.selected_variable
            type_name = (
                var._type_.__name__
                if var and getattr(var, "_type_", None)
                else "entity"
            )
            conditions = []
            if root._where_expression_ is not None:
                conditions = RuleAnalyzer._flatten_and(
                    root._where_expression_.condition
                )
            return type_name, conditions

        if isinstance(root, Variable):
            type_name = (
                root._type_.__name__ if getattr(root, "_type_", None) else "variable"
            )
            return type_name, []

        return "entity", []

    @staticmethod
    def _flatten_and(expression) -> List[Any]:
        """Recursively flatten a nested AND tree into a flat list of conjuncts."""
        if isinstance(expression, AND):
            return RuleAnalyzer._flatten_and(
                expression.left
            ) + RuleAnalyzer._flatten_and(expression.right)
        return [expression]
