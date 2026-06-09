"""
Inference-rule **planner** — pure structural analysis of an
:class:`~krrood.entity_query_language.query.query.Entity` whose selected variable is an
:class:`~krrood.entity_query_language.core.variable.InstantiatedVariable` (an inference
rule) into a :class:`RuleStructure` (the IF/THEN decomposition).

This is the analysis half of the planner/assembler split (see
:class:`~krrood.entity_query_language.verbalization.grammar.planning.base.Planner`):
it produces a plain data record of decisions and never builds fragments, touches the
context, or recurses.  The realisation lives in
:class:`~krrood.entity_query_language.verbalization.grammar.assembly.inference.InferenceAssembler`.

Reference: Reiter & Dale (2000) — content/structure determination (microplanning).
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import Any, FrozenSet, List, Optional, Tuple

from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import AND
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.chain_utils import chain_root
from krrood.entity_query_language.verbalization.grammar.planning.base import Planner


class AggregationStatus(Enum):
    """
    How a consequent binding or antecedent relates to the GROUP BY clause.

    :cvar GROUP_KEY: This expression is one of the ``grouped_by`` key variables.
    :cvar AGGREGATED: Present in the query but not a group key — rendered in plural form.
    :cvar NONE: No grouping context in this query.
    """

    GROUP_KEY = auto()
    AGGREGATED = auto()
    NONE = auto()


@dataclass
class AntecedentInfo:
    """Descriptor for one antecedent variable in the IF clause."""

    root: Any
    """The underlying Variable/Entity (unwrapped from any ResultQuantifier)."""

    type_name: str
    """Human-readable Python type name of *root* (e.g. ``"Robot"``)."""

    aggregation_status: AggregationStatus
    """Whether this antecedent is a group key, aggregated, or neither."""

    conditions: List[Any] = field(default_factory=list)
    """All WHERE conditions attributable to this antecedent."""


@dataclass
class ConsequentBinding:
    """Descriptor for one field binding in the THEN clause."""

    field_name: str
    """Python attribute name on the consequent type (e.g. ``"tasks"``)."""

    value_expression: Any
    """EQL expression providing the value for *field_name*."""

    is_plural_field: bool
    """``True`` when *field_name* is already plural."""

    aggregation_status: AggregationStatus
    """Whether the value is a group key, aggregated, or neither."""


@dataclass
class RuleStructure:
    """Complete decomposition of an inference-rule Entity query (the plan)."""

    primary_antecedents: List[AntecedentInfo]
    """Antecedents with at least one condition — items in the IF block."""

    secondary_antecedents: List[AntecedentInfo]
    """Antecedents with no conditions — only registered for coreference."""

    consequent_type: str
    """Python type name of the inferred variable (e.g. ``"Drawer"``)."""

    consequent_bindings: List[ConsequentBinding]
    """Ordered field bindings for the THEN clause."""

    unmatched_conditions: List[Any]
    """Outer WHERE conditions not attributable to any antecedent."""

    group_key_ids: FrozenSet[uuid.UUID]
    """``_id_`` values of the GROUP BY key variables."""


@dataclass
class InferencePlanner(Planner[Entity, RuleStructure]):
    """
    Decompose an inference-rule :class:`~krrood.entity_query_language.query.query.Entity`
    into a :class:`RuleStructure`.

    Algorithm: collect GROUP BY key ids; walk each consequent field binding to build
    :class:`ConsequentBinding` entries and discover antecedent roots; attribute outer
    WHERE conditions to antecedents by matching the left-hand root id; split antecedents
    into primary (have conditions) and secondary (none).
    """

    @staticmethod
    def can_handle(entity) -> bool:
        """Return ``True`` when *entity*'s selected variable is an InstantiatedVariable."""
        entity.build()
        return isinstance(entity.selected_variable, InstantiatedVariable)

    def plan(self) -> RuleStructure:
        entity = self.node
        entity.build()
        inferred: InstantiatedVariable = entity.selected_variable
        type_name = getattr(inferred._type_, "__name__", str(inferred._type_))

        grouped_expression = entity._grouped_by_expression_
        group_key_ids: FrozenSet[uuid.UUID] = frozenset()
        if grouped_expression is not None and grouped_expression.variables_to_group_by:
            group_key_ids = frozenset(
                variable._id_ for variable in grouped_expression.variables_to_group_by
            )
        has_grouping = bool(group_key_ids)

        seen_root_ids: dict = {}
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

        where_expression = entity._where_expression_
        extra: List[Any] = []
        if where_expression is not None:
            extra = self._flatten_and(where_expression.condition)

        primary, secondary, unmatched = self._attribute_conditions(
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

    # ── analysis sub-steps (methods) ────────────────────────────────────────────

    def _attribute_conditions(
        self, antecedents: List[AntecedentInfo], extra_conditions: List[Any]
    ) -> Tuple[List[AntecedentInfo], List[AntecedentInfo], List[Any]]:
        """Distribute outer-WHERE conditions to owning antecedents; split primary/secondary."""
        id_to_antecedent = {self._antecedent_var_id(a): a for a in antecedents}
        unmatched: List[Any] = []
        for condition in extra_conditions:
            owner_id = self._condition_left_owner_id(condition)
            if owner_id is not None and owner_id in id_to_antecedent:
                id_to_antecedent[owner_id].conditions.append(condition)
            else:
                unmatched.append(condition)
        primary = [a for a in antecedents if a.conditions]
        secondary = [a for a in antecedents if not a.conditions]
        return primary, secondary, unmatched

    def _antecedent_var_id(self, antecedent: AntecedentInfo) -> Optional[object]:
        """Stable ``_id_`` of the underlying variable for an antecedent."""
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            return getattr(root.selected_variable, "_id_", None)
        return getattr(root, "_id_", None)

    def _condition_left_owner_id(self, condition) -> Optional[object]:
        """``_id_`` of the root variable on the LHS of an equality condition, else ``None``."""
        if (
            not isinstance(condition, Comparator)
            or condition.operation is not operator.eq
        ):
            return None
        current = chain_root(condition.left)
        while isinstance(current, ResultQuantifier):
            current = current._child_
        return getattr(current, "_id_", None)

    def _find_root(self, expression) -> Optional[Any]:
        current = chain_root(expression)
        while isinstance(current, ResultQuantifier):
            current = current._child_
        if isinstance(current, (Variable, Entity)):
            return current
        return None

    def _extract_root_info(self, root) -> Tuple[str, List[Any]]:
        """Return ``(type_name, own_conditions)`` for a root Variable or Entity."""
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
                conditions = self._flatten_and(root._where_expression_.condition)
            return type_name, conditions
        if isinstance(root, Variable):
            type_name = (
                root._type_.__name__ if getattr(root, "_type_", None) else "variable"
            )
            return type_name, []
        return "entity", []

    def _flatten_and(self, expression) -> List[Any]:
        """Recursively flatten a nested AND tree into a flat list of conjuncts."""
        if isinstance(expression, AND):
            return self._flatten_and(expression.left) + self._flatten_and(
                expression.right
            )
        return [expression]
