from __future__ import annotations

from dataclasses import dataclass

import uuid

from typing_extensions import List, Optional, Set, Union

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.query.operations import GroupedBy
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.chain_utils import chain_root
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner


@dataclass(frozen=True)
class GroupPlan:
    """The GROUP BY keys and the expressions aggregated over them."""

    keys: List[SymbolicExpression]
    """The group-by key expressions (empty ⇒ a bare *"grouped"* / no clause)."""

    aggregated: List[SymbolicExpression]
    """Selected expressions aggregated (not group keys) — rendered plural; empty for a bare
    grouped-by node."""

    @property
    def has_keys(self) -> bool:
        """:return: ``True`` when the query carries at least one group-by key."""
        return bool(self.keys)


@dataclass
class GroupedByPlanner(Planner[Union[Query, GroupedBy], GroupPlan]):
    """
    Decompose the GROUP BY of *node* (a query or a bare grouped-by node) into a ``GroupPlan``.

    Reference: Reiter & Dale (2000) — content/structure determination (microplanning).

    >>> employee = variable(Employee, [])
    >>> GroupedByPlanner(
    ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
    ... ).plan().has_keys
    True
    """

    def plan(self) -> GroupPlan:
        """:return: The group plan: the group-by keys and the expressions aggregated over them."""
        grouped = self._grouped_by()
        if grouped is None or not grouped.variables_to_group_by:
            return GroupPlan(keys=[], aggregated=[])
        keys = list(grouped.variables_to_group_by)
        return GroupPlan(
            keys=keys, aggregated=self._aggregated(self._root_variable_ids(keys))
        )

    def _grouped_by(self) -> Optional[GroupedBy]:
        if isinstance(self.node, GroupedBy):
            return self.node
        return getattr(self.node, "_grouped_by_expression_", None)

    @staticmethod
    def _root_variable_ids(
        expressions: List[SymbolicExpression],
    ) -> Set[uuid.UUID]:
        ids: Set[uuid.UUID] = set()
        for expression in expressions:
            root = chain_root(expression)
            if isinstance(root, Variable):
                ids.add(root._id_)
        return ids

    def _aggregated(
        self, group_key_root_ids: Set[uuid.UUID]
    ) -> List[SymbolicExpression]:
        """
        :param group_key_root_ids: Root variable ids of the group-by keys.
        :return: Selected expressions that are aggregated (not group keys); ``[]`` off a query.
        """
        selected = (
            self.node.selected_variable if isinstance(self.node, Entity) else None
        )
        if isinstance(selected, InstantiatedVariable):
            return [
                child
                for child in selected._child_vars_.values()
                if not (
                    isinstance(chain_root(child), Variable)
                    and chain_root(child)._id_ in group_key_root_ids
                )
            ]
        if isinstance(self.node, Query):
            return [
                variable
                for variable in self.node._selected_variables_
                if variable._id_ not in group_key_root_ids
            ]
        return []
