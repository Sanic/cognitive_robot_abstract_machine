from __future__ import annotations

from typing_extensions import List, Optional, Union

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.query.builders import OrderedByBuilder
from krrood.entity_query_language.query.operations import GroupedBy, OrderedBy
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_comma,
    PhraseFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
    Separator,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.clauses.planner import (
    GroupedByPlanner,
    GroupPlan,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    Keywords,
    SortDirections,
)


class GroupedByAssembler(Assembler[Union[Query, GroupedBy], GroupPlan]):
    """
    *"grouped by <keys>"* — or *"and the <aggregated> are grouped by <keys>"* in a query.

    Reference: Reiter & Dale (2000) — aggregation / clause structuring; Gatt & Reiter (2009),
    SimpleNLG — surface realisation.

    A grouped set-of *report* fronts its grouping as *"For each <key>, report …"* in the query
    assembler, so this clause is used for the in-query weave (*"and the <aggregated> are grouped by
    …"*) and bare grouped-by nodes.

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(
    ...     a(set_of(employee.department, sum(employee.salary)).grouped_by(employee.department))
    ... )
    'For each department, report the sum of salaries of Employees'
    """

    planner = GroupedByPlanner

    def realize(self, node: Union[Query, GroupedBy], plan: GroupPlan) -> Fragment:
        """
        :param node: The query (or bare grouped-by node) being rendered.
        :param plan: The group plan.
        :return: *"grouped by <keys>"* — or *"and the <aggregated> are grouped by <keys>"* when
            the query also selects aggregations (bare *"grouped"* when there are no keys).

        It renders the grouping key (*"department"*) that the report is grouped on. For the grouped
        set-of report in the class example, the query assembler fronts that key as the leading *"For
        each department, report"* frame and drops this trailing clause as redundant.
        """
        if not plan.has_keys:
            return Keywords.GROUPED.as_fragment()
        groups_phrase = self._keys_phrase(plan.keys)
        if plan.aggregated and plan.weaves_aggregated:
            aggregated_phrase = oxford_comma(
                [
                    self.context.child(expression, number=GrammaticalNumber.PLURAL)
                    for expression in plan.aggregated
                ],
                Conjunctions.AND.as_fragment(),
            )
            return PhraseFragment(
                parts=[
                    Conjunctions.AND.as_fragment(),
                    Articles.THE.as_fragment(),
                    aggregated_phrase,
                    Copulas.ARE.as_fragment(),
                    Keywords.GROUPED_BY.as_fragment(),
                    groups_phrase,
                ]
            )
        return PhraseFragment(parts=[Keywords.GROUPED_BY.as_fragment(), groups_phrase])

    def clause(self, node: Union[Query, GroupedBy]) -> Optional[Fragment]:
        """
        :param node: The query being rendered.
        :return: The in-query GROUP BY clause, or ``None`` when there are no group keys.

        It is the in-query grouped-by gate: it renders the clause only when group keys are present.
        For the grouped report shown the keys surface as the fronted *"For each department, report"*
        in the query assembler, so this trailing clause is suppressed there:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(
        ...     an(entity(employee).grouped_by(employee.department))
        ... )
        'For each department, report all Employees'
        """
        plan = self.plan(node)
        return self.realize(node, plan) if plan.has_keys else None

    def _keys_phrase(self, variables: List[SymbolicExpression]) -> Fragment:
        """
        :param variables: The group-by key expressions.
        :return: The comma-joined group keys *"<key1>, <key2>, …"*.

        It assembles the list of grouping keys (*"department"*, *"name"*). For the fronted report
        shown those same keys are coordinated as *"For each department and name, report"* by the
        query assembler:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(
        ...     a(set_of(employee.department, employee.name, sum(employee.salary)).grouped_by(employee.department, employee.name))
        ... )
        'For each department and name, report the sum of salaries of Employees'
        """
        return PhraseFragment(
            parts=[self.context.child(variable) for variable in variables],
            separator=Separator.COMMA,
        )


class OrderedByAssembler(Assembler[Union[OrderedBy, OrderedByBuilder], None]):
    """*"ordered by <variable> from lowest to highest / from highest to lowest"*. Realisation-only
    (no plan).

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(a(set_of(employee).ordered_by(employee.salary, descending=True)))
    'Report Employees ordered by their salaries from highest to lowest'
    """

    def realize(
        self, node: Union[OrderedBy, OrderedByBuilder], plan: None = None
    ) -> Fragment:
        """
        *node* is "ordered-like": an ``OrderedBy`` expression or an ``OrderedByBuilder``, both
        exposing ``.variable`` and ``.descending``.

        :param node: The ordered-by expression or builder.
        :param plan: Unused (this assembler has no plan).
        :return: The clause *"ordered by <variable> from lowest to highest"* (ascending) or
            *"… from highest to lowest"* (descending).

        It emits only the trailing ordering span — here *"ordered by their salaries from highest to
        lowest"* — and chooses the descending direction word; *"Report Employees"* is supplied by the
        query.
        """
        direction = (
            SortDirections.DESCENDING if node.descending else SortDirections.ASCENDING
        )
        return PhraseFragment(
            parts=[
                Keywords.ORDERED_BY.as_fragment(),
                self.context.child(node.variable),
                direction.as_fragment(),
            ]
        )

    def clause(self, query: Query) -> Optional[Fragment]:
        """
        :param query: The query being rendered.
        :return: The in-query ORDER BY clause, or ``None`` when the query is unordered.

        It is the in-query ordering gate: present an order builder and it renders the trailing
        *"ordered by their salaries from lowest to highest"* span, otherwise nothing:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
        """
        builder = query._ordered_by_builder_
        return self.realize(builder) if builder is not None else None


class HavingAssembler(Assembler[Query, None]):
    """*"having <condition>"* (compact comparators). Realisation-only (no plan).

    >>> employee = variable(Employee, [])
    >>> total = sum(employee.salary)
    >>> verbalize_expression(
    ...     a(set_of(employee.department, total).grouped_by(employee.department).having(total > 30000))
    ... )
    'For each department, report the sum of salaries of Employees having the sum greater than 30000'
    """

    def realize(self, node: Query, plan: None = None) -> Fragment:
        """
        :param node: The query whose HAVING condition to render.
        :param plan: Unused (this assembler has no plan).
        :return: *"having <condition>"* — the condition rendered with compact (copula-less)
            comparators.

        It emits only the trailing *"having the sum greater than 30000"* span, using compact
        comparators so the condition reads tersely; the rest of the report comes from elsewhere.
        """
        with self.context.configuration.compact_predicates_scope():
            having_fragment = self.context.child(node._having_expression_.condition)
        return PhraseFragment(parts=[Keywords.HAVING.as_fragment(), having_fragment])

    def clause(self, query: Query) -> Optional[Fragment]:
        """
        :param query: The query being rendered.
        :return: The in-query HAVING clause, or ``None`` when there is no HAVING.

        It is the in-query having gate: present a HAVING expression and it renders the trailing
        *"having the number greater than 5"* span, otherwise nothing:

        >>> employee = variable(Employee, [])
        >>> headcount = count(employee.name)
        >>> verbalize_expression(
        ...     a(set_of(employee.department, headcount).grouped_by(employee.department).having(headcount > 5))
        ... )
        'For each department, report the number of names of Employees having the number greater than 5'
        """
        return self.realize(query) if query._having_expression_ is not None else None
