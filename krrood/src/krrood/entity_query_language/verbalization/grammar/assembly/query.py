"""
Query **assembler** — realise a :class:`~krrood.entity_query_language.verbalization.grammar.planning.query.QueryPlan`
into the *"Find … such that … grouped by … having … ordered by …"* block (and the nested
noun-phrase / aggregation-value forms).

This is the realisation half of the planner/assembler split: it owns recursion
(``self.ctx.child``), the render-scope mutations (query-depth, subject, compact
predicates) and the coreference bookkeeping (``self.ctx.context.seen``) — concerns that
are order-dependent and cannot be pre-computed (Reiter & Dale 2000).  All *what to say*
decisions already live in the plan; the assembler only combines.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.verbalization.grammar.agreement import noun_phrase
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import (
    phrase,
    role,
    word,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.aggregation_kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.assembly.clauses import (
    GroupedByAssembler,
    HavingAssembler,
    OrderedByAssembler,
)
from krrood.entity_query_language.verbalization.grammar.planning.query import (
    QueryPlan,
    QueryPlanner,
    RestrictionPlan,
    SelectionKind,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    build_between,
    RangeFold,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    FallbackNouns,
    Keywords,
    Prepositions,
)
from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm


class QueryAssembler(Assembler[Query, QueryPlan]):
    """Realise a query / nested-entity / set-of from its :class:`QueryPlan`."""

    planner = QueryPlanner

    # ── entry points ─────────────────────────────────────────────────────────

    def realize(self, node, plan: QueryPlan) -> VerbFragment:
        """Top-level imperative form: *"Find X such that …"*."""
        seen = self.ctx.context.seen_reference(node)
        if seen is not None:
            return seen
        with self.ctx.context.query_depth_scope():
            if plan.kind is SelectionKind.ENTITY_SELECTOR:
                selection = self._as_noun(node.selected_variable)
                return self._query_body(
                    node, plan, selection, where_item=self._where_clause(plan)
                )
            if plan.kind is SelectionKind.EMPTY:
                self.ctx.context.register_label(node, FallbackNouns.ENTITY.text)
                return self._query_body(
                    node,
                    plan,
                    FallbackNouns.ENTITY.plural_fragment(),
                    where_item=self._where_clause(plan),
                )
            return self._assemble_subject(node, plan)

    def assemble_nested(self, node) -> VerbFragment:
        """Noun-phrase form for a nested Entity (never emits *"Find …"*)."""
        plan = self.plan(node)
        seen = self.ctx.context.seen_reference(node)
        if seen is not None:
            return seen
        if plan.is_aggregation_subquery:
            return self._aggregation_value(node, plan)
        return self._as_noun(node)

    def assemble_set_of(self, node) -> VerbFragment:
        """*"Find (v1, v2, …) such that …"* for a SetOf query."""
        plan = self.plan(node)
        with self.ctx.context.query_depth_scope():
            variable_fragments = [
                self.ctx.child(variable) for variable in node._selected_variables_
            ]
            vars_phrase = PhraseFragment(parts=variable_fragments, separator=", ")
            selection = PhraseFragment(
                parts=[word("("), vars_phrase, word(")")], separator=""
            )
            return self._query_body(
                node,
                plan,
                selection,
                where_item=self._where_clause(plan),
                find_header=Keywords.FIND_SETS_OF.as_fragment(),
            )

    # ── subject selection ──────────────────────────────────────────────────────

    def _assemble_subject(self, node, plan: QueryPlan) -> VerbFragment:
        var = node.selected_variable
        self.ctx.context.push_subject(var)
        try:
            selected = self._build_selection(node, var, plan)
            selected, where_item = self._apply_subject_restrictions(plan, selected)
            return self._query_body(node, plan, selected, where_item=where_item)
        finally:
            self.ctx.context.pop_subject()

    def _build_selection(self, node, var, plan: QueryPlan) -> VerbFragment:
        if plan.is_the:
            selected_type = plan.selected_type
            self.ctx.context.register_label(var, selected_type)
            self.ctx.context.register_label(node, selected_type)
            return phrase(
                Articles.THE_UNIQUE.as_fragment(),
                role(selected_type, SemanticRole.VARIABLE),
            )
        selected = self.ctx.child(var)
        # The entity shares its selected variable's label (registered by ctx.child above).
        self.ctx.context.alias(node, var)
        return selected

    def _apply_subject_restrictions(
        self, plan: QueryPlan, selected: VerbFragment
    ) -> Tuple[VerbFragment, Optional[VerbFragment]]:
        restriction = plan.subject_restriction
        if restriction is None:
            return selected, None
        whose, residual = self._render_restrictions(restriction, plan.subject)
        if whose is not None:
            selected = phrase(selected, whose)
        where_item = (
            phrase(Keywords.SUCH_THAT.as_fragment(), residual)
            if residual is not None
            else None
        )
        return selected, where_item

    # ── noun forms ───────────────────────────────────────────────────────────

    def _as_noun(self, entity) -> VerbFragment:
        """Standalone-noun form: *"a Robot where …"* (for nested Entity selectors)."""
        seen = self.ctx.context.seen_reference(entity)
        if seen is not None:
            return seen
        plan = self.plan(entity)
        var = entity.selected_variable
        selected_type = plan.selected_type
        self.ctx.context.register_label(entity, selected_type)
        if var is not None:
            self.ctx.context.register_label(var, selected_type)

        if plan.is_the:
            article_noun: VerbFragment = phrase(
                Articles.THE_UNIQUE.as_fragment(),
                RoleFragment.for_variable(selected_type, var),
            )
        else:
            article_noun = phrase(
                Articles.indefinite(selected_type),
                RoleFragment.for_variable(selected_type, var),
            )

        if plan.subject_restriction is None:
            return article_noun
        with self.ctx.context.query_depth_scope():
            self.ctx.context.push_subject(var)
            try:
                whose, residual = self._render_restrictions(
                    plan.subject_restriction, plan.subject
                )
            finally:
                self.ctx.context.pop_subject()
        result = article_noun
        if whose is not None:
            result = phrase(result, whose)
        if residual is not None:
            result = phrase(result, Keywords.WHERE.as_fragment(), residual)
        return result

    # ── aggregation value-subquery ──────────────────────────────────────────────

    def _aggregation_value(self, node, plan: QueryPlan) -> VerbFragment:
        av = plan.aggregation_value
        if av.leaf is None:
            with self.ctx.context.query_depth_scope():
                return self.ctx.child(av.aggregator)

        aggregation_kind = AGGREGATION_KIND[type(av.aggregator)]
        plural_leaf = aggregation_kind.value.child_form == ChildForm.PLURAL
        leaf_frag = RoleFragment.for_attribute(
            av.leaf._owner_class_,
            av.leaf._attribute_name_,
            number=Number.of(plural_leaf),
        )
        aggregate = phrase(
            Articles.THE.as_fragment(), aggregation_kind.as_fragment(), leaf_frag
        )

        if av.aggregator._id_ not in self.ctx.context.seen:
            self.ctx.context.register(
                av.aggregator, phrase(aggregation_kind.as_fragment(), leaf_frag)
            )

        if not av.is_constrained:
            return aggregate
        return self._aggregation_scope(node, plan, aggregate)

    def _aggregation_scope(self, node, plan: QueryPlan, aggregate) -> VerbFragment:
        """Append *"among <plural source> such that <filter> having <filter>"*."""
        source = plan.aggregation_value.source
        source_frag = (
            noun_phrase(source, Number.PLURAL, self.ctx.context, self.ctx.child)
            if source is not None
            else FallbackNouns.ENTITY.plural_fragment()
        )
        parts: List[VerbFragment] = [
            aggregate,
            Prepositions.AMONG.as_fragment(),
            source_frag,
        ]

        if plan.subject_restriction is not None:
            with self.ctx.context.query_depth_scope():
                whose, residual = self._render_restrictions(
                    plan.subject_restriction, plan.subject
                )
            if whose is not None:
                parts.append(whose)
            if residual is not None:
                parts += [Keywords.SUCH_THAT.as_fragment(), residual]

        with self.ctx.context.query_depth_scope():
            having = HavingAssembler(self.ctx).clause(node)
        if having is not None:
            parts.append(having)

        return phrase(*parts)

    # ── restriction rendering ──────────────────────────────────────────────────

    def _render_restrictions(
        self, restriction: RestrictionPlan, subject
    ) -> Tuple[Optional[VerbFragment], Optional[VerbFragment]]:
        """Render the *"whose <grouped>"* modifier and the residual condition."""
        grouped_frags = [
            rule.render(item, subject, self.ctx) for rule, item in restriction.grouped
        ]
        whose = None
        if grouped_frags:
            whose = PhraseFragment(
                parts=[
                    Keywords.WHOSE.as_fragment(),
                    oxford_and(grouped_frags, Conjunctions.AND.as_fragment()),
                ]
            )
        residual = (
            self._render_residual(restriction.residual)
            if restriction.has_residual
            else None
        )
        return whose, residual

    def _render_residual(self, items: List) -> VerbFragment:
        parts: List[VerbFragment] = []
        for item in items:
            if isinstance(item, RangeFold):
                parts.append(
                    build_between(
                        self.ctx.child(item.chain_expression),
                        self.ctx.child(item.lower_expression),
                        self.ctx.child(item.upper_expression),
                        compact=self.ctx.context.compact_predicates,
                    )
                )
            else:
                parts.append(self.ctx.child(item))
        if len(parts) == 1:
            return parts[0]
        return oxford_and(parts, Conjunctions.AND.as_fragment())

    # ── query-body clauses ─────────────────────────────────────────────────────

    def _query_body(
        self,
        node,
        plan: QueryPlan,
        selection: VerbFragment,
        where_item: Optional[VerbFragment],
        find_header: Optional[VerbFragment] = None,
    ) -> VerbFragment:
        if find_header is None:
            find_header = Keywords.FIND.as_fragment()
        header = phrase(find_header, selection)
        clauses = [
            clause
            for clause in [where_item, *self._trailing_clauses(node)]
            if clause is not None
        ]
        return BlockFragment(header=header, items=clauses)

    def _trailing_clauses(self, node) -> List[Optional[VerbFragment]]:
        """The post-selection clauses, in canonical reading order, each rendered by its
        own component (``None`` when absent)."""
        return [
            GroupedByAssembler(self.ctx).clause(node),
            HavingAssembler(self.ctx).clause(node),
            OrderedByAssembler(self.ctx).clause(node),
        ]

    def _where_clause(self, plan: QueryPlan) -> Optional[VerbFragment]:
        if plan.where_condition is None:
            return None
        return phrase(
            Keywords.SUCH_THAT.as_fragment(), self.ctx.child(plan.where_condition)
        )

    def inline_noun(self, entity) -> VerbFragment:
        """
        Inline-noun form used as a chain root inside an InstantiatedVariable.

        Defers the entity's WHERE condition to the binding scope so the enclosing rule
        can emit it as a *"such that …"* clause after all binding overrides are
        registered.  Used by the chain assembler for Entity-rooted chains.
        """
        seen = self.ctx.context.seen_reference(entity)
        if seen is not None:
            return seen

        entity.build()
        var = entity.selected_variable
        variable_type = getattr(var, "_type_", None)
        type_name = (
            variable_type.__name__ if variable_type else FallbackNouns.ENTITY.text
        )

        self.ctx.context.register_label(entity, type_name)
        self.ctx.context.register_label(var, type_name)

        where_expression = entity._where_expression_
        if where_expression is not None:
            self.ctx.context.defer_constraint(where_expression.condition)

        return phrase(
            Articles.indefinite(type_name), RoleFragment.for_variable(type_name, var)
        )
