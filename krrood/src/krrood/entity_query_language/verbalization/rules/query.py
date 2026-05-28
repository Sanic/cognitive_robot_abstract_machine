"""
Verbalization rules for Entity, SetOf, and query-body clause assembly.

This module is the single source of truth for query verbalization: the rules own
both the *decision* (which form) and the *rendering* (the fragment tree).  All
clause helpers, noun forms, and aggregation-value rendering that previously lived
in ``EntityVerbalizer`` are now module-level functions called directly by the rules.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import Filter
from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.query.operations import GroupedBy, OrderedBy
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier, The
from krrood.entity_query_language.query.query import Entity, Query, SetOf
from krrood.entity_query_language.verbalization.chain_utils import (
    chain_root,
    verbalize_plural,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, role, word
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.restriction import (
    RestrictionClauseBuilder,
    restriction_subject,
)
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization.rules.aggregators import _AGGREGATION_KIND
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_leaf_attribute,
    aggregation_source_root,
    is_aggregation_subquery,
    is_constrained_query,
    selected_aggregator,
)
from krrood.entity_query_language.verbalization.utils import _str
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    FallbackNouns,
    Keywords,
    Prepositions,
    SortDirections,
)
from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer

_UNSET = object()

# ── Ordered-by / Grouped-by shared helpers ───────────────────────────────────────


def _render_ordered_by(
    ob: OrderedBy, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Render an OrderedBy expression as *"ordered by <var> (ascending|descending)"*."""
    direction_frag = (
        SortDirections.DESCENDING.as_fragment()
        if ob.descending
        else SortDirections.ASCENDING.as_fragment()
    )
    ordered_frag = verbalizer.build(ob.variable, ctx)
    paren_frag = PhraseFragment(
        parts=[word("("), direction_frag, word(")")], separator=""
    )
    return phrase(Keywords.ORDERED_BY.as_fragment(), ordered_frag, paren_frag)


def _render_group_keys(
    variables: list, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Render group-by key expressions as a comma-separated phrase."""
    group_frags = [verbalizer.build(v, ctx) for v in variables]
    return PhraseFragment(parts=group_frags, separator=", ")


# ── Query entry points ──────────────────────────────────────────────────────────


def verbalize_query(
    expr: Entity, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Full query form: *"Find X such that …"*.

    Assembles ``FIND + SUCH THAT + GROUPED BY + HAVING + ORDERED BY`` clauses.
    """
    seen = ctx.seen_reference(expr)
    if seen is not None:
        return seen

    expr.build()

    ctx.query_depth += 1
    try:
        is_the = (
            expr._quantifier_builder_ is not None
            and expr._quantifier_builder_.type is The
        )
        var = expr.selected_variable

        if isinstance(var, Entity):
            return _verbalize_query_body_(
                expr, ctx, verbalizer, as_noun(var, ctx, verbalizer)
            )
        if var is None:
            ctx.seen[expr._id_] = FallbackNouns.ENTITY.text
            return _verbalize_query_body_(
                expr, ctx, verbalizer, FallbackNouns.ENTITY.plural_fragment()
            )

        ctx.push_subject(var)
        try:
            if is_the:
                selected_type = (
                    var._type_.__name__
                    if getattr(var, "_type_", None)
                    else FallbackNouns.ENTITY.text
                )
                ctx.seen[var._id_] = selected_type
                ctx.seen[expr._id_] = selected_type
                selected = phrase(
                    Articles.THE_UNIQUE.as_fragment(),
                    role(selected_type, SemanticRole.VARIABLE),
                )
            else:
                selected = verbalizer.build(var, ctx)
                selected_type = ctx.seen.get(
                    getattr(var, "_id_", None), FallbackNouns.ENTITY.text
                )
                ctx.seen[expr._id_] = selected_type
            selected, where_item = _apply_subject_restrictions_(
                expr, var, selected, ctx, verbalizer
            )
            return _verbalize_query_body_(
                expr, ctx, verbalizer, selected, where_item=where_item
            )
        finally:
            ctx.pop_subject()
    finally:
        ctx.query_depth -= 1


def verbalize_nested(
    expr: Entity, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Noun-phrase form for a nested Entity (never emits *"Find …"*).

    * **Unconstrained aggregation subquery** → collapsed aggregate noun.
    * **Constrained aggregation subquery** → full form preserving the filter.
    * **Any other nested entity** → :func:`as_noun`.
    """
    seen = ctx.seen_reference(expr)
    if seen is not None:
        return seen

    expr.build()

    if is_aggregation_subquery(expr):
        return _verbalize_aggregation_value_(expr, ctx, verbalizer)

    return as_noun(expr, ctx, verbalizer)


def verbalize_set_of(
    expr: SetOf, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Verbalize a SetOf query as *"Find (v1, v2, …) such that …"*."""
    expr.build()
    ctx.query_depth += 1
    try:
        var_frags = [verbalizer.build(v, ctx) for v in expr._selected_variables_]
        vars_phrase = PhraseFragment(parts=var_frags, separator=", ")
        selection = PhraseFragment(
            parts=[word("("), vars_phrase, word(")")], separator=""
        )
        return _verbalize_query_body_(
            expr,
            ctx,
            verbalizer,
            selection,
            find_header=Keywords.FIND_SETS_OF.as_fragment(),
        )
    finally:
        ctx.query_depth -= 1


# ── Noun forms ──────────────────────────────────────────────────────────────────


def as_noun(
    expr: Entity, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Standalone-noun form: *"a Robot where …"* (for nested Entity selectors)."""
    seen = ctx.seen_reference(expr)
    if seen is not None:
        return seen

    expr.build()
    is_the = (
        expr._quantifier_builder_ is not None
        and expr._quantifier_builder_.type is The
    )
    var = expr.selected_variable
    selected_type = (
        var._type_.__name__
        if var and getattr(var, "_type_", None)
        else FallbackNouns.ENTITY.text
    )
    ctx.seen[expr._id_] = selected_type
    if var is not None:
        ctx.seen[var._id_] = selected_type

    if is_the:
        article_noun: VerbFragment = phrase(
            Articles.THE_UNIQUE.as_fragment(),
            RoleFragment.for_variable(selected_type, var),
        )
    else:
        article_noun = phrase(
            Articles.indefinite(selected_type),
            RoleFragment.for_variable(selected_type, var),
        )

    where_expr = expr._where_expression_
    if where_expr is None:
        return article_noun
    ctx.query_depth += 1
    ctx.push_subject(var)
    try:
        restrictions = RestrictionClauseBuilder(verbalizer)
        whose, residual = restrictions.build(var, where_expr.condition, ctx)
    finally:
        ctx.pop_subject()
        ctx.query_depth -= 1
    result = article_noun
    if whose is not None:
        result = phrase(result, whose)
    if residual is not None:
        result = phrase(result, Keywords.WHERE.as_fragment(), residual)
    return result


def as_inline_noun(
    entity: Entity, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Inline-noun form used as a chain root inside an InstantiatedVariable.

    Defers the entity's WHERE condition to
    :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.constraint_exprs`
    so the enclosing rule can emit it as a *"such that …"* clause after all
    binding overrides are registered.
    """
    seen = ctx.seen_reference(entity)
    if seen is not None:
        return seen

    entity.build()
    var = entity.selected_variable
    var_type = getattr(var, "_type_", None)
    type_name = var_type.__name__ if var_type else FallbackNouns.ENTITY.text

    ctx.seen[entity._id_] = type_name
    ctx.seen[var._id_] = type_name

    where_expr = entity._where_expression_
    if where_expr is not None:
        ctx.defer_constraint(where_expr.condition)

    return phrase(
        Articles.indefinite(type_name), RoleFragment.for_variable(type_name, var)
    )


# ── Aggregation sub-query rendering ─────────────────────────────────────────────


def _verbalize_aggregation_value_(
    expr: Entity, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Render an aggregation sub-query as a compact aggregate noun phrase.

    * **Unconstrained** → *"the <aggregation> <leaf>"*
    * **Constrained** → *"the <aggregation> <leaf> among <plural source> such that <filter>"*
    * **No attribute leaf** → falls back to aggregator's own verbose rendering.
    """
    aggregator = selected_aggregator(expr)
    leaf = aggregation_leaf_attribute(expr)
    if leaf is None:
        ctx.query_depth += 1
        try:
            return verbalizer.build(aggregator, ctx)
        finally:
            ctx.query_depth -= 1

    agg_kind = _AGGREGATION_KIND[type(aggregator)]
    plural_leaf = agg_kind.value.child_form == ChildForm.PLURAL
    leaf_frag = RoleFragment.for_attribute(
        leaf._owner_class_, leaf._attribute_name_, plural=plural_leaf
    )
    aggregate = phrase(Articles.THE.as_fragment(), agg_kind.as_fragment(), leaf_frag)

    if aggregator._id_ not in ctx.seen:
        ctx.seen[aggregator._id_] = _str(
            phrase(agg_kind.as_fragment(), leaf_frag)
        )

    if not is_constrained_query(expr):
        return aggregate
    return _aggregation_scope_(expr, aggregate, ctx, verbalizer)


def _aggregation_scope_(
    expr: Entity,
    aggregate: VerbFragment,
    ctx: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> VerbFragment:
    """Append *"among <plural source> such that <filter>"* to a constrained aggregate."""
    source = aggregation_source_root(expr)
    source_frag = (
        verbalize_plural(source, ctx, verbalizer.build)
        if source is not None
        else FallbackNouns.ENTITY.plural_fragment()
    )
    parts = [aggregate, Prepositions.AMONG.as_fragment(), source_frag]

    where_expr = expr._where_expression_
    if where_expr is not None:
        ctx.query_depth += 1
        try:
            restrictions = RestrictionClauseBuilder(verbalizer)
            whose, residual = restrictions.build(source, where_expr.condition, ctx)
        finally:
            ctx.query_depth -= 1
        if whose is not None:
            parts.append(whose)
        if residual is not None:
            parts += [Keywords.SUCH_THAT.as_fragment(), residual]

    having_expr = expr._having_expression_
    if having_expr is not None:
        ctx.compact_predicates = True
        ctx.query_depth += 1
        try:
            having_frag = verbalizer.build(having_expr.condition, ctx)
        finally:
            ctx.query_depth -= 1
            ctx.compact_predicates = False
        parts += [Keywords.HAVING.as_fragment(), having_frag]

    return phrase(*parts)


# ── Subject restriction ─────────────────────────────────────────────────────────


def _apply_subject_restrictions_(
    expr,
    var,
    selected: VerbFragment,
    ctx: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> tuple[VerbFragment, object]:
    """
    Fold a subject's single-hop attribute predicates into a *"whose …"* modifier.
    Returns ``(selected, _UNSET)`` when no grouping applies.
    """
    where_expr = expr._where_expression_
    subject = restriction_subject(expr, var, ctx)
    if where_expr is None or subject is None:
        return selected, _UNSET
    restrictions = RestrictionClauseBuilder(verbalizer)
    whose, residual = restrictions.build(subject, where_expr.condition, ctx)
    if whose is not None:
        selected = phrase(selected, whose)
    where_item = (
        phrase(Keywords.SUCH_THAT.as_fragment(), residual)
        if residual is not None
        else None
    )
    return selected, where_item


# ── Query body assembly ─────────────────────────────────────────────────────────


def _verbalize_query_body_(
    expr,
    ctx: VerbalizationContext,
    verbalizer: EQLVerbalizer,
    selection: VerbFragment,
    where_item=_UNSET,
    find_header: Optional[VerbFragment] = None,
) -> VerbFragment:
    """Assemble the full *"Find <selection> such that … grouped by … having … ordered by …"* block."""
    if find_header is None:
        find_header = Keywords.FIND.as_fragment()
    header = phrase(find_header, selection)
    where = (
        _where_clause(expr, ctx, verbalizer) if where_item is _UNSET else where_item
    )
    clauses = [
        c
        for c in [
            where,
            _grouped_by_clause(expr, ctx, verbalizer),
            _having_clause(expr, ctx, verbalizer),
            _ordered_by_clause(expr, ctx, verbalizer),
        ]
        if c is not None
    ]
    return BlockFragment(header=header, items=clauses)


def _where_clause(
    expr, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"such that <condition>"* fragment, or ``None``."""
    where_expr = expr._where_expression_
    if where_expr is None:
        return None
    return phrase(
        Keywords.SUCH_THAT.as_fragment(), verbalizer.build(where_expr.condition, ctx)
    )


def _grouped_by_clause(
    expr, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"and the <aggregated> are grouped by <keys>"* fragment, or ``None``."""
    grouped_expr = expr._grouped_by_expression_
    if grouped_expr is None or not grouped_expr.variables_to_group_by:
        return None
    group_key_root_ids = _root_var_ids_(grouped_expr.variables_to_group_by)
    groups_phrase = _render_group_keys(
        grouped_expr.variables_to_group_by, ctx, verbalizer
    )
    aggregated_frags = _aggregated_noun_frags_(
        expr, group_key_root_ids, ctx, verbalizer
    )
    if aggregated_frags and not isinstance(expr, SetOf):
        aggregated_phrase = oxford_and(
            aggregated_frags, Conjunctions.AND.as_fragment()
        )
        return phrase(
            Conjunctions.AND.as_fragment(),
            Articles.THE.as_fragment(),
            aggregated_phrase,
            Copulas.ARE.as_fragment(),
            Keywords.GROUPED_BY.as_fragment(),
            groups_phrase,
        )
    return phrase(Keywords.GROUPED_BY.as_fragment(), groups_phrase)


def _having_clause(
    expr, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"having <condition>"* fragment with compact comparators, or ``None``."""
    having_expr = expr._having_expression_
    if having_expr is None:
        return None
    ctx.compact_predicates = True
    having_frag = verbalizer.build(having_expr.condition, ctx)
    ctx.compact_predicates = False
    return phrase(Keywords.HAVING.as_fragment(), having_frag)


def _ordered_by_clause(
    expr, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"ordered by <variable> (ascending|descending)"* fragment, or ``None``."""
    ob = expr._ordered_by_builder_
    if ob is None:
        return None
    return _render_ordered_by(ob, ctx, verbalizer)


# ── Grouping helpers ────────────────────────────────────────────────────────────


def _root_var_ids_(exprs) -> set:
    ids: set = set()
    for e in exprs:
        root = chain_root(e)
        if isinstance(root, Variable):
            ids.add(root._id_)
    return ids


def _aggregated_expressions_(query_expr, group_key_root_ids: set) -> list:
    """Return the list of child expressions that are aggregated (not group keys)."""
    selected_var = (
        query_expr.selected_variable if isinstance(query_expr, Entity) else None
    )
    if isinstance(selected_var, InstantiatedVariable):
        result = []
        for child in selected_var._child_vars_.values():
            root = chain_root(child)
            if not (
                isinstance(root, Variable) and root._id_ in group_key_root_ids
            ):
                result.append(child)
        return result
    if isinstance(query_expr, Query):
        return [
            v
            for v in query_expr._selected_variables_
            if v._id_ not in group_key_root_ids
        ]
    return []


def _aggregated_noun_frags_(
    query_expr, group_key_root_ids: set, ctx: VerbalizationContext, verbalizer: EQLVerbalizer
) -> list[VerbFragment]:
    """Pluralise the aggregated expressions into noun fragments for the grouped-by clause."""
    return [
        verbalize_plural(e, ctx, verbalizer.build)
        for e in _aggregated_expressions_(query_expr, group_key_root_ids)
    ]


# ── Rules ───────────────────────────────────────────────────────────────────────


class EntityRule(VerbalizationRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.query.query.Entity` expressions.

    Uses :func:`verbalize_query` at the top level (:attr:`~VerbalizationContext.query_depth`
    ``== 0``) for the imperative *"Find …"* form, or :func:`verbalize_nested` for a
    nested sub-query used as a value.
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for Entity expressions."""
        return isinstance(expr, Entity)

    @classmethod
    def transform(
        cls,
        expr: Entity,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the imperative *"Find …"* form (top level) or noun phrase (nested)."""
        if ctx.query_depth > 0:
            return verbalize_nested(expr, ctx, verbalizer)
        return verbalize_query(expr, ctx, verbalizer)


class SetOfRule(VerbalizationRule):
    """Verbalizes SetOf expressions as *"Find (v1, v2, …) such that …"*."""

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for SetOf expressions."""
        return isinstance(expr, SetOf)

    @classmethod
    def transform(
        cls,
        expr: SetOf,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the SetOf query via :func:`verbalize_set_of`."""
        return verbalize_set_of(expr, ctx, verbalizer)


class ResultQuantifierRule(VerbalizationRule):
    """
    Transparent wrapper: delegates to the child expression.

    An, The, and other ResultQuantifier subclasses carry selection metadata but add
    no natural-language content.
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for any ResultQuantifier."""
        return isinstance(expr, ResultQuantifier)

    @classmethod
    def transform(
        cls,
        expr: ResultQuantifier,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Unwrap and verbalizer to the child expression."""
        return verbalizer.build(expr._child_, ctx)


class FilterRule(VerbalizationRule):
    """Transparent wrapper: delegates to the filter's condition expression."""

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for any Filter (Where / Having)."""
        return isinstance(expr, Filter)

    @classmethod
    def transform(
        cls,
        expr: Filter,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Delegate to the condition expression."""
        return verbalizer.build(expr.condition, ctx)


class GroupedByRule(VerbalizationRule):
    """Verbalizes GroupedBy as *"grouped by <key1>, <key2>, …"*."""

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for GroupedBy."""
        return isinstance(expr, GroupedBy)

    @classmethod
    def transform(
        cls,
        expr: GroupedBy,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Build *"grouped by <key1>, <key2>, …"*, or *"grouped"* when no keys."""
        if expr.variables_to_group_by:
            return phrase(
                Keywords.GROUPED_BY.as_fragment(),
                _render_group_keys(expr.variables_to_group_by, ctx, verbalizer),
            )
        return Keywords.GROUPED.as_fragment()


class OrderedByRule(VerbalizationRule):
    """Verbalizes OrderedBy as *"ordered by <variable> (ascending|descending)"*."""

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for OrderedBy."""
        return isinstance(expr, OrderedBy)

    @classmethod
    def transform(
        cls,
        expr: OrderedBy,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Build *"ordered by <variable> (ascending|descending)"*."""
        return _render_ordered_by(expr, ctx, verbalizer)
