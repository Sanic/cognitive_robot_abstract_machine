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
    ordered_by: OrderedBy, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Render an OrderedBy expression as *"ordered by <var> (ascending|descending)"*."""
    direction_frag = (
        SortDirections.DESCENDING.as_fragment()
        if ordered_by.descending
        else SortDirections.ASCENDING.as_fragment()
    )
    ordered_frag = verbalizer.build(ordered_by.variable, context)
    paren_frag = PhraseFragment(
        parts=[word("("), direction_frag, word(")")], separator=""
    )
    return phrase(Keywords.ORDERED_BY.as_fragment(), ordered_frag, paren_frag)


def _render_group_keys(
    variables: list, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Render group-by key expressions as a comma-separated phrase."""
    group_frags = [verbalizer.build(variable, context) for variable in variables]
    return PhraseFragment(parts=group_frags, separator=", ")


# ── Query entry points ──────────────────────────────────────────────────────────


def verbalize_query(
    expression: Entity, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Full query form: *"Find X such that …"*.

    Assembles ``FIND + SUCH THAT + GROUPED BY + HAVING + ORDERED BY`` clauses.
    """
    seen = context.seen_reference(expression)
    if seen is not None:
        return seen

    expression.build()

    context.query_depth += 1
    try:
        is_the = (
            expression._quantifier_builder_ is not None
            and expression._quantifier_builder_.type is The
        )
        var = expression.selected_variable

        if isinstance(var, Entity):
            return _verbalize_query_body_(
                expression, context, verbalizer, as_noun(var, context, verbalizer)
            )
        if var is None:
            context.seen[expression._id_] = FallbackNouns.ENTITY.text
            return _verbalize_query_body_(
                expression, context, verbalizer, FallbackNouns.ENTITY.plural_fragment()
            )

        context.push_subject(var)
        try:
            if is_the:
                selected_type = (
                    var._type_.__name__
                    if getattr(var, "_type_", None)
                    else FallbackNouns.ENTITY.text
                )
                context.seen[var._id_] = selected_type
                context.seen[expression._id_] = selected_type
                selected = phrase(
                    Articles.THE_UNIQUE.as_fragment(),
                    role(selected_type, SemanticRole.VARIABLE),
                )
            else:
                selected = verbalizer.build(var, context)
                selected_type = context.seen.get(
                    getattr(var, "_id_", None), FallbackNouns.ENTITY.text
                )
                context.seen[expression._id_] = selected_type
            selected, where_item = _apply_subject_restrictions_(
                expression, var, selected, context, verbalizer
            )
            return _verbalize_query_body_(
                expression, context, verbalizer, selected, where_item=where_item
            )
        finally:
            context.pop_subject()
    finally:
        context.query_depth -= 1


def verbalize_nested(
    expression: Entity, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Noun-phrase form for a nested Entity (never emits *"Find …"*).

    * **Unconstrained aggregation subquery** → collapsed aggregate noun.
    * **Constrained aggregation subquery** → full form preserving the filter.
    * **Any other nested entity** → :func:`as_noun`.
    """
    seen = context.seen_reference(expression)
    if seen is not None:
        return seen

    expression.build()

    if is_aggregation_subquery(expression):
        return _verbalize_aggregation_value_(expression, context, verbalizer)

    return as_noun(expression, context, verbalizer)


def verbalize_set_of(
    expression: SetOf, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Verbalize a SetOf query as *"Find (v1, v2, …) such that …"*."""
    expression.build()
    context.query_depth += 1
    try:
        variable_fragments = [verbalizer.build(variable, context) for variable in expression._selected_variables_]
        vars_phrase = PhraseFragment(parts=variable_fragments, separator=", ")
        selection = PhraseFragment(
            parts=[word("("), vars_phrase, word(")")], separator=""
        )
        return _verbalize_query_body_(
            expression,
            context,
            verbalizer,
            selection,
            find_header=Keywords.FIND_SETS_OF.as_fragment(),
        )
    finally:
        context.query_depth -= 1


# ── Noun forms ──────────────────────────────────────────────────────────────────


def as_noun(
    expression: Entity, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Standalone-noun form: *"a Robot where …"* (for nested Entity selectors)."""
    seen = context.seen_reference(expression)
    if seen is not None:
        return seen

    expression.build()
    is_the = (
        expression._quantifier_builder_ is not None
        and expression._quantifier_builder_.type is The
    )
    var = expression.selected_variable
    selected_type = (
        var._type_.__name__
        if var and getattr(var, "_type_", None)
        else FallbackNouns.ENTITY.text
    )
    context.seen[expression._id_] = selected_type
    if var is not None:
        context.seen[var._id_] = selected_type

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

    where_expression = expression._where_expression_
    if where_expression is None:
        return article_noun
    context.query_depth += 1
    context.push_subject(var)
    try:
        restrictions = RestrictionClauseBuilder(verbalizer)
        whose, residual = restrictions.build(var, where_expression.condition, context)
    finally:
        context.pop_subject()
        context.query_depth -= 1
    result = article_noun
    if whose is not None:
        result = phrase(result, whose)
    if residual is not None:
        result = phrase(result, Keywords.WHERE.as_fragment(), residual)
    return result


def as_inline_noun(
    entity: Entity, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Inline-noun form used as a chain root inside an InstantiatedVariable.

    Defers the entity's WHERE condition to
    :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.constraint_exprs`
    so the enclosing rule can emit it as a *"such that …"* clause after all
    binding overrides are registered.
    """
    seen = context.seen_reference(entity)
    if seen is not None:
        return seen

    entity.build()
    var = entity.selected_variable
    variable_type = getattr(var, "_type_", None)
    type_name = variable_type.__name__ if variable_type else FallbackNouns.ENTITY.text

    context.seen[entity._id_] = type_name
    context.seen[var._id_] = type_name

    where_expression = entity._where_expression_
    if where_expression is not None:
        context.defer_constraint(where_expression.condition)

    return phrase(
        Articles.indefinite(type_name), RoleFragment.for_variable(type_name, var)
    )


# ── Aggregation sub-query rendering ─────────────────────────────────────────────


def _verbalize_aggregation_value_(
    expression: Entity, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """
    Render an aggregation sub-query as a compact aggregate noun phrase.

    * **Unconstrained** → *"the <aggregation> <leaf>"*
    * **Constrained** → *"the <aggregation> <leaf> among <plural source> such that <filter>"*
    * **No attribute leaf** → falls back to aggregator's own verbose rendering.
    """
    aggregator = selected_aggregator(expression)
    leaf = aggregation_leaf_attribute(expression)
    if leaf is None:
        context.query_depth += 1
        try:
            return verbalizer.build(aggregator, context)
        finally:
            context.query_depth -= 1

    aggregation_kind = _AGGREGATION_KIND[type(aggregator)]
    plural_leaf = aggregation_kind.value.child_form == ChildForm.PLURAL
    leaf_frag = RoleFragment.for_attribute(
        leaf._owner_class_, leaf._attribute_name_, plural=plural_leaf
    )
    aggregate = phrase(Articles.THE.as_fragment(), aggregation_kind.as_fragment(), leaf_frag)

    if aggregator._id_ not in context.seen:
        context.seen[aggregator._id_] = _str(
            phrase(aggregation_kind.as_fragment(), leaf_frag)
        )

    if not is_constrained_query(expression):
        return aggregate
    return _aggregation_scope_(expression, aggregate, context, verbalizer)


def _aggregation_scope_(
    expression: Entity,
    aggregate: VerbFragment,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> VerbFragment:
    """Append *"among <plural source> such that <filter>"* to a constrained aggregate."""
    source = aggregation_source_root(expression)
    source_frag = (
        verbalize_plural(source, context, verbalizer.build)
        if source is not None
        else FallbackNouns.ENTITY.plural_fragment()
    )
    parts = [aggregate, Prepositions.AMONG.as_fragment(), source_frag]

    where_expression = expression._where_expression_
    if where_expression is not None:
        context.query_depth += 1
        try:
            restrictions = RestrictionClauseBuilder(verbalizer)
            whose, residual = restrictions.build(source, where_expression.condition, context)
        finally:
            context.query_depth -= 1
        if whose is not None:
            parts.append(whose)
        if residual is not None:
            parts += [Keywords.SUCH_THAT.as_fragment(), residual]

    having_expression = expression._having_expression_
    if having_expression is not None:
        context.compact_predicates = True
        context.query_depth += 1
        try:
            having_frag = verbalizer.build(having_expression.condition, context)
        finally:
            context.query_depth -= 1
            context.compact_predicates = False
        parts += [Keywords.HAVING.as_fragment(), having_frag]

    return phrase(*parts)


# ── Subject restriction ─────────────────────────────────────────────────────────


def _apply_subject_restrictions_(
    expression,
    var,
    selected: VerbFragment,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> tuple[VerbFragment, object]:
    """
    Fold a subject's single-hop attribute predicates into a *"whose …"* modifier.
    Returns ``(selected, _UNSET)`` when no grouping applies.
    """
    where_expression = expression._where_expression_
    subject = restriction_subject(expression, var, context)
    if where_expression is None or subject is None:
        return selected, _UNSET
    restrictions = RestrictionClauseBuilder(verbalizer)
    whose, residual = restrictions.build(subject, where_expression.condition, context)
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
    expression,
    context: VerbalizationContext,
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
        _where_clause(expression, context, verbalizer) if where_item is _UNSET else where_item
    )
    clauses = [
        clause
        for clause in [
            where,
            _grouped_by_clause(expression, context, verbalizer),
            _having_clause(expression, context, verbalizer),
            _ordered_by_clause(expression, context, verbalizer),
        ]
        if c is not None
    ]
    return BlockFragment(header=header, items=clauses)


def _where_clause(
    expression, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"such that <condition>"* fragment, or ``None``."""
    where_expression = expression._where_expression_
    if where_expression is None:
        return None
    return phrase(
        Keywords.SUCH_THAT.as_fragment(), verbalizer.build(where_expression.condition, context)
    )


def _grouped_by_clause(
    expression, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"and the <aggregated> are grouped by <keys>"* fragment, or ``None``."""
    grouped_expression = expression._grouped_by_expression_
    if grouped_expression is None or not grouped_expression.variables_to_group_by:
        return None
    group_key_root_ids = _root_var_ids_(grouped_expression.variables_to_group_by)
    groups_phrase = _render_group_keys(
        grouped_expression.variables_to_group_by, context, verbalizer
    )
    aggregated_frags = _aggregated_noun_frags_(
        expression, group_key_root_ids, context, verbalizer
    )
    if aggregated_frags and not isinstance(expression, SetOf):
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
    expression, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"having <condition>"* fragment with compact comparators, or ``None``."""
    having_expression = expression._having_expression_
    if having_expression is None:
        return None
    context.compact_predicates = True
    having_frag = verbalizer.build(having_expression.condition, context)
    context.compact_predicates = False
    return phrase(Keywords.HAVING.as_fragment(), having_frag)


def _ordered_by_clause(
    expression, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> Optional[VerbFragment]:
    """Build the *"ordered by <variable> (ascending|descending)"* fragment, or ``None``."""
    ordered_by = expression._ordered_by_builder_
    if ordered_by is None:
        return None
    return _render_ordered_by(ordered_by, context, verbalizer)


# ── Grouping helpers ────────────────────────────────────────────────────────────


def _root_var_ids_(exprs) -> set:
    ids: set = set()
    for expression in exprs:
        root = chain_root(expression)
        if isinstance(root, Variable):
            ids.add(root._id_)
    return ids


def _aggregated_expressions_(query_expr, group_key_root_ids: set) -> list:
    """Return the list of child expressions that are aggregated (not group keys)."""
    selected_variable = (
        query_expr.selected_variable if isinstance(query_expr, Entity) else None
    )
    if isinstance(selected_variable, InstantiatedVariable):
        result = []
        for child in selected_variable._child_vars_.values():
            root = chain_root(child)
            if not (
                isinstance(root, Variable) and root._id_ in group_key_root_ids
            ):
                result.append(child)
        return result
    if isinstance(query_expr, Query):
        return [
            variable
            for variable in query_expr._selected_variables_
            if variable._id_ not in group_key_root_ids
        ]
    return []


def _aggregated_noun_frags_(
    query_expr, group_key_root_ids: set, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> list[VerbFragment]:
    """Pluralise the aggregated expressions into noun fragments for the grouped-by clause."""
    return [
        verbalize_plural(expression, context, verbalizer.build)
        for expression in _aggregated_expressions_(query_expr, group_key_root_ids)
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
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for Entity expressions."""
        return isinstance(expression, Entity)

    @classmethod
    def transform(
        cls,
        expression: Entity,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the imperative *"Find …"* form (top level) or noun phrase (nested)."""
        if context.query_depth > 0:
            return verbalize_nested(expression, context, verbalizer)
        return verbalize_query(expression, context, verbalizer)


class SetOfRule(VerbalizationRule):
    """Verbalizes SetOf expressions as *"Find (v1, v2, …) such that …"*."""

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for SetOf expressions."""
        return isinstance(expression, SetOf)

    @classmethod
    def transform(
        cls,
        expression: SetOf,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Render the SetOf query via :func:`verbalize_set_of`."""
        return verbalize_set_of(expression, context, verbalizer)


class ResultQuantifierRule(VerbalizationRule):
    """
    Transparent wrapper: delegates to the child expression.

    An, The, and other ResultQuantifier subclasses carry selection metadata but add
    no natural-language content.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for any ResultQuantifier."""
        return isinstance(expression, ResultQuantifier)

    @classmethod
    def transform(
        cls,
        expression: ResultQuantifier,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Unwrap and verbalizer to the child expression."""
        return verbalizer.build(expression._child_, context)


class FilterRule(VerbalizationRule):
    """Transparent wrapper: delegates to the filter's condition expression."""

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for any Filter (Where / Having)."""
        return isinstance(expression, Filter)

    @classmethod
    def transform(
        cls,
        expression: Filter,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Delegate to the condition expression."""
        return verbalizer.build(expression.condition, context)


class GroupedByRule(VerbalizationRule):
    """Verbalizes GroupedBy as *"grouped by <key1>, <key2>, …"*."""

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for GroupedBy."""
        return isinstance(expression, GroupedBy)

    @classmethod
    def transform(
        cls,
        expression: GroupedBy,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Build *"grouped by <key1>, <key2>, …"*, or *"grouped"* when no keys."""
        if expression.variables_to_group_by:
            return phrase(
                Keywords.GROUPED_BY.as_fragment(),
                _render_group_keys(expression.variables_to_group_by, context, verbalizer),
            )
        return Keywords.GROUPED.as_fragment()


class OrderedByRule(VerbalizationRule):
    """Verbalizes OrderedBy as *"ordered by <variable> (ascending|descending)"*."""

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for OrderedBy."""
        return isinstance(expression, OrderedBy)

    @classmethod
    def transform(
        cls,
        expression: OrderedBy,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """Build *"ordered by <variable> (ascending|descending)"*."""
        return _render_ordered_by(expression, context, verbalizer)
