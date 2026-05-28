"""
Verbalization of inference-rule queries as ``IF … THEN …`` blocks.

An :class:`~krrood.entity_query_language.query.query.Entity` whose selected variable
is an :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable` encodes
an inference rule.  :class:`InferenceRuleRule` is a more-specific
:class:`~krrood.entity_query_language.verbalization.rules.query.EntityRule` whose
precondition is exactly that shape (at top level), so the rule engine selects it ahead
of the generic ``Find …`` form — no buried ``if`` in the query path.

Structural decomposition lives in
:class:`~krrood.entity_query_language.verbalization.rule_analysis.RuleAnalyzer`; this
module only renders the analysed structure.
"""

from __future__ import annotations

import operator
from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    verbalize_plural,
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, role
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rule_analysis import (
    AggregationStatus,
    AntecedentInfo,
    ConsequentBinding,
    RuleAnalyzer,
    RuleStructure,
)
from krrood.entity_query_language.verbalization.rules.query import TopLevelEntityRule
from krrood.entity_query_language.verbalization.utils import _ensure_plural, inflect_engine
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer

_ANALYZER = RuleAnalyzer()


class InferenceRuleRule(TopLevelEntityRule):
    """
    Verbalizes an inference-rule :class:`~krrood.entity_query_language.query.query.Entity`
    (selected variable is an
    :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`) as an
    ``IF … THEN …`` :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`.

    Precondition (declarative): top-level entity
    (:attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.query_depth`
    ``== 0``) that
    :meth:`~krrood.entity_query_language.verbalization.rule_analysis.RuleAnalyzer.can_handle`.
    Takes priority over :class:`~krrood.entity_query_language.verbalization.rules.query.TopLevelEntityRule`
    via MRO depth; nested inference entities fall through to the noun-phrase form.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for a top-level inference-rule Entity."""
        return (
            isinstance(expression, Entity)
            and context.query_depth == 0
            and _ANALYZER.can_handle(expression)
        )

    @classmethod
    def transform(
        cls, expression: Entity, context: VerbalizationContext, verbalizer: EQLVerbalizer
    ) -> VerbFragment:
        """Build the two-block ``IF … THEN …`` fragment."""
        structure = _ANALYZER.analyze(expression)
        if_frag = _verbalize_rule_if_(structure, context, verbalizer)
        then_frag = _verbalize_rule_then_(structure, context, verbalizer)
        return BlockFragment(
            header=None,
            items=[
                BlockFragment(header=Keywords.IF.as_fragment(), items=if_frag),
                BlockFragment(header=Keywords.THEN.as_fragment(), items=then_frag),
            ],
        )


# ── IF clause ───────────────────────────────────────────────────────────────────


def _verbalize_rule_if_(
    s: RuleStructure, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> list[VerbFragment]:
    """Build the fragments for the IF block: one item per primary antecedent, plus unmatched conditions."""
    for antecedent in s.secondary_antecedents:
        _register_antecedent_(antecedent, context)

    items: list[VerbFragment] = []
    for antecedent in s.primary_antecedents:
        intro = _antecedent_intro_frag_(antecedent)
        _register_antecedent_(antecedent, context)
        cond_frags = _condition_frags_(antecedent.conditions, antecedent, context, verbalizer)
        items.append(
            BlockFragment(header=intro, items=cond_frags) if cond_frags else intro
        )

    for condition in s.unmatched_conditions:
        items.append(verbalizer.build(condition, context))

    return items or [Keywords.TRUE.as_fragment()]


def _antecedent_intro_frag_(antecedent: AntecedentInfo) -> VerbFragment:
    """Return *"there is a <Type>"* or *"there are <Types>"* based on aggregation status."""
    if antecedent.aggregation_status == AggregationStatus.AGGREGATED:
        return ExistentialPhrase.THERE_ARE.build_phrase(antecedent.type_name)
    return ExistentialPhrase.THERE_IS_A.build_phrase(antecedent.type_name)


def _register_antecedent_(antecedent: AntecedentInfo, context: VerbalizationContext) -> None:
    """Mark an antecedent's root and selected variable as seen for coreference tracking."""
    root = antecedent.root
    context.seen[root._id_] = antecedent.type_name
    if isinstance(root, Entity):
        root.build()
        sel = root.selected_variable
        if sel is not None and hasattr(sel, "_id_"):
            context.seen[sel._id_] = antecedent.type_name


def _condition_frags_(
    conditions: list,
    antecedent: AntecedentInfo,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> list[VerbFragment]:
    """Render each condition, preferring a *"whose …"* fold when possible."""
    return [
        _try_whose_from_condition_(condition, antecedent, context, verbalizer) or verbalizer.build(condition, context)
        for condition in conditions
    ]


def _try_whose_from_condition_(
    condition,
    antecedent: AntecedentInfo,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> Optional[VerbFragment]:
    """If *condition* is a foldable single-attribute equality, return a *"whose <attr> is …"* fragment."""
    if not isinstance(condition, Comparator) or condition.operation is not operator.eq:
        return None
    if not isinstance(condition.left, Attribute):
        return None
    attr_names = _extract_attr_names_(condition.left)
    if not attr_names:
        return None
    aggregated = antecedent.aggregation_status == AggregationStatus.AGGREGATED
    attr_word = _ensure_plural(attr_names[-1]) if aggregated else attr_names[-1]
    right_frag = (
        verbalize_plural(condition.right, context, verbalizer.build)
        if aggregated
        else verbalizer.build(condition.right, context)
    )
    return phrase(
        Keywords.WHOSE.as_fragment(),
        role(attr_word, SemanticRole.ATTRIBUTE),
        Copulas.ARE.as_fragment() if aggregated else Copulas.IS.as_fragment(),
        right_frag,
    )


def _extract_attr_names_(left: Attribute) -> list[str]:
    """Walk a MappedVariable chain collecting Attribute names from innermost to outermost."""
    attr_names: list[str] = []
    current = left
    while isinstance(current, MappedVariable):
        if isinstance(current, Attribute):
            attr_names.append(current._attribute_name_)
        current = current._child_
    return attr_names


# ── THEN clause ───────────────────────────────────────────────────────────────


def _verbalize_rule_then_(
    s: RuleStructure, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> list[VerbFragment]:
    """Build the fragment for the THEN block: an intro phrase plus whose-binding items."""
    type_name = s.consequent_type
    intro: VerbFragment = ExistentialPhrase.THERE_IS_A.build_phrase(type_name)
    binding_frags = [
        _verbalize_binding_frag_(b, context, verbalizer) for b in s.consequent_bindings
    ]
    if not binding_frags:
        return [intro]
    return [BlockFragment(header=intro, items=binding_frags)]


def _verbalize_binding_frag_(
    binding: ConsequentBinding,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> VerbFragment:
    """Render a single consequent binding as *"whose <field> is <value>"*."""
    field_text = (
        _ensure_plural(binding.field_name)
        if binding.is_plural_field
        else binding.field_name
    )
    return phrase(
        Keywords.WHOSE.as_fragment(),
        role(field_text, SemanticRole.ATTRIBUTE),
        Copulas.ARE.as_fragment() if binding.is_plural_field else Copulas.IS.as_fragment(),
        _binding_value_frag_(binding, context, verbalizer),
    )


def _binding_value_frag_(
    binding: ConsequentBinding,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> VerbFragment:
    """Render the value part of a consequent binding, handling plural/group-key special cases."""
    if (
        binding.is_plural_field
        and binding.aggregation_status == AggregationStatus.AGGREGATED
    ):
        return phrase(
            Articles.THE.as_fragment(),
            verbalize_plural(binding.value_expression, context, verbalizer.build),
        )
    if binding.is_plural_field:
        return verbalize_plural(binding.value_expression, context, verbalizer.build)
    if binding.aggregation_status == AggregationStatus.GROUP_KEY:
        return _verbalize_group_key_value_(binding.value_expression, context, verbalizer)
    return verbalizer.build(binding.value_expression, context)


def _verbalize_group_key_value_(
    expression, context: VerbalizationContext, verbalizer: EQLVerbalizer
) -> VerbFragment:
    """Render a GROUP KEY value as *"the common <field> of <roots>"*."""
    chain, current = walk_chain(expression)

    if not chain or not isinstance(current, Variable):
        return verbalizer.build(expression, context)

    root_type = (
        current._type_.__name__
        if getattr(current, "_type_", None)
        else FallbackNouns.ENTITY.text
    )
    root_plural = inflect_engine.plural(root_type)
    context.seen[current._id_] = root_type

    parts = build_path_parts(chain)
    field = list(reversed(parts))[0][0] if parts else root_type
    return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_plural)
