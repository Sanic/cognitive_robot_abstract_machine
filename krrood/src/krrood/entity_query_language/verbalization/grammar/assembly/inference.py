"""
Inference-rule **assembler** — realise a :class:`RuleStructure` (from
:class:`~krrood.entity_query_language.verbalization.grammar.planning.inference.InferencePlanner`)
into an ``IF … THEN …`` :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`.

Realisation sub-steps are methods sharing ``self.ctx`` (recursion via ``self.ctx.child``,
coreference via ``self.ctx.refer``).  This is the realisation half of the planner/assembler
split (see :class:`~krrood.entity_query_language.verbalization.grammar.assembly.base.Assembler`).

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

import operator

from typing_extensions import List, Optional

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
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
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.planning.inference import (
    AggregationStatus,
    AntecedentInfo,
    ConsequentBinding,
    RuleStructure,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
)


class InferenceAssembler(Assembler[RuleStructure]):
    """Realise the IF/THEN block from a :class:`RuleStructure`."""

    def assemble(self, node, plan: RuleStructure) -> VerbFragment:
        return BlockFragment(
            header=None,
            items=[
                BlockFragment(
                    header=Keywords.IF.as_fragment(), items=self._if_items(plan)
                ),
                BlockFragment(
                    header=Keywords.THEN.as_fragment(), items=self._then_items(plan)
                ),
            ],
        )

    def _render_plural(self, expression, _context=None) -> VerbFragment:
        """``build_fn`` adapter for :func:`verbalize_plural` — recurses via the fold."""
        return self.ctx.child(expression)

    # ── IF clause ───────────────────────────────────────────────────────────────

    def _if_items(self, s: RuleStructure) -> List[VerbFragment]:
        for antecedent in s.secondary_antecedents:
            self._register_antecedent(antecedent)

        items: List[VerbFragment] = []
        for antecedent in s.primary_antecedents:
            intro = self._antecedent_intro(antecedent)
            self._register_antecedent(antecedent)
            cond_frags = self._condition_frags(antecedent.conditions, antecedent)
            items.append(
                BlockFragment(header=intro, items=cond_frags) if cond_frags else intro
            )

        for condition in s.unmatched_conditions:
            items.append(self.ctx.child(condition))

        return items or [Keywords.TRUE.as_fragment()]

    def _antecedent_intro(self, antecedent: AntecedentInfo) -> VerbFragment:
        if antecedent.aggregation_status == AggregationStatus.AGGREGATED:
            return ExistentialPhrase.THERE_ARE.build_phrase(antecedent.type_name)
        return ExistentialPhrase.THERE_IS_A.build_phrase(antecedent.type_name)

    def _register_antecedent(self, antecedent: AntecedentInfo) -> None:
        root = antecedent.root
        self.ctx.refer.seen[root._id_] = antecedent.type_name
        if isinstance(root, Entity):
            root.build()
            sel = root.selected_variable
            if sel is not None and hasattr(sel, "_id_"):
                self.ctx.refer.seen[sel._id_] = antecedent.type_name

    def _condition_frags(
        self, conditions: list, antecedent: AntecedentInfo
    ) -> List[VerbFragment]:
        return [
            self._try_whose_from_condition(condition, antecedent)
            or self.ctx.child(condition)
            for condition in conditions
        ]

    def _try_whose_from_condition(
        self, condition, antecedent: AntecedentInfo
    ) -> Optional[VerbFragment]:
        if (
            not isinstance(condition, Comparator)
            or condition.operation is not operator.eq
        ):
            return None
        if not isinstance(condition.left, Attribute):
            return None
        attr_names = self._extract_attr_names(condition.left)
        if not attr_names:
            return None
        aggregated = antecedent.aggregation_status == AggregationStatus.AGGREGATED
        attr_word = (
            morphology.ensure_plural(attr_names[-1]) if aggregated else attr_names[-1]
        )
        right_frag = (
            verbalize_plural(condition.right, self.ctx.context, self._render_plural)
            if aggregated
            else self.ctx.child(condition.right)
        )
        return phrase(
            Keywords.WHOSE.as_fragment(),
            role(attr_word, SemanticRole.ATTRIBUTE),
            Copulas.ARE.as_fragment() if aggregated else Copulas.IS.as_fragment(),
            right_frag,
        )

    def _extract_attr_names(self, left: Attribute) -> List[str]:
        attr_names: List[str] = []
        current = left
        while isinstance(current, MappedVariable):
            if isinstance(current, Attribute):
                attr_names.append(current._attribute_name_)
            current = current._child_
        return attr_names

    # ── THEN clause ───────────────────────────────────────────────────────────

    def _then_items(self, s: RuleStructure) -> List[VerbFragment]:
        intro: VerbFragment = ExistentialPhrase.THERE_IS_A.build_phrase(
            s.consequent_type
        )
        binding_frags = [self._binding_frag(b) for b in s.consequent_bindings]
        if not binding_frags:
            return [intro]
        return [BlockFragment(header=intro, items=binding_frags)]

    def _binding_frag(self, binding: ConsequentBinding) -> VerbFragment:
        field_text = (
            morphology.ensure_plural(binding.field_name)
            if binding.is_plural_field
            else binding.field_name
        )
        return phrase(
            Keywords.WHOSE.as_fragment(),
            role(field_text, SemanticRole.ATTRIBUTE),
            (
                Copulas.ARE.as_fragment()
                if binding.is_plural_field
                else Copulas.IS.as_fragment()
            ),
            self._binding_value(binding),
        )

    def _binding_value(self, binding: ConsequentBinding) -> VerbFragment:
        if (
            binding.is_plural_field
            and binding.aggregation_status == AggregationStatus.AGGREGATED
        ):
            return phrase(
                Articles.THE.as_fragment(),
                verbalize_plural(
                    binding.value_expression, self.ctx.context, self._render_plural
                ),
            )
        if binding.is_plural_field:
            return verbalize_plural(
                binding.value_expression, self.ctx.context, self._render_plural
            )
        if binding.aggregation_status == AggregationStatus.GROUP_KEY:
            return self._group_key_value(binding.value_expression)
        return self.ctx.child(binding.value_expression)

    def _group_key_value(self, expression) -> VerbFragment:
        chain, current = walk_chain(expression)
        if not chain or not isinstance(current, Variable):
            return self.ctx.child(expression)
        root_type = (
            current._type_.__name__
            if getattr(current, "_type_", None)
            else FallbackNouns.ENTITY.text
        )
        root_plural = morphology.plural(root_type)
        self.ctx.refer.seen[current._id_] = root_type
        parts = build_path_parts(chain)
        field = list(reversed(parts))[0][0] if parts else root_type
        return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_plural)
