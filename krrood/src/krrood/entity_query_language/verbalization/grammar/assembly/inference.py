from __future__ import annotations

import uuid

from typing_extensions import List, Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    PhraseFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.conditions.verbalizer import (
    ConditionVerbalizer,
)
from krrood.entity_query_language.verbalization.grammar.planning.inference import (
    AggregationStatus,
    AntecedentInformation,
    ConsequentBinding,
    InferencePlanner,
    ConditionPlan,
    RuleStructure,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    ExistentialPhrase,
    FallbackNouns,
    GroupKeyPhrases,
    Keywords,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class InferenceAssembler(Assembler[Entity, RuleStructure]):
    """
    Realise the IF/THEN block from a rule structure.

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    planner = InferencePlanner

    def realize(self, node: Entity, plan: RuleStructure) -> Fragment:
        """
        :param node: The inference-rule query.
        :param plan: The IF/THEN rule structure.
        :return: *"If <antecedents…>, then <consequent…>"* — the two-block IF/THEN form.
        """
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

    @staticmethod
    def _number(antecedent: AntecedentInformation) -> Number:
        """:return: The grammatical number of an antecedent — plural if and only if aggregated."""
        return Number.of(antecedent.aggregation_status == AggregationStatus.AGGREGATED)

    # ── IF clause ───────────────────────────────────────────────────────────────

    def _if_items(self, structure: RuleStructure) -> List[Fragment]:
        """
        :return: One item per antecedent — *"there's a <Type> [whose …]"* — plus any unmatched
            conditions; *"true"* when there are none.
        """
        for antecedent in structure.secondary_antecedents:
            self._register_antecedent(antecedent)

        items: List[Fragment] = []
        for antecedent in structure.primary_antecedents:
            intro = self._antecedent_intro(antecedent)
            self._register_antecedent(antecedent)
            condition_fragments = self._condition_fragments(
                antecedent.conditions, antecedent
            )
            items.append(
                BlockFragment(header=intro, items=condition_fragments)
                if condition_fragments
                else intro
            )

        for condition in structure.unmatched_conditions:
            items.append(self.context.child(condition))

        return items or [Keywords.TRUE.as_fragment()]

    def _antecedent_intro(self, antecedent: AntecedentInformation) -> Fragment:
        """:return: *"there's a <Type>"* / *"there are <Types>"* — the antecedent's existential intro."""
        return ExistentialPhrase.for_number(self._number(antecedent)).build_phrase(
            antecedent.type_name, referent_id=self._antecedent_referent_id(antecedent)
        )

    @staticmethod
    def _antecedent_referent_id(
        antecedent: AntecedentInformation,
    ) -> Optional[uuid.UUID]:
        """
        :return: The antecedent's canonical referent id — the selected variable for an entity
            root, else the root's own id (matching the variable the THEN-clause chains reference).
        """
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            return getattr(root.selected_variable, "_id_", None)
        return getattr(root, "_id_", None)

    def _register_antecedent(self, antecedent: AntecedentInformation) -> None:
        """Mark the antecedent (and its selected variable) introduced, so later mentions in
        the THEN clause read *"the <Type>"*."""
        root = antecedent.root
        self.context.refer.mark_introduced(root)
        if isinstance(root, Entity):
            root.build()
            selected = root.selected_variable
            if selected is not None and hasattr(selected, "_id_"):
                self.context.refer.mark_introduced(selected)

    def _condition_fragments(
        self, conditions: List[ConditionPlan], antecedent: AntecedentInformation
    ) -> List[Fragment]:
        """:return: One fragment per antecedent condition."""
        return [
            self._condition_fragment(condition_plan, antecedent)
            for condition_plan in conditions
        ]

    def _condition_fragment(
        self, condition_plan: ConditionPlan, antecedent: AntecedentInformation
    ) -> Fragment:
        """:return: One rendered condition — a *"whose <attribute> is …"* modifier when foldable, else
        the recursive rendering."""
        if condition_plan.whose_attribute_name is None:
            return self.context.child(condition_plan.expression)
        number = self._number(antecedent)
        value = self._value(condition_plan.expression.right, number)
        return ConditionVerbalizer(self.context).whose_attribute(
            condition_plan.whose_attribute_name, number, value
        )

    def _value(self, expression: SymbolicExpression, number: Number) -> Fragment:
        """:return: *expression* rendered agreeing with *number* (plural folds the chain)."""
        return self.context.child(expression, number=number)

    # ── THEN clause ───────────────────────────────────────────────────────────

    def _then_items(self, structure: RuleStructure) -> List[Fragment]:
        """:return: *"there's a <Consequent> [whose <field> is <value> …]"* — the THEN-clause block."""
        intro: Fragment = ExistentialPhrase.for_number(Number.SINGULAR).build_phrase(
            structure.consequent_type
        )
        binding_fragments = [
            self._binding_fragment(binding) for binding in structure.consequent_bindings
        ]
        if not binding_fragments:
            return [intro]
        return [BlockFragment(header=intro, items=binding_fragments)]

    def _binding_fragment(self, binding: ConsequentBinding) -> Fragment:
        """:return: *"whose <field> is/are <value>"* — one consequent field binding."""
        number = Number.of(binding.is_plural_field)
        return ConditionVerbalizer(self.context).whose_attribute(
            binding.field_name, number, self._binding_value(binding)
        )

    def _binding_value(self, binding: ConsequentBinding) -> Fragment:
        """
        :return: The binding's value: *"the <plural chain>"* (aggregated), bare plural, the
            group-key *"common …"* phrase, or the plain rendering.
        """
        if (
            binding.is_plural_field
            and binding.aggregation_status == AggregationStatus.AGGREGATED
        ):
            return PhraseFragment(
                parts=[
                    Articles.THE.as_fragment(),
                    self.context.child(binding.value_expression, number=Number.PLURAL),
                ]
            )
        if binding.is_plural_field:
            return self.context.child(binding.value_expression, number=Number.PLURAL)
        if binding.aggregation_status == AggregationStatus.GROUP_KEY:
            return self._group_key_value(binding.value_expression)
        return self.context.child(binding.value_expression)

    def _group_key_value(self, expression: SymbolicExpression) -> Fragment:
        """:return: *"the common <field> of the <Roots>"* — a binding that refers to a GROUP BY key."""
        chain, current = walk_chain(expression)
        if not chain or not isinstance(current, Variable):
            return self.context.child(expression)
        root_type = (
            current._type_.__name__
            if getattr(current, "_type_", None)
            else FallbackNouns.ENTITY.text
        )
        self.context.refer.mark_introduced(current)
        parts = build_path_parts(chain)
        field = list(reversed(parts))[0].name if parts else root_type
        return GroupKeyPhrases.COMMON_OF.build_phrase(field, root_type)
