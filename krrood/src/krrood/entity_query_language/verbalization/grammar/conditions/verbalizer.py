from __future__ import annotations

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.aggregation_kinds import (
    AGGREGATION_KIND,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import Assembler
from krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase import (
    comparator_operator,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    single_hop_attribute,
    superlative_aggregation,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    RangeFold,
    build_between,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Keywords,
    Prepositions,
)
from krrood.entity_query_language.verbalization.vocabulary.words import Number


class ConditionVerbalizer(Assembler[Comparator, None]):
    """
    Render a condition in a requested surface form (predicate / modifier / …) — the single owner
    of every surface form a condition can take.

    A comparator/condition is said differently depending on where it sits: a standalone predicate
    (*"x is greater than 5"*), a post-nominal attribute modifier on a subject (the bare *"<attribute>
    op <value>"* that a *"whose …"* envelope wraps), a range modifier (*"<attribute> is between low and
    high"*), or the inference whose-attribute body (*"<attribute> is <value>"* agreeing in number).

    Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
    """

    def realize(self, node: Comparator, plan: None = None) -> Fragment:
        """
        :param node: The condition (comparator) to render.
        :param plan: Unused (this assembler has no plan).
        :return: The default form — a standalone predicate.
        """
        return self.predicate(node)

    def predicate(self, comparator: Comparator, *, negated: bool = False) -> Fragment:
        """
        :param comparator: The comparator to render.
        :param negated: Whether an outer negation applies.
        :return: The standalone comparator form *"<left> <operator> <right>"*.
        """
        return PhraseFragment(
            parts=[
                self.context.child(comparator.left),
                comparator_operator(comparator, self.context.services, negated=negated),
                self.context.child(comparator.right),
            ]
        )

    def attribute_modifier(self, comparator: Comparator, subject: Variable) -> Fragment:
        """
        :param comparator: The comparator on *subject*'s single-hop attribute.
        :param subject: The subject variable.
        :return: The bare *"<attribute> <operator> <value>"* grouped predicate a *"whose …"* envelope
            wraps.
        """
        attribute = single_hop_attribute(comparator.left, subject)
        return PhraseFragment(
            parts=[
                RoleFragment.for_attribute(
                    attribute._owner_class_, attribute._attribute_name_
                ),
                comparator_operator(comparator, self.context.services, compact=False),
                self.context.child(comparator.right),
            ]
        )

    def superlative_modifier(
        self, comparator: Comparator, subject: Variable
    ) -> Fragment:
        """
        :param comparator: The ``subject.<chain> == max/min(over all <Type>.<chain>)`` comparator.
        :param subject: The subject variable.
        :return: The superlative selection modifier *"with the maximum <leaf>"* / *"with the
            minimum <leaf>"*.
        """
        fold = superlative_aggregation(comparator, subject)
        leaf = fold.aggregator._leaf_attribute_
        return PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                Articles.THE.as_fragment(),
                AGGREGATION_KIND[type(fold.aggregator)].as_fragment(),
                RoleFragment.for_attribute(leaf._owner_class_, leaf._attribute_name_),
            ]
        )

    def range_modifier(self, range_fold: RangeFold, subject: Variable) -> Fragment:
        """
        :param range_fold: The folded lower/upper bound pair on *subject*'s single-hop attribute.
        :param subject: The subject variable.
        :return: The modifier *"<attribute> is between low and high"*.
        """
        attribute = single_hop_attribute(range_fold.chain_expression, subject)
        left = RoleFragment.for_attribute(
            attribute._owner_class_, attribute._attribute_name_
        )
        return build_between(
            left,
            self.context.child(range_fold.lower_expression),
            self.context.child(range_fold.upper_expression),
            compact=False,
        )

    def whose_attribute(
        self, attribute_name: str, number: Number, value: Fragment
    ) -> Fragment:
        """
        The attribute noun and copula agree with *number*.

        :param attribute_name: The attribute's name.
        :param number: The grammatical number the noun and copula agree with.
        :param value: The value fragment (supplied by the caller; it may itself be number-folded).
        :return: The full *"whose <attribute> <copula> <value>"* modifier.
        """
        return PhraseFragment(
            parts=[
                Keywords.WHOSE.as_fragment(),
                self._attribute_noun(attribute_name, number),
                Copulas.for_number(number),
                value,
            ]
        )

    def _attribute_noun(self, name: str, number: Number) -> Fragment:
        """
        :param name: The attribute's name.
        :param number: The grammatical number to tag for inflection.
        :return: A role-tagged attribute noun tagged with *number* for inflection.
        """
        return RoleFragment(text=name, role=SemanticRole.ATTRIBUTE, number=number)
