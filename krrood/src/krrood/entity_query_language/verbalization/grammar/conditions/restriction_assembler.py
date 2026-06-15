from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from typing_extensions import Dict, List, Optional, Union

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_comma,
    PhraseFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import RuleContext
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    RestrictionPlan,
)
from krrood.entity_query_language.verbalization.grammar.conditions.restriction import Placement
from krrood.entity_query_language.verbalization.exceptions import (
    UnplacedRestrictionError,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    RangeFold,
    fragment_for_folded_conjunct,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Keywords,
)


@dataclass(frozen=True)
class RestrictionFragments:
    """The rendered pieces of a subject restriction, for the caller to place.

    Only two pieces, because they go to two different sentence positions: the *modifiers* attach to
    the selection noun, the *residual* becomes a separate clause. The finer superlative-vs-appositive
    split is internal — callers always place those together — so it is not exposed."""

    modifiers: List[Fragment] = field(default_factory=list)
    """The noun-attaching modifiers, in order — superlative selection phrases (*"with the maximum
    amount"*) then the appositive *"whose <grouped>"* — placed right after the selection noun."""

    residual: Optional[Fragment] = None
    """The residual condition for a separate *"such that …"* / *"where …"* clause; the caller picks
    the keyword and position. ``None`` when the WHERE folds entirely into modifiers."""


@dataclass
class RestrictionAssembler:
    """
    Render a subject's WHERE partition into its surface pieces: superlative selection modifiers
    (*"with the maximum <leaf>"*), the appositive *"whose <grouped>"* modifier, and the residual
    *"such that …"* / *"where …"* condition.

    A restriction yields several pieces, not a single fragment, so this is realisation-only but
    not an ``Assembler`` subclass.

    Reference: Reiter & Dale (2000) — content structuring (the WHERE partition is the plan).
    """

    context: RuleContext
    """The per-node context (recursion entry and microplanning services)."""

    def render(
        self, restriction: RestrictionPlan, subject: Variable
    ) -> RestrictionFragments:
        """
        Render each matched conjunct via its rule and place it by the rule's placement, then
        build the residual condition.

        :param restriction: The subject's WHERE partition.
        :param subject: The variable the restriction is on.
        :return: The rendered restriction pieces.
        """
        by_placement: Dict[Placement, List[Fragment]] = defaultdict(list)
        for matched in restriction.matched:
            by_placement[matched.rule.placement].append(
                matched.rule.render(matched.item, subject, self.context)
            )

        superlatives = by_placement.pop(Placement.SELECTION_MODIFIER, [])
        grouped = by_placement.pop(Placement.WHOSE_GROUP, [])
        # Loud, not silent: a rule declaring a placement no slot surfaces is a bug, not a drop.
        if by_placement:
            raise UnplacedRestrictionError(placements=list(by_placement))

        whose = (
            PhraseFragment(
                parts=[
                    Keywords.WHOSE.as_fragment(),
                    oxford_comma(grouped, Conjunctions.AND.as_fragment()),
                ]
            )
            if grouped
            else None
        )
        residual = (
            self._residual(restriction.residual) if restriction.has_residual else None
        )
        modifiers = superlatives + ([whose] if whose is not None else [])
        return RestrictionFragments(modifiers=modifiers, residual=residual)

    def _residual(self, items: List[Union[SymbolicExpression, RangeFold]]) -> Fragment:
        """
        :param items: The residual conjuncts (raw expressions or folded ranges).
        :return: The residual conjuncts joined into one condition.
        """
        parts: List[Fragment] = [
            fragment_for_folded_conjunct(
                item,
                self.context.child,
                compact=self.context.configuration.compact_predicates,
            )
            for item in items
        ]
        if len(parts) == 1:
            return parts[0]
        return oxford_comma(parts, Conjunctions.AND.as_fragment())
