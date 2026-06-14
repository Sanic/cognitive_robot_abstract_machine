"""
Unit tests for the restriction *placement* taxonomy — the open-closed seam that lets a new
fold target an existing surface slot with one rule class, and fails *loudly* (rather than
silently dropping fragments) when a rule declares a slot the assembler does not surface.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.verbalization.context import MicroplanningServices
from krrood.entity_query_language.verbalization.fragments.base import WordFragment
from krrood.entity_query_language.verbalization.grammar.assembly.restrictions import (
    RestrictionAssembler,
)
from krrood.entity_query_language.verbalization.exceptions import (
    UnplacedRestrictionError,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import RuleContext
from krrood.entity_query_language.verbalization.grammar.planning.query import (
    MatchedRestriction,
    RestrictionPlan,
)
from krrood.entity_query_language.verbalization.grammar.restriction import (
    AttributePredicateRestrictionRule,
    Placement,
    RangeRestrictionRule,
    RestrictionRule,
    SuperlativeRestrictionRule,
)


def _rule_context() -> RuleContext:
    return RuleContext(
        child=lambda node, number=None: node, services=MicroplanningServices()
    )


def test_every_restriction_rule_declares_a_placement():
    """Each self-registered rule fixes where its output lands — the assembler relies on it."""
    expected = {
        RangeRestrictionRule: Placement.WHOSE_GROUP,
        AttributePredicateRestrictionRule: Placement.WHOSE_GROUP,
        SuperlativeRestrictionRule: Placement.SELECTION_MODIFIER,
    }
    for rule in RestrictionRule.alternatives():
        assert rule.placement is expected[rule]


def test_unhandled_placement_raises_loudly():
    """A rule declaring a placement no RestrictionFragments slot surfaces must raise, not drop —
    so adding a Placement member without surfacing it fails fast at the assembler boundary.
    """

    class _MysteryPlacement:  # a hashable placement the assembler does not surface
        name = "MYSTERY"

    # A plain duck-typed rule (NOT a RestrictionRule subclass, so it doesn't self-register and
    # pollute the real registry) — the assembler only reads .placement and calls .render.
    class _MysteryRule:
        placement = _MysteryPlacement()

        @classmethod
        def render(cls, item, subject_variable, context):
            return WordFragment(text="x")

    plan = RestrictionPlan(matched=[MatchedRestriction(_MysteryRule, object())])
    with pytest.raises(UnplacedRestrictionError, match="MYSTERY"):
        RestrictionAssembler(_rule_context()).render(plan, subject=None)
