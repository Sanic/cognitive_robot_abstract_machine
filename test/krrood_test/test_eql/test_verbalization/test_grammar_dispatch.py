"""
Standalone unit tests for the grammar dispatch primitive (``select``) and the
EQL-tree fold (``engine.fold``) — exercised with synthetic constructs/rules so
the dispatch mechanics are validated independently of the real grammar.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.verbalization.context import VerbalizationContext
from krrood.entity_query_language.verbalization.engine import (
    fold,
    UnverbalizableExpressionError,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    PhraseFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.grammar.phrase_rule import (
    Ctx,
    PhraseRule,
    select,
)


# Synthetic construct hierarchy (deeper class == more specific).
class Base:
    _id_ = None
    _name_ = "base"


class Mid(Base):
    pass


class Leaf(Mid):
    pass


class Other:
    _id_ = None
    _name_ = "other"


def _ctx():
    """A Ctx for dispatch tests (no recursion needed by select)."""
    return Ctx(child=lambda node: node, context=VerbalizationContext())


def _custom(construct, name, build_fn, *, guard=None, tiebreak=0):
    """Build a flat PhraseRule subclass instance for testing."""
    namespace = {
        "construct": construct,
        "name": name,
        "tiebreak": tiebreak,
        "build": lambda self, node, ctx: build_fn(node, ctx),
    }
    if guard is not None:
        namespace["when"] = lambda self, node, ctx: guard(node)
    return type(f"R_{name}", (PhraseRule,), namespace)()


def _rule(construct, name, **kw):
    """A rule whose build just emits its own name."""
    return _custom(construct, name, lambda node, ctx: WordFragment(name), **kw)


# ── select: specificity ──────────────────────────────────────────────────────


def test_select_prefers_deeper_construct():
    rules = [_rule(Base, "base"), _rule(Mid, "mid"), _rule(Leaf, "leaf")]
    assert select(Leaf(), rules, _ctx()).name == "leaf"
    assert select(Mid(), rules, _ctx()).name == "mid"
    assert select(Base(), rules, _ctx()).name == "base"


def test_select_guarded_beats_unguarded_same_construct():
    rules = [_rule(Mid, "plain"), _rule(Mid, "guarded", guard=lambda n: True)]
    assert select(Mid(), rules, _ctx()).name == "guarded"


def test_select_tiebreak_breaks_same_construct_both_guarded():
    rules = [
        _rule(Mid, "low", guard=lambda n: True, tiebreak=0),
        _rule(Mid, "high", guard=lambda n: True, tiebreak=5),
    ]
    assert select(Mid(), rules, _ctx()).name == "high"


def test_select_guard_can_exclude():
    rules = [_rule(Mid, "only-ok", guard=lambda n: getattr(n, "ok", False))]
    node = Mid()
    assert select(node, rules, _ctx()) is None
    node.ok = True
    assert select(node, rules, _ctx()).name == "only-ok"


def test_select_returns_none_when_nothing_matches():
    rules = [_rule(Mid, "mid")]
    assert select(Other(), rules, _ctx()) is None


# ── fold: dispatch, recursion, override, fallback ────────────────────────────


def test_fold_dispatches_to_selected_rule():
    rules = [_rule(Leaf, "leaf")]
    assert (
        flatten_fragment_to_plain_text(fold(Leaf(), VerbalizationContext(), rules))
        == "leaf"
    )


def test_fold_child_re_enters_the_fold():
    # A parent rule recurses into a child node via ctx.child.
    child = Other()
    parent = Mid()
    rules = [
        _custom(
            Mid,
            "parent",
            lambda node, ctx: PhraseFragment([WordFragment("p"), ctx.child(child)]),
        ),
        _custom(Other, "child", lambda node, ctx: WordFragment("c")),
    ]
    assert (
        flatten_fragment_to_plain_text(fold(parent, VerbalizationContext(), rules))
        == "p c"
    )


def test_fold_binding_override_short_circuits_before_dispatch():
    node = Mid()
    node._id_ = "k"
    context = VerbalizationContext()
    context.binding.binding_overrides["k"] = WordFragment("OVERRIDE")
    # No rules at all — the override must still win.
    assert flatten_fragment_to_plain_text(fold(node, context, [])) == "OVERRIDE"


def test_fold_raises_when_no_rule_covers_the_node():
    node = Mid()
    node._name_ = "uncovered"
    with pytest.raises(UnverbalizableExpressionError):
        fold(node, VerbalizationContext(), [])
