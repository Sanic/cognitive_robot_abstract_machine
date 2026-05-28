"""
Range folding — collapsing a lower-bound and an upper-bound comparison on the
same attribute into a single ``between`` phrase.

``t.booking_date >= lo`` and ``t.booking_date <= hi`` (in either order, joined by
``AND``) fold into ``RangeFold(t.booking_date, lo, hi)``, which both the generic
conjunction rule and the subject-restriction rule render as
*"… is between lo and hi"*.

This module is a pure utility: :func:`fold_range_pairs` / :func:`has_pair` are the
pattern detector and :func:`build_between` is the shared phrase builder.  It must
not import from the verbalizer subsystem files to avoid circular dependencies
(core/operator imports are done lazily).
"""

from __future__ import annotations

import operator as _operator
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, TYPE_CHECKING, Union

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.chain_utils import walk_chain
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    RangePhrases,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression


@dataclass
class RangeFold:
    """A folded lower/upper bound pair on one attribute chain."""

    chain_expr: SymbolicExpression
    """The shared attribute chain (e.g. ``t.booking_date``)."""

    lower_expression: SymbolicExpression
    """The lower-bound value expression (the ``>=`` / ``>`` right operand)."""

    upper_expression: SymbolicExpression
    """The upper-bound value expression (the ``<=`` / ``<`` right operand)."""


class _Bound(Enum):
    """Internal marker for the direction of a bound comparison in range folding."""
    LOWER = auto()
    UPPER = auto()


def _chain_key(expr) -> Optional[tuple]:
    """Hashable identity of a pure attribute chain: ``(root_id, ((name, owner), …))`` or ``None``."""
    if not isinstance(expr, MappedVariable):
        return None
    chain, root = walk_chain(expr)
    if not isinstance(root, Variable):
        return None
    parts = []
    for node in chain:
        if not isinstance(node, Attribute):
            return None  # only pure attribute chains fold cleanly
        parts.append((node._attribute_name_, node._owner_class_))
    return (root._id_, tuple(parts))


def _classify(conjunct) -> Optional[tuple]:
    """Return ``(chain_key, _Bound)`` when *conjunct* is a bound comparison, else ``None``."""
    if not isinstance(conjunct, Comparator):
        return None
    key = _chain_key(conjunct.left)
    if key is None:
        return None
    if conjunct.operation in (_operator.gt, _operator.ge):
        return key, _Bound.LOWER
    if conjunct.operation in (_operator.lt, _operator.le):
        return key, _Bound.UPPER
    return None


def fold_range_pairs(conjuncts: List) -> List[Union[SymbolicExpression, RangeFold]]:
    """
    Fold complementary lower/upper bound comparisons on the same chain into
    :class:`RangeFold` items, preserving the order of everything else.

    Direction (not position) decides which operand is the lower vs upper bound, so
    ``t.x <= hi`` written before ``t.x >= lo`` still yields ``between lo and hi``.

    :param conjuncts: A flat list of conjuncts (e.g. the operands of an ``AND``).
    :returns: A list whose items are either the original expressions or
        :class:`RangeFold` instances.
    :rtype: list
    """
    infos = [_classify(c) for c in conjuncts]
    used: set = set()
    result: List[Union[SymbolicExpression, RangeFold]] = []
    for i, conjunct in enumerate(conjuncts):
        if i in used:
            continue
        info = infos[i]
        if info is None:
            result.append(conjunct)
            continue
        key_i, bound_i = info
        partner = None
        for j in range(i + 1, len(conjuncts)):
            if j in used or infos[j] is None:
                continue
            key_j, bound_j = infos[j]
            if key_j == key_i and bound_j is not bound_i:
                partner = j
                break
        if partner is None:
            result.append(conjunct)
            continue
        used.add(partner)
        lower, upper = (
            (conjunct, conjuncts[partner])
            if bound_i is _Bound.LOWER
            else (conjuncts[partner], conjunct)
        )
        result.append(
            RangeFold(chain_expr=lower.left, lower_expression=lower.right, upper_expression=upper.right)
        )
    return result


def has_pair(conjuncts: List) -> bool:
    """Return ``True`` when :func:`fold_range_pairs` would produce at least one :class:`RangeFold`."""
    return any(isinstance(item, RangeFold) for item in fold_range_pairs(conjuncts))


def build_between(
    left_frag: VerbFragment,
    lo_frag: VerbFragment,
    hi_frag: VerbFragment,
    *,
    compact: bool,
) -> VerbFragment:
    """
    Build *"<left> is between <lo> and <hi>"* (or copula-less *"<left> between …"* when *compact*).

    Bounds are joined with :func:`~krrood.entity_query_language.verbalization.fragments.base.oxford_and`
    to match the codebase's Oxford-comma style (e.g. *"between May 15, 2026, and May 30, 2026"*).

    :param left_frag: The already-rendered left side (a full chain, or a bare attribute).
    :param lo_frag: Rendered lower-bound value.
    :param hi_frag: Rendered upper-bound value.
    :param compact: Drop the copula (for HAVING / post-nominal contexts).
    :returns: The range phrase fragment.
    :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
    """
    op = (RangePhrases.BETWEEN if compact else RangePhrases.IS_BETWEEN).as_fragment()
    bounds = oxford_and([lo_frag, hi_frag], Conjunctions.AND.as_fragment())
    return PhraseFragment(parts=[left_frag, op, bounds])
