from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

import inflect

_engine = inflect.engine()


def _article(type_name: str) -> str:
    return _engine.a(type_name).split()[0]


def _build_disambiguation_map(expr) -> Dict[uuid.UUID, str]:
    """
    Pre-scan *expr* and return a mapping of ``variable._id_`` → display label.

    Types that appear only once keep the plain type name (no numbering).
    Types that appear two or more times get ``"TypeName 1"``, ``"TypeName 2"``, …
    labels assigned in the order their variables are first encountered in the tree.

    ``Literal`` nodes are excluded — they are constants, not named variables.
    """
    from krrood.entity_query_language.core.variable import Variable, Literal
    from krrood.entity_query_language.query.query import Query

    if isinstance(expr, Query):
        expr.build()

    type_to_ids: Dict[str, List[uuid.UUID]] = defaultdict(list)
    seen_ids: set = set()

    for node in expr._all_expressions_:
        if isinstance(node, Variable) and not isinstance(node, Literal):
            type_name = node._type_.__name__ if getattr(node, "_type_", None) else node.__class__.__name__
            if node._id_ not in seen_ids:
                seen_ids.add(node._id_)
                type_to_ids[type_name].append(node._id_)

    result: Dict[uuid.UUID, str] = {}
    for type_name, ids in type_to_ids.items():
        if len(ids) == 1:
            result[ids[0]] = type_name
        else:
            for n, vid in enumerate(ids, 1):
                result[vid] = f"{type_name} {n}"
    return result


@dataclass
class VerbalizationContext:
    """
    Carries per-verbalization state: coreference tracking and chain-flattening utilities.

    Pass a single instance through an entire ``EQLVerbalizer.verbalize()`` call so that
    the same variable object is rendered as "a Robot" on first mention and "the Robot" on
    every subsequent mention.
    """

    seen: dict = field(default_factory=dict)
    """Maps expression UUID → display label for every expression already verbalized."""

    compact_predicates: bool = False
    """When True, comparators omit the copula "is" (e.g. "greater than" not "is greater than").
    Set by the verbalizer while rendering HAVING conditions."""

    deferred_constraints: List[List[str]] = field(default_factory=list)
    """Stack of constraint-text frames. Each frame belongs to one nesting level of
    InstantiatedVariable verbalization. When an Entity is verbalized as an inline noun
    its where-conditions are deferred into the top frame so the enclosing
    InstantiatedVariable can emit them as a 'such that …' clause."""

    disambiguation_map: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Maps variable ``_id_`` → display label, pre-computed before verbalization begins.
    Types with a single variable keep the plain type name; types with multiple variables
    get ``"TypeName 1"``, ``"TypeName 2"``, … labels."""

    @classmethod
    def from_expression(cls, expr) -> "VerbalizationContext":
        """Create a context pre-loaded with a disambiguation map for *expr*."""
        return cls(disambiguation_map=_build_disambiguation_map(expr))

    def push_constraint_frame(self) -> None:
        """Open a new constraint frame for the current InstantiatedVariable."""
        self.deferred_constraints.append([])

    def pop_constraint_frame(self) -> List[str]:
        """Close the current frame and return its accumulated constraint strings."""
        return self.deferred_constraints.pop() if self.deferred_constraints else []

    def add_constraint(self, text: str) -> None:
        """Append *text* to the top constraint frame (no-op when no frame is open)."""
        if self.deferred_constraints:
            self.deferred_constraints[-1].append(text)

    def noun_for(self, var) -> str:
        """
        Return the noun phrase for *var*, applying disambiguation numbering when needed.

        * Single-type variable — first mention: ``"a Robot"`` / ``"an Employee"``;
          subsequent mentions: ``"the Robot"``.
        * Numbered variable (same type as another in the same expression tree) —
          every mention: the bare label ``"Employee 1"`` / ``"Employee 2"`` (no
          article, since the number already disambiguates).
        Also registers *var* in :attr:`seen`.
        """
        type_name = var._type_.__name__ if getattr(var, "_type_", None) else var.__class__.__name__
        label = self.disambiguation_map.get(var._id_, type_name)
        is_numbered = label != type_name
        if var._id_ in self.seen:
            return label if is_numbered else f"the {label}"
        self.seen[var._id_] = label
        return label if is_numbered else f"{_article(label)} {label}"

    def flatten_same_type(self, expr, operator_type) -> List:
        """
        Recursively collect a homogeneous binary chain into a flat list.

        ``AND(AND(a, b), c)`` with ``operator_type=AND`` → ``[a, b, c]``.
        Stops at nodes of a different type, so ``AND(a, OR(b, c))`` yields
        ``[a, OR(b, c)]`` — the inner ``OR`` is left intact.
        """
        if not isinstance(expr, operator_type):
            return [expr]
        left = self.flatten_same_type(expr.left, operator_type)
        right = self.flatten_same_type(expr.right, operator_type)
        return left + right

    def type_name_of_value(self, value: Any) -> str:
        """
        Render a Python value as a readable string.

        * A bare ``type`` object → its ``__name__`` (e.g. ``Apple`` → ``"Apple"``).
        * A tuple of ``type`` objects → ``"A or B or C"``.
        * Anything else → ``repr(value)``.
        """
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, tuple) and all(isinstance(v, type) for v in value):
            return " or ".join(v.__name__ for v in value)
        return repr(value)
