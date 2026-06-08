"""
Verbalization context — coreference tracking, disambiguation, and constraint deferral.

:class:`VerbalizationContext` is the mutable state threaded through a single
:meth:`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer.verbalize`
call.  It tracks which variables have been mentioned (so the second occurrence gets
*"the"*), resolves type-name collisions (*"Robot 1"* / *"Robot 2"*), defers
constraints for InstantiatedVariable rendering, and manages pronoun-eligible subjects.
"""

from __future__ import annotations

import contextlib
import datetime
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, List, Optional

from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_source_root,
    selected_aggregator,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Pronouns,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression


class ArticleSelection(Enum):
    """
    Signal from :class:`VerbalizationContext` to the verbalizer: which article form to use.

    :cvar NONE: Numbered variable (e.g. ``Robot 1``) — no article prepended.
    :cvar DEFINITE: Subsequent mention of a single-typed variable → ``"the"``.
    :cvar INDEFINITE: First mention of a single-typed variable → ``"a"`` / ``"an"``.
    """

    NONE = auto()  # numbered variable — no article
    DEFINITE = auto()  # subsequent mention → "the"
    INDEFINITE = auto()  # first mention → "a" / "an"


def _aggregation_source_ids(expression) -> set:
    """
    Return the ``_id_`` of every variable that serves as the source *population*
    of an aggregation sub-query (e.g. the ``BankTransaction`` behind
    ``max(t.amount_details.amount)``).

    Such a variable denotes a population to aggregate over, not a specific
    entity, so it must not consume an entity-disambiguation number — otherwise
    the outer subject would pick up a spurious *"1"* with no matching *"2"*, and
    a constrained aggregation scope would read *"among BankTransaction 2"* rather
    than *"among BankTransactions"*.

    :param expression: Root expression to scan.
    :return: Set of variable ids to exclude from numbering.
    :rtype: set
    """
    ids: set = set()
    for node in expression._all_expressions_:
        if isinstance(node, Entity) and selected_aggregator(node) is not None:
            root = aggregation_source_root(node)
            if root is not None:
                ids.add(root._id_)
    return ids


def _build_disambiguation_map(expression) -> Dict[uuid.UUID, str]:
    """
    Pre-scan *expression* and return a mapping of variable._id_ → display label.

    Types appearing once keep the plain type name; types appearing two or more
    times get "TypeName 1", "TypeName 2", … labels in encounter order.
    Literal nodes are excluded, as are variables that only serve as the source
    population of an aggregation sub-query (see :func:`_aggregation_source_ids`).
    """
    if isinstance(expression, Query):
        expression.build()

    suppressed = _aggregation_source_ids(expression)

    type_to_ids: Dict[str, List[uuid.UUID]] = defaultdict(list)
    seen_ids: set = set()

    for node in expression._all_expressions_:
        if isinstance(node, Variable) and not isinstance(node, Literal):
            if node._id_ in suppressed:
                continue
            type_name = (
                node._type_.__name__
                if getattr(node, "_type_", None)
                else node.__class__.__name__
            )
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
    Carries per-verbalization state: coreference tracking, constraint deferral,
    and binding overrides.

    A single :class:`VerbalizationContext` instance is threaded through an entire
    :meth:`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer.verbalize`
    call.  It ensures the same variable is rendered as ``"a Robot"`` on first
    mention and ``"the Robot"`` on every subsequent mention, and that
    :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
    field-reference fragments are re-used instead of re-verbalized.

    Create via :meth:`from_expression` to pre-load the disambiguation map.
    """

    seen: dict = field(default_factory=dict)
    """Maps expression ``_id_`` → display label for every expression already
    verbalized in this pass."""

    compact_predicates: bool = False
    """When ``True``, comparators omit the copula *"is"* (e.g. *"greater than"*
    instead of *"is greater than"*).  Set temporarily by the HAVING clause
    renderer in :mod:`~krrood.entity_query_language.verbalization.rules.query`."""

    constraint_exprs: List[List[SymbolicExpression]] = field(default_factory=list)
    """Stack of deferred-expression frames.  Each frame belongs to one nesting
    level of
    :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
    verbalization."""

    disambiguation_map: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Maps variable ``_id_`` → display label, pre-computed before verbalization
    begins.  Single-type variables keep the plain type name; colliding types get
    ``"TypeName 1"``, ``"TypeName 2"`` labels."""

    binding_overrides: Dict[uuid.UUID, VerbFragment] = field(default_factory=dict)
    """Maps a child expression's ``_id_`` → a ``VerbFragment`` that substitutes
    for it on subsequent encounters.  Populated by
    ``_verbalize_instantiated_natural`` and checked by
    :meth:`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine.build`
    before any rule dispatches."""

    query_depth: int = 0
    """Number of enclosing query/noun renderings currently on the stack.
    ``0`` means the next :class:`~krrood.entity_query_language.query.query.Entity`
    is the top-level request and is rendered in the imperative
    *"Find … such that …"* form; ``> 0`` means the Entity is nested (a sub-query
    used as a value) and is rendered as a noun phrase instead."""

    coref_subjects: List[uuid.UUID] = field(default_factory=list)
    """Stack of subject variable ``_id_`` s (or ``None`` when the enclosing clause
    has no single coreference subject, e.g. ``SetOf``).  A chain rooted at the
    top-of-stack subject is eligible for pronominalisation — see
    :meth:`pronoun_for`.  Pushed/popped by the entity and instantiated-variable
    verbalizers around the clauses that describe a subject."""

    @classmethod
    def from_expression(cls, expression) -> VerbalizationContext:
        """
        Create a context pre-loaded with a disambiguation map for *expression*.

        Scans the full expression tree to determine which variable type names
        appear more than once, then assigns numbered labels (``"TypeName 1"``,
        ``"TypeName 2"``, …) to disambiguate them.

        :param expression: Root EQL expression or
            :class:`~krrood.entity_query_language.query.query.Query` to scan.
        :return: A fresh :class:`VerbalizationContext` with :attr:`disambiguation_map` populated.
        :rtype: VerbalizationContext
        """
        return cls(disambiguation_map=_build_disambiguation_map(expression))

    def push_constraint_frame(self) -> None:
        """
        Open a new constraint frame for the current
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`.

        All expressions passed to :meth:`defer_constraint` until the matching
        :meth:`pop_constraint_frame` are collected in this frame.
        """
        self.constraint_exprs.append([])

    def pop_constraint_frame(self) -> List[SymbolicExpression]:
        """
        Close the current frame and return its deferred expressions.

        Returns an empty list when no frame is open (defensive behaviour; should
        not occur in well-formed verbalization calls).

        :return: Deferred expressions from the closed frame, in deferral order.
        :rtype: list
        """
        return self.constraint_exprs.pop() if self.constraint_exprs else []

    def defer_constraint(self, expression: SymbolicExpression) -> None:
        """
        Defer *expression* into the top constraint frame.

        No-op when no frame is open (i.e. when verbalization is not currently
        inside an :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
        rendering pass).

        :param expression: EQL expression to defer.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        """
        if self.constraint_exprs:
            self.constraint_exprs[-1].append(expression)

    def push_subject(self, var) -> None:
        """
        Push *var* as the current coreference subject.

        Stores the variable's ``_id_`` when *var* is a single
        :class:`~krrood.entity_query_language.core.variable.Variable`; otherwise
        stores ``None`` so no pronoun fires (e.g. a ``SetOf`` with several subjects).
        Always pushes exactly one frame so callers can pair it with
        :meth:`pop_subject` unconditionally.

        :param var: The subject variable being described, or any non-Variable.
        """
        self.coref_subjects.append(var._id_ if isinstance(var, Variable) else None)

    def pop_subject(self) -> None:
        """Pop the current coreference subject pushed by :meth:`push_subject`."""
        if self.coref_subjects:
            self.coref_subjects.pop()

    @contextlib.contextmanager
    def query_depth_scope(self):
        """Context manager that increments :attr:`query_depth` for the duration of a ``with`` block.

        Usage::

            with context.query_depth_scope():
                ...  # query_depth is incremented here
            # query_depth is restored on exit
        """
        self.query_depth += 1
        try:
            yield
        finally:
            self.query_depth -= 1

    @contextlib.contextmanager
    def compact_predicates_scope(self):
        """Context manager that sets :attr:`compact_predicates` to ``True`` for the duration of a ``with`` block.

        Restores the previous value on exit even when an exception is raised, preventing
        the flag from being stuck ``True`` and corrupting subsequent verbalization passes.

        Usage::

            with context.compact_predicates_scope():
                ...  # compact_predicates is True here
            # compact_predicates is restored on exit
        """
        self.compact_predicates = True
        try:
            yield
        finally:
            self.compact_predicates = False

    @property
    def current_subject_id(self):
        """``_id_`` of the current coreference subject, or ``None`` when there is none."""
        return self.coref_subjects[-1] if self.coref_subjects else None

    def seen_reference(self, expression) -> Optional[VerbFragment]:
        """
        Return *"the <label>"* when *expression* has already been verbalized in this pass,
        else ``None``.

        Centralises the coreference short-circuit that every Entity / nested-noun /
        InstantiatedVariable rendering path performs on re-encountering a variable.

        :param expression: Any expression carrying an ``_id_``.
        :return: The definite-reference phrase, or ``None`` when *expression* is unseen.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        variable_id = getattr(expression, "_id_", None)
        if variable_id is None or variable_id not in self.seen:
            return None
        return PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                RoleFragment(text=self.seen[variable_id], role=SemanticRole.VARIABLE),
            ]
        )

    def pronoun_for(self, root) -> Optional[VerbFragment]:
        """
        Return the possessive-pronoun fragment (*"its"*) for *root* when it is the
        current, unambiguous, already-introduced coreference subject; else ``None``.

        Eligibility (all required):

        * *root* is a :class:`~krrood.entity_query_language.core.variable.Variable`;
        * ``root._id_`` is the top of :attr:`coref_subjects`;
        * *root* is not numbered in :attr:`disambiguation_map` (a numbered variable
          such as ``BankTransaction 2`` would make a pronoun ambiguous);
        * *root* has already been mentioned (is in :attr:`seen`).

        :param root: Candidate chain-root expression.
        :return: The *"its"* fragment, or ``None`` when pronominalisation is unsafe.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        if not isinstance(root, Variable):
            return None
        if root._id_ != self.current_subject_id or root._id_ not in self.seen:
            return None
        type_name = root._type_.__name__ if getattr(root, "_type_", None) else None
        label = self.disambiguation_map.get(root._id_, type_name)
        if type_name is not None and label != type_name:
            return None
        return Pronouns.ITS.as_fragment()

    def noun_for_parts(self, var) -> tuple[ArticleSelection, str]:
        """
        Return ``(ArticleSelection, label)`` for *var*.

        Consults :attr:`disambiguation_map` to determine the display label, then
        :attr:`seen` to determine first vs. subsequent mention.

        * :attr:`~ArticleSelection.NONE` — numbered variable (``"Robot 1"``); no article.
        * :attr:`~ArticleSelection.DEFINITE` — subsequent mention → ``"the"``.
        * :attr:`~ArticleSelection.INDEFINITE` — first mention → ``"a"`` / ``"an"``.

        :param var: A :class:`~krrood.entity_query_language.core.variable.Variable` instance.
        :return: Tuple of ``(ArticleSelection, display_label)``.
        :rtype: tuple
        """
        type_name = (
            var._type_.__name__
            if getattr(var, "_type_", None)
            else var.__class__.__name__
        )
        label = self.disambiguation_map.get(var._id_, type_name)
        is_numbered = label != type_name
        if var._id_ in self.seen:
            return (
                ArticleSelection.NONE if is_numbered else ArticleSelection.DEFINITE
            ), label
        self.seen[var._id_] = label
        return (
            ArticleSelection.NONE if is_numbered else ArticleSelection.INDEFINITE
        ), label

    def type_name_of_value(self, value: Any) -> str:
        """
        Render a Python value as a human-readable string.

        Conversion rules:

        * A bare ``type`` object → its ``__name__`` (e.g. ``Apple`` → ``"Apple"``).
        * A tuple of ``type`` objects → ``"A or B or C"``.
        * A :class:`datetime.datetime` with no time component → ``"May 23, 2026"``.
        * A :class:`datetime.datetime` with a time component → ``"May 23, 2026 at 14:30"``.
        * Anything else → ``repr(value)``.

        :param value: Python value from a
            :class:`~krrood.entity_query_language.core.variable.Literal` node.
        :return: Human-readable string representation.
        :rtype: str
        """
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, tuple) and all(
            isinstance(variable, type) for variable in value
        ):
            return " or ".join(variable.__name__ for variable in value)
        if isinstance(value, datetime.datetime):
            if value.time() == datetime.time.min:
                return value.strftime("%B %-d, %Y")
            return value.strftime("%B %-d, %Y at %H:%M")
        return repr(value)
