"""
Planner — abstract parent for the **analysis** half of a complex construct's
verbalization.

A planner inspects an EQL node and produces a *plan*: a plain, frozen data record of
the decisions about *what to say* (which clauses are present, the subject, the
bindings, …).  A planner does **not** build fragments, does **not** touch the
:class:`~krrood.entity_query_language.verbalization.context.VerbalizationContext`,
and does **not** recurse — so it is pure and unit-testable without any rendering.

This is the *content / structure* stage of the Reiter & Dale microplanning model,
separated from the realisation stage (the
:class:`~krrood.entity_query_language.verbalization.grammar.assembly.base.Assembler`),
so each side has a single responsibility.

Reference: Reiter, E. & Dale, R. (2000), "Building Natural Language Generation
Systems", CUP — microplanning vs. surface realisation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

N = TypeVar("N")
"""The EQL node type the planner analyses."""

P = TypeVar("P")
"""The plan (data record) the planner produces."""


@dataclass
class Planner(ABC, Generic[N, P]):
    """
    Pure analysis of a single EQL *node* into a plan of type ``P``.

    Concrete planners hold the node and decompose the analysis into **methods**
    (e.g. ``_resolve_subject``, ``_plan_clauses``) rather than free functions, so a
    family's analysis is one cohesive class.
    """

    node: N
    """The EQL expression being analysed."""

    @abstractmethod
    def plan(self) -> P:
        """Return the plan computed from :attr:`node`. Pure — no fragments, no ctx."""
