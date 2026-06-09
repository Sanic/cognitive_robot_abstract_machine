"""
Assembler — abstract parent for the **realisation** half of a complex construct's
verbalization.

An assembler takes a *plan* (from the matching
:class:`~krrood.entity_query_language.verbalization.grammar.planning.base.Planner`)
plus the per-node :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx`,
and builds the :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`.
It owns the things that genuinely cannot be pre-planned: recursion into children
(``self.ctx.child``) and the render-scope mutations (query depth, coreference
subject, compact predicates, constraint deferral).  Realisation sub-steps are
**methods** sharing ``self.ctx`` (no parameter threading), mirroring the existing
microplanning service classes
(:class:`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions`,
…).

Reference: Gatt, A. & Reiter, E. (2009), "SimpleNLG: A realisation engine for
practical applications", ENLG — surface realisation as a dedicated stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import Generic, TypeVar

from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.grammar.phrase_rule import Ctx

P = TypeVar("P")
"""The plan (data record) the assembler realises."""


@dataclass
class Assembler(ABC, Generic[P]):
    """
    Realise a plan of type ``P`` into a
    :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`.

    Holds the :class:`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx`
    so realisation sub-steps (methods) share ``self.ctx`` rather than threading it.
    """

    ctx: Ctx
    """The per-node context (recursion entry + microplanning services)."""

    @abstractmethod
    def assemble(self, node, plan: P) -> VerbFragment:
        """Build the fragment for *node* from its *plan*, recursing via ``self.ctx.child``."""
