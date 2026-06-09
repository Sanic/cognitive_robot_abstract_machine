"""
Tiny fragment constructors shared across verbalization rules.

Prefers the :class:`~krrood.entity_query_language.verbalization.vocabulary.english`
constants (``Keywords.FIND.as_fragment()`` etc.) for fixed words; use these only
for dynamic text and ad-hoc composition.
"""

from __future__ import annotations

from typing_extensions import Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef


def word(text: str) -> WordFragment:
    """Plain, role-less text fragment."""
    return WordFragment(text=text)


def phrase(*parts: VerbFragment, separator: str = " ") -> PhraseFragment:
    """Inline sequence of *parts* joined by *separator* (default single space)."""
    return PhraseFragment(parts=list(parts), separator=separator)


def role(
    text: str,
    semantic_role: SemanticRole,
    source_ref: Optional[SourceRef] = None,
    number: Number = Number.SINGULAR,
) -> RoleFragment:
    """Role-tagged (coloured / linkable) text fragment, optionally number-tagged."""
    return RoleFragment(
        text=text, role=semantic_role, source_ref=source_ref, number=number
    )
