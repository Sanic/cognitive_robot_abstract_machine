"""
Tiny fragment constructors shared across verbalization rules.

These replace the per-file ``_word`` / ``_phrase`` / ``_role`` one-liners that were
previously copy-pasted into every rule module.  Prefer the
:class:`~krrood.entity_query_language.verbalization.vocabulary.english` constants
(``Keywords.FIND.as_fragment()`` etc.) for fixed words; use these only for dynamic
text and ad-hoc composition.
"""

from __future__ import annotations

from typing import Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef


def word(text: str) -> WordFragment:
    """Plain, role-less text fragment."""
    return WordFragment(text=text)


def phrase(*parts: VerbFragment, sep: str = " ") -> PhraseFragment:
    """Inline sequence of *parts* joined by *sep* (default single space)."""
    return PhraseFragment(parts=list(parts), separator=sep)


def role(
    text: str, semantic_role: SemanticRole, source_ref: Optional[SourceRef] = None
) -> RoleFragment:
    """Role-tagged (coloured / linkable) text fragment."""
    return RoleFragment(text=text, role=semantic_role, source_ref=source_ref)
