"""
Grammatical **features** carried by fragments — small typed values the morphology pass
reads to inflect a leaf.

Kept dependency-free (no fragment / vocabulary imports) so it can sit *below* both the
fragment IR (:mod:`~krrood.entity_query_language.verbalization.fragments.base`) and the
lexicon (:mod:`~krrood.entity_query_language.verbalization.vocabulary.words`) without a
cycle.
"""

from __future__ import annotations

from enum import Enum


class Number(Enum):
    """
    Grammatical **number** — the morphological feature a planner decides, an assembler
    *tags* onto a fragment, and the morphology pass *applies* (pluralising the leaf's text).
    """

    SINGULAR = "singular"
    PLURAL = "plural"

    @classmethod
    def of(cls, is_plural: bool) -> "Number":
        """``PLURAL`` when *is_plural* else ``SINGULAR`` (bridges boolean plan features)."""
        return cls.PLURAL if is_plural else cls.SINGULAR
