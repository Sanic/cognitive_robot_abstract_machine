"""
Low-level string utilities for the verbalization subsystem.

Pure helpers — fragment flattening, CamelCase splitting, English ordinals,
and safe pluralisation.  No imports from sibling verbalization modules
except for the fragment classes used by :func:`_str` (imported inline to
avoid a circular dependency through ``fragments/base.py``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import inflect

inflect_engine = inflect.engine()

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.fragments.base import VerbFragment


def _str(fragment: VerbFragment) -> str:
    """
    Flatten a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
    to a plain string (no colour markup) for internal comparisons and logging.

    :param fragment: Root of the fragment tree to flatten.
    :type fragment: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
    :returns: Plain-text representation with spaces between tokens.
    :rtype: str
    """
    from krrood.entity_query_language.verbalization.fragments.base import (
        BlockFragment, PhraseFragment, RoleFragment, WordFragment,
    )
    match fragment:
        case WordFragment(text=t):
            return t
        case RoleFragment(text=t):
            return t
        case PhraseFragment(parts=parts, separator=separator):
            return separator.join(_str(p) for p in parts)
        case BlockFragment(header=header, items=items):
            parts_text = ", ".join(_str(i) for i in items)
            if header is None:
                return parts_text
            return f"{_str(header)} {parts_text}" if parts_text else _str(header)
        case _:
            return ""


def _camel_to_words(name: str) -> str:
    """
    Convert a CamelCase class name to space-separated lowercase words.

    :param name: CamelCase identifier string.
    :type name: str
    :returns: Space-separated lowercase words.
    :rtype: str

    Examples::

        _camel_to_words("HasRole")     # → "has role"
        _camel_to_words("IsReachable") # → "is reachable"
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()


def _ordinal(n: int) -> str:
    """
    Convert a zero-based integer index to an ordinal word (e.g. ``0`` → ``"first"``).

    Delegates to the ``inflect`` library for correct English ordinals.

    :param n: Zero-based integer index.
    :type n: int
    :returns: English ordinal word (e.g. ``"first"``, ``"second"``, ``"third"``).
    :rtype: str
    """
    return inflect_engine.ordinal(inflect_engine.number_to_words(n + 1))


def _ensure_plural(word: str) -> str:
    """
    Return *word* in plural form without double-pluralising already-plural words.

    Uses ``inflect.singular_noun`` to detect whether *word* is already plural;
    if so returns it unchanged.

    :param word: English noun in singular or plural form.
    :type word: str
    :returns: Plural form of *word*.
    :rtype: str
    """
    return word if inflect_engine.singular_noun(word) else inflect_engine.plural(word)


