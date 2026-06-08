"""
English morphology facade — the single point of access to inflection.

Every call into the ``inflect`` library is concentrated behind this small, named
API.  This mirrors the **MorphologyProcessor** of SimpleNLG (Gatt & Reiter 2009),
which isolates morphological realisation — pluralisation, indefinite-article
selection, ordinals — from content selection and surface layout.  No other
verbalization module imports ``inflect`` directly, so the choice of morphology
engine is a single, replaceable dependency.

Reference:

* Gatt, A. & Reiter, E. (2009), "SimpleNLG: A realisation engine for practical
  applications", *Proceedings of ENLG 2009* — dedicated morphology processor.
"""

from __future__ import annotations

import inflect

#: The one shared ``inflect`` engine for the whole verbalization subsystem.
_engine = inflect.engine()


def plural(word: str) -> str:
    """
    Return the plural form of *word* unconditionally.

    :param word: An English noun (assumed singular).
    :return: The plural form (e.g. ``"Robot"`` → ``"Robots"``).
    :rtype: str
    """
    return _engine.plural(word)


def ensure_plural(word: str) -> str:
    """
    Return *word* in plural form without double-pluralising already-plural words.

    Uses :meth:`inflect.engine.singular_noun` to detect existing plurals.

    :param word: An English noun in either number.
    :return: The plural form of *word*.
    :rtype: str
    """
    return word if _engine.singular_noun(word) else _engine.plural(word)


def is_plural(word: str) -> bool:
    """
    Return ``True`` when *word* is already in plural form.

    :param word: An English noun.
    :rtype: bool
    """
    return bool(_engine.singular_noun(word))


def indefinite_article(following_word: str) -> str:
    """
    Return the indefinite article (``"a"`` / ``"an"``) for *following_word*,
    chosen phonologically (e.g. ``"hour"`` → ``"an"``, ``"robot"`` → ``"a"``).

    :param following_word: The word the article precedes.
    :return: ``"a"`` or ``"an"``.
    :rtype: str
    """
    return _engine.a(following_word).split()[0]


def ordinal(index: int) -> str:
    """
    Return the English ordinal *word* for a zero-based *index* (``0`` → ``"first"``).

    :param index: Zero-based integer index.
    :return: English ordinal word (e.g. ``"first"``, ``"second"``).
    :rtype: str
    """
    return _engine.ordinal(_engine.number_to_words(index + 1))
