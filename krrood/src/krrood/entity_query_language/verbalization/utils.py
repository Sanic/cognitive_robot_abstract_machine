"""
Low-level string utilities for the verbalization subsystem.

Pure orthographic helpers.  English morphology (pluralisation, articles,
ordinals) lives in
:mod:`~krrood.entity_query_language.verbalization.morphology`; fragment
flattening lives in
:func:`~krrood.entity_query_language.verbalization.fragments.base.flatten_fragment_to_plain_text`.
"""

from __future__ import annotations

import re


def _camel_to_words(name: str) -> str:
    """
    Convert a CamelCase class name to space-separated lowercase words.

    :param name: CamelCase identifier string.
    :type name: str
    :return: Space-separated lowercase words.
    :rtype: str

    Examples::

        _camel_to_words("HasRole")     # → "has role"
        _camel_to_words("IsReachable") # → "is reachable"
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()
