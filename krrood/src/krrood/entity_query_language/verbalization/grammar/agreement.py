"""
Noun-phrase **number agreement** — render an expression's noun phrase in a given
:class:`~krrood.entity_query_language.verbalization.fragments.features.Number`.

This is the one helper assemblers call instead of the scattered
``if number is PLURAL: verbalize_plural(...) else child(...)`` checks: plural folds the
expression to a bare plural noun phrase (*"amounts of Robots"* — a structural rebuild that
tags its leaves, inflected later by the morphology pass), singular just recurses.
"""

from __future__ import annotations

from typing_extensions import Any, Callable

from krrood.entity_query_language.verbalization.chain_utils import verbalize_plural
from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
from krrood.entity_query_language.verbalization.fragments.features import Number


def noun_phrase(
    expression,
    number: Number,
    context,
    child: Callable[[Any], VerbFragment],
) -> VerbFragment:
    """Render *expression* as a noun phrase agreeing with *number* (plural folds the chain)."""
    if number is Number.PLURAL:
        return verbalize_plural(expression, context, child)
    return child(expression)
