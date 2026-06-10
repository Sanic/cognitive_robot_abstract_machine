"""
Realisation pipeline — the **one** place the lowering passes and their order are defined.

Both the whole-expression build and the local realisation of an opaque
:class:`~krrood.entity_query_language.predicate.Verbalizable` template need the same
ordered sequence of passes (lower the DP, then apply morphology).  Defining it once here —
rather than re-spelling ``DeterminerProcessor`` → ``MorphologyProcessor`` at each call site —
means the ordering lives in a single location, and a future pass (e.g. a coreference
resolution stage) is inserted in exactly one spot.

Reference: Gatt & Reiter (2009), SimpleNLG — the ordered realisation stages.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.rendering.coreference_processor import (
    CoreferenceProcessor,
)
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)

_COREFERENCE = CoreferenceProcessor()
_DETERMINER = DeterminerProcessor()
_MORPHOLOGY = MorphologyProcessor()


def realize_tree(
    fragment: VerbFragment,
    already_seen: Optional[Iterable[uuid.UUID]] = None,
) -> VerbFragment:
    """Run the ordered realisation passes over *fragment*: coreference resolution → DP lowering
    → morphology.  *already_seen* carries referents introduced by prior builds on a shared
    context (see :meth:`CoreferenceProcessor.process`)."""
    resolved = _COREFERENCE.process(fragment, already_seen=already_seen)
    return _MORPHOLOGY.process(_DETERMINER.process(resolved))


def realize_subtree(fragment: VerbFragment) -> str:
    """
    Fully realise a sub-tree to plain text — the realisation passes, then flatten.

    For an **opaque leaf** (a user :class:`~krrood.entity_query_language.predicate.Verbalizable`
    template that string-formats its children): the template's content is opaque text, so it
    must realise its children *here*, locally, rather than deferring to the global passes.
    """
    return flatten_fragment_to_plain_text(realize_tree(fragment))
