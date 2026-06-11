"""
OrthographyProcessor — the realisation pass that fixes **spacing around punctuation**.

Rules emit punctuation as ordinary, normally-separated tokens carrying a
:class:`~krrood.entity_query_language.verbalization.fragments.features.Glue` (``,`` / ``)`` hug
the *preceding* token; ``(`` hugs the *following* one).  This pass walks each
:class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment` and regroups
its parts so a glued token has no adjacent separator — so a rule writes
``PhraseFragment(parts=[OPEN_PAREN, x, CLOSE_PAREN])`` (normal spacing) and gets *"(x)"*, instead
of hand-managing ``separator=""``.  Spacing thus lives in one place rather than at every call site.

It is the last realisation pass (after morphology), and so sees only Word/Role/Phrase/Block leaves
(``NounPhrase`` already lowered; ``SubjectScope`` / ``PossessiveChain`` already stripped).

Reference: Reiter & Dale (2000) — *linguistic realisation* (orthography); Gatt & Reiter (2009),
SimpleNLG — the realisation passes.
"""

from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.verbalization.fragments.base import (
    map_structural_children,
    PhraseFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Glue


class OrthographyProcessor:
    """Remove the space adjacent to glued punctuation in every ``PhraseFragment`` (idempotent)."""

    def process(self, fragment: VerbFragment) -> VerbFragment:
        """Return a new tree with punctuation spacing fixed."""
        rebuilt = map_structural_children(fragment, self.process)
        node = rebuilt if rebuilt is not None else fragment
        if isinstance(node, PhraseFragment):
            return PhraseFragment(
                parts=self._apply_glue(node.parts), separator=node.separator
            )
        return node

    def _apply_glue(self, parts: List[VerbFragment]) -> List[VerbFragment]:
        """Regroup *parts* so a ``LEFT`` token hugs the previous part and a ``RIGHT`` token the
        next — each merge is a zero-separator subgroup, so the surrounding separator is dropped.
        """
        out: List[VerbFragment] = []
        # A RIGHT token (e.g. "(") held until its following part arrives, to attach to it.
        pending_right: List[VerbFragment] = []
        for part in parts:
            glue = part.glue if isinstance(part, WordFragment) else Glue.NONE
            if pending_right:  # attach the held "(" to this part
                part = self._merge(pending_right + [part])
                pending_right = []
            if glue is Glue.RIGHT:
                pending_right = [part]
                continue
            if glue is Glue.LEFT and out:  # hug the preceding part
                out[-1] = self._merge([out[-1], part])
            else:
                out.append(part)
        # A trailing RIGHT token with no following part (degenerate) stays as-is.
        out.extend(pending_right)
        return out

    @staticmethod
    def _merge(items: List[VerbFragment]) -> VerbFragment:
        """A zero-separator group of *items* (the single item itself when there is only one)."""
        return (
            items[0] if len(items) == 1 else PhraseFragment(parts=items, separator="")
        )
