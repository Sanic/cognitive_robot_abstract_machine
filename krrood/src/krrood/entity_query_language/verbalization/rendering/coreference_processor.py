"""
CoreferenceProcessor — the **one** place the discourse (coreference) decision is made.

A referring expression is named differently each time it appears: an indefinite first mention
(*"a Robot"*), a definite subsequent mention (*"the Robot"*), or a pronoun (*"its …"*) when it is
the current discourse subject.  Rules emit the *first-mention* form — a
:class:`~krrood.entity_query_language.verbalization.fragments.base.NounPhrase` tagged with a
``referent_id`` (and construction definiteness + label + modifiers), wrapped where appropriate in a
:class:`~krrood.entity_query_language.verbalization.fragments.base.SubjectScope`.  This pass walks
the finished tree in **document order**, tracking which referents have been introduced and which is
the current subject, and **downgrades** every repeat mention to a definite reference (dropping the
first-mention modifiers, keeping the head label) or a pronoun.

It runs *first* in the realisation pipeline (before the determiner phase), so by the time
``DeterminerProcessor`` runs every NP carries a resolved definiteness and no ``referent_id`` matters
any more, and every ``SubjectScope`` has been replaced by its child.

Reference: Reiter & Dale (2000) — referring-expression generation as a microplanning subtask;
Gatt & Reiter (2009), SimpleNLG — ordered realisation stages.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional
import uuid

from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    NounPhrase,
    PhraseFragment,
    SubjectScope,
    VerbFragment,
)


class CoreferenceProcessor:
    """Resolve every referring :class:`NounPhrase` in document order (first / repeat / pronoun)."""

    def process(self, fragment: VerbFragment) -> VerbFragment:
        """Return a new tree with referring NPs resolved and ``SubjectScope`` markers stripped."""
        self._seen: set[uuid.UUID] = set()
        self._subject_stack: List[Optional[uuid.UUID]] = []
        return self._walk(fragment)

    def _walk(self, fragment: VerbFragment) -> VerbFragment:
        """Document-order rebuild, threading the accumulating discourse state."""
        match fragment:
            case SubjectScope(subject_id=subject_id, child=child):
                self._subject_stack.append(subject_id)
                try:
                    return self._walk(child)
                finally:
                    self._subject_stack.pop()
            case PhraseFragment(parts=parts, separator=separator):
                return PhraseFragment(
                    parts=[self._walk(p) for p in parts], separator=separator
                )
            case BlockFragment(header=header, items=items):
                return BlockFragment(
                    header=None if header is None else self._walk(header),
                    items=[self._walk(i) for i in items],
                )
            case NounPhrase():
                return self._noun_phrase(fragment)
            case _:
                return fragment

    def _noun_phrase(self, np: NounPhrase) -> VerbFragment:
        """Recurse structurally through an NP's head and modifiers (discourse resolution is
        layered on in later steps; a non-referring NP is just rebuilt around its children).
        """
        head = self._walk(np.head)
        modifiers = [self._walk(m) for m in np.modifiers]
        return replace(np, head=head, modifiers=modifiers)
