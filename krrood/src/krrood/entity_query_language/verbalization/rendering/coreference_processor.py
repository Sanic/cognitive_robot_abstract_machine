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
from typing import Iterable, List, Optional
import uuid

from krrood.entity_query_language.verbalization.fragments.base import (
    map_structural_children,
    NounPhrase,
    PossessiveChain,
    SubjectScope,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
    pronominal_path,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Pronouns


class CoreferenceProcessor:
    """Resolve every referring :class:`NounPhrase` in document order (first / repeat / pronoun)."""

    def process(
        self,
        fragment: VerbFragment,
        already_seen: Optional[Iterable[uuid.UUID]] = None,
    ) -> VerbFragment:
        """
        Return a new tree with referring NPs resolved and ``SubjectScope`` markers stripped.

        :param already_seen: Referents introduced by *prior* builds sharing the same context
            (so the same expression verbalized twice against one context reads *"a Robot"* then
            *"the Robot"*).  These are treated as already-mentioned before the walk begins.
        """
        self._seen: set[uuid.UUID] = set(already_seen or ())
        self._subject_stack: List[Optional[uuid.UUID]] = []
        return self._walk(fragment)

    def _walk(self, fragment: VerbFragment) -> VerbFragment:
        """Document-order rebuild, threading the accumulating discourse state.

        Only the two coreference-relevant nodes are handled here — a ``SubjectScope`` pushes its
        subject for the extent of its child and is then stripped; a ``NounPhrase`` is resolved.
        Every other structural container is rebuilt by the shared
        :func:`~krrood.entity_query_language.verbalization.fragments.base.map_structural_children`
        (recursing through ``self._walk``), and a leaf is returned unchanged.
        """
        match fragment:
            case SubjectScope(subject_id=subject_id, child=child):
                self._subject_stack.append(subject_id)
                try:
                    return self._walk(child)
                finally:
                    self._subject_stack.pop()
            case NounPhrase():
                return self._noun_phrase(fragment)
            case PossessiveChain():
                return self._possessive_chain(fragment)
            case _:
                rebuilt = map_structural_children(fragment, self._walk)
                return rebuilt if rebuilt is not None else fragment

    def _possessive_chain(self, pc: PossessiveChain) -> VerbFragment:
        """Render a chain as *"its …"* when its root is the current subject, else as the
        possessive *"the … of <root>"* (resolving the root NP for first/subsequent mention).
        """
        if self._pronominalises(pc):
            return pronominal_path(pc.parts, Pronouns.ITS.as_fragment())
        return possessive_path(pc.parts, self._walk(pc.root_fragment))

    def _pronominalises(self, pc: PossessiveChain) -> bool:
        """The chain root is the current, already-introduced, non-numbered subject."""
        if pc.root_referent_id is None or pc.root_referent_id not in self._seen:
            return False
        if not self._subject_stack or self._subject_stack[-1] != pc.root_referent_id:
            return False
        # A numbered root ("Robot 2") renders BARE and is never pronominalised.
        return not (
            isinstance(pc.root_fragment, NounPhrase)
            and pc.root_fragment.definiteness is Definiteness.BARE
        )

    def _noun_phrase(self, np: NounPhrase) -> VerbFragment:
        """Resolve a referring NP (first / repeat) in document order; recurse otherwise.

        Every mention (singular or plural) marks its referent introduced.  Only a **repeat
        singular** mention is downgraded to a definite reference — dropping the first-mention
        modifiers and keeping the head label (*"a Robot, where …"* → *"the Robot"*).  A plural
        mention (*"Robots"*) only introduces the referent (it never carries an article), and a
        ``BARE`` numbered label (*"Robot 2"*) never downgrades.  A non-referring NP is just
        rebuilt around its (recursed) children.
        """
        if np.referent_id is None:
            return self._rebuilt(np)
        repeat = np.referent_id in self._seen
        self._seen.add(np.referent_id)
        downgrade = (
            repeat
            and np.number is Number.SINGULAR
            and np.definiteness is not Definiteness.BARE
        )
        if downgrade:
            return NounPhrase(
                head=self._walk(np.head),
                number=np.number,
                definiteness=Definiteness.DEFINITE,
                referent_id=np.referent_id,
            )
        return self._rebuilt(np)

    def _rebuilt(self, np: NounPhrase) -> NounPhrase:
        """Rebuild *np* with its head and modifiers recursed (document order preserved)."""
        return replace(
            np,
            head=self._walk(np.head),
            modifiers=[self._walk(m) for m in np.modifiers],
        )
