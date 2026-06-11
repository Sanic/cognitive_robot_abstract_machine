"""
Possessive / pronominal chain **surface forms** — the two ways a navigation chain renders.

A chain rooted at the discourse subject reads *"its booking_date"* / *"the amount of its
amount_details"* (pronominal); any other chain reads *"the amount of the amount_details of the
BankTransaction"* (possessive).  The *choice* between them is a coreference decision (is the root
the current subject?), made by the
:class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`;
these are the pure surface-form builders it calls once it has decided.  Both are extracted here so
the builder lives in one place, shared by ``ChainAssembler`` (which still renders the possessive
form directly for predicative chains) and the coreference pass.

The leading genitive *"the"* is an invariant structural article (kept out of the determiner
concord — see the Phase-1 head-NPs-only scope), so it is emitted directly here.
"""

from __future__ import annotations

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Prepositions,
)

PathPart = Tuple[str, Optional[SourceRef]]


def _attr(name: str, source_ref: Optional[SourceRef]) -> RoleFragment:
    """A role-tagged attribute fragment."""
    return RoleFragment(text=name, role=SemanticRole.ATTRIBUTE, source_ref=source_ref)


def possessive_path(parts: List[PathPart], root_fragment: VerbFragment) -> VerbFragment:
    """*"the <inner> of the <outer> of <root>"* (parts iterated innermost-first)."""
    if not parts:
        return root_fragment
    reversed_parts = list(reversed(parts))
    first_name, first_ref = reversed_parts[0]
    fragment_parts: List[VerbFragment] = [
        Articles.THE.as_fragment(),
        _attr(first_name, first_ref),
    ]
    for attribute_name, attribute_reference in reversed_parts[1:]:
        fragment_parts.extend(
            [
                Prepositions.OF_THE.as_fragment(),
                _attr(attribute_name, attribute_reference),
            ]
        )
    fragment_parts.extend([Prepositions.OF.as_fragment(), root_fragment])
    return PhraseFragment(parts=fragment_parts)


def pronominal_path(parts: List[PathPart], pronoun: VerbFragment) -> VerbFragment:
    """*"its attr"* (single hop) or *"the attr of its foo"* (multi-hop)."""
    if not parts:
        return pronoun
    reversed_parts = list(reversed(parts))
    last = len(reversed_parts) - 1
    fragment_parts: List[VerbFragment] = []
    for index, (attribute_name, attribute_reference) in enumerate(reversed_parts):
        attribute_fragment = _attr(attribute_name, attribute_reference)
        if index == 0 and index != last:
            fragment_parts.extend([Articles.THE.as_fragment(), attribute_fragment])
        elif index == 0:  # single attribute → "its booking_date"
            fragment_parts.extend([pronoun, attribute_fragment])
        elif index == last:  # adjacent to the elided root → "of its amount_details"
            fragment_parts.extend(
                [Prepositions.OF.as_fragment(), pronoun, attribute_fragment]
            )
        else:
            fragment_parts.extend(
                [Prepositions.OF_THE.as_fragment(), attribute_fragment]
            )
    return PhraseFragment(parts=fragment_parts)
