"""
Unit tests for the CoreferenceProcessor — the document-order pass that resolves referring
noun phrases (first / repeat / pronoun) and strips ``SubjectScope`` markers.

Step 0 pins the scaffold: a no-op on referent-free trees, ``SubjectScope`` reduced to its child.
"""

from __future__ import annotations

import uuid

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    SubjectScope,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.coreference_processor import (
    CoreferenceProcessor,
)
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)


def _noun(text: str) -> RoleFragment:
    return RoleFragment(text=text, role=SemanticRole.VARIABLE)


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(CoreferenceProcessor().process(fragment))


def test_noop_on_plain_tree():
    tree = PhraseFragment(parts=[WordFragment(text="Find"), _noun("Robot")])
    assert _text(tree) == "Find Robot"


def test_non_referring_noun_phrase_is_preserved():
    np = NounPhrase(head=_noun("Robot"))
    out = CoreferenceProcessor().process(np)
    assert isinstance(out, NounPhrase)
    assert out.referent_id is None
    assert flatten_fragment_to_plain_text(out.head) == "Robot"


def test_subject_scope_is_reduced_to_its_child():
    scope = SubjectScope(
        subject_id=uuid.uuid4(),
        child=PhraseFragment(parts=[WordFragment(text="its"), _noun("parent")]),
    )
    out = CoreferenceProcessor().process(scope)
    assert not isinstance(out, SubjectScope)
    assert flatten_fragment_to_plain_text(out) == "its parent"


def test_recurses_into_noun_phrase_modifiers():
    np = NounPhrase(
        head=_noun("sum"),
        modifiers=[WordFragment(text="of"), NounPhrase(head=_noun("amounts"))],
    )
    out = CoreferenceProcessor().process(np)
    assert isinstance(out, NounPhrase)
    assert isinstance(out.modifiers[1], NounPhrase)  # nested NP preserved, recursed


# ── referring resolution (first / repeat / numbered) ─────────────────────────────


def _realise(fragment) -> str:
    """Coreference then determiner phase → plain text (the pipeline's first two stages)."""
    resolved = CoreferenceProcessor().process(fragment)
    return flatten_fragment_to_plain_text(DeterminerProcessor().process(resolved))


def test_repeat_mention_is_downgraded_to_definite():
    rid = uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(head=_noun("Robot"), referent_id=rid),  # first → "a Robot"
            WordFragment(text="and"),
            NounPhrase(head=_noun("Robot"), referent_id=rid),  # repeat → "the Robot"
        ]
    )
    assert _realise(tree) == "a Robot and the Robot"


def test_first_mention_modifiers_dropped_on_repeat():
    rid = uuid.uuid4()
    full = NounPhrase(
        head=_noun("Robot"),
        referent_id=rid,
        modifiers=[WordFragment(text="of"), _noun("Cabinet")],  # "a Robot of Cabinet"
    )
    repeat = NounPhrase(head=_noun("Robot"), referent_id=rid)
    tree = PhraseFragment(parts=[full, WordFragment(text="and"), repeat])
    assert _realise(tree) == "a Robot of Cabinet and the Robot"


def test_numbered_referent_never_downgrades():
    rid = uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(
                head=_noun("Robot 2"), definiteness=Definiteness.BARE, referent_id=rid
            ),
            WordFragment(text="and"),
            NounPhrase(
                head=_noun("Robot 2"), definiteness=Definiteness.BARE, referent_id=rid
            ),
        ]
    )
    assert _realise(tree) == "Robot 2 and Robot 2"


def test_distinct_referents_do_not_interfere():
    a, b = uuid.uuid4(), uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(head=_noun("Robot"), referent_id=a),
            WordFragment(text="and"),
            NounPhrase(head=_noun("Cabinet"), referent_id=b),
        ]
    )
    assert _realise(tree) == "a Robot and a Cabinet"
