"""
Unit tests for the :class:`HasText` / :class:`HasNumber` fragment mixins — the single definition
of the ``text`` and ``number`` fields, shared by the leaf/phrase fragments instead of being
redeclared on each.

These pin the two properties that justify the mixins: (1) the ``text`` / ``number`` fields are
contributed by the mixins (one definition, reused), and (2) the migration preserves the existing
construction contract — ``text`` stays positional, ``number`` is keyword-only — so no call site
changes.  ``SubjectScope`` deliberately does *not* use ``HasNumber`` (its ``subject_number`` is a
different concept), which is asserted too.
"""

from __future__ import annotations

import dataclasses as dc

from krrood.entity_query_language.verbalization.fragments.base import (
    HasNumber,
    HasText,
    NounPhrase,
    RoleFragment,
    SubjectScope,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole


def _field(cls, name) -> dc.Field:
    return {f.name: f for f in dc.fields(cls)}[name]


def test_text_field_is_defined_once_on_hastext():
    """``text`` is contributed by :class:`HasText` — both text-bearing fragments inherit it,
    none redeclares it."""
    assert HasText in WordFragment.__mro__
    assert HasText in RoleFragment.__mro__
    # The field exists on the concrete classes but is owned by the mixin, not redefined locally.
    assert "text" not in WordFragment.__dict__.get("__annotations__", {})
    assert "text" not in RoleFragment.__dict__.get("__annotations__", {})
    # A non-text fragment does not pick up HasText.
    assert HasText not in NounPhrase.__mro__


def test_number_field_is_defined_once_on_hasnumber():
    """``number`` is contributed by :class:`HasNumber` — every fragment whose own text the
    morphology pass pluralises inherits it, none redeclares it."""
    for cls in (WordFragment, RoleFragment, NounPhrase):
        assert HasNumber in cls.__mro__
        assert "number" not in cls.__dict__.get("__annotations__", {})


def test_number_is_keyword_only_and_text_is_positional():
    """The migration keeps the construction contract: ``text`` positional (so ``WordFragment("x")``
    keeps working), ``number`` keyword-only (so it never disturbs required-field ordering).
    """
    assert _field(WordFragment, "text").kw_only is False
    assert _field(RoleFragment, "text").kw_only is False
    for cls in (WordFragment, RoleFragment, NounPhrase):
        assert _field(cls, "number").kw_only is True
        assert _field(cls, "number").default is Number.SINGULAR


def test_positional_construction_preserved():
    """The existing positional constructions still build (``text`` first), with the inherited
    ``number`` defaulting to SINGULAR."""
    assert WordFragment("the").text == "the"
    assert WordFragment("the").number is Number.SINGULAR
    role = RoleFragment("Robot", SemanticRole.VARIABLE)
    assert (role.text, role.role, role.number) == (
        "Robot",
        SemanticRole.VARIABLE,
        Number.SINGULAR,
    )


def test_number_passed_by_keyword():
    """``number=`` keyword construction reaches the inherited field on every fragment."""
    assert WordFragment(text="x", number=Number.PLURAL).number is Number.PLURAL
    assert (
        RoleFragment(text="a", role=SemanticRole.ATTRIBUTE, number=Number.PLURAL).number
        is Number.PLURAL
    )
    assert (
        NounPhrase(head=WordFragment("x"), number=Number.PLURAL).number is Number.PLURAL
    )


def test_subject_scope_carries_no_number():
    """``SubjectScope`` is identity-only: it carries no grammatical number — neither the shared
    ``HasNumber`` ``number`` nor its own ``subject_number``. The coreference pass derives the
    subject's number from its noun phrase when it walks it, so rules supply none.
    """
    assert HasNumber not in SubjectScope.__mro__
    field_names = {f.name for f in dc.fields(SubjectScope)}
    assert "subject_number" not in field_names
    assert "number" not in field_names
