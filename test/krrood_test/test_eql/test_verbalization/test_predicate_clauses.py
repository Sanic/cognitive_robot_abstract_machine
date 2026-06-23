"""
Tests for the typed predicate-clause vocabulary and its automatic negation.

A predicate states only the affirmative, present-tense clause from typed part-of-speech elements
(``Noun`` / ``Verb`` / ``Copula`` / ``Preposition`` / ``Adjective``); the morphology pass inflects
the verb (agreement) and a wrapping ``Not`` negates it automatically — a verb with do-support, a
copula with suppletion.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from typing_extensions import Any

from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.exceptions import (
    NonFragmentPredicateError,
)
from krrood.entity_query_language.verbalization.example_domain import (
    Department,
    IsReachable,
    Location,
    StaffMember,
    WorksIn,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.conditions.predication import (
    negate_clause,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    Copula,
    Noun,
    Preposition,
    Verb,
)


# ── verb morphology ──────────────────────────────────────────────────────────────


def test_third_person_singular_regular_and_irregular():
    assert morphology.third_person_singular("work") == "works"
    assert morphology.third_person_singular("contain") == "contains"
    assert morphology.third_person_singular("have") == "has"
    assert morphology.third_person_singular("go") == "goes"


def _verb_leaf(lemma: str, *, number: Number = Number.SINGULAR, negated: bool = False):
    return RoleFragment(
        text=lemma, role=SemanticRole.VERB, number=number, negated=negated
    )


def test_morphology_realizes_verb_present_tense():
    assert MorphologyProcessor().rewrite(_verb_leaf("work")).text == "works"
    assert (
        MorphologyProcessor().rewrite(_verb_leaf("work", number=Number.PLURAL)).text
        == "work"
    )


def test_morphology_realizes_verb_do_support_negation():
    assert MorphologyProcessor().rewrite(_verb_leaf("work", negated=True)).text == (
        "does not work"
    )
    assert MorphologyProcessor().rewrite(
        _verb_leaf("work", number=Number.PLURAL, negated=True)
    ).text == "do not work"


def test_morphology_realizes_negated_copula():
    copula = RoleFragment(text="is", role=SemanticRole.OPERATOR, negated=True)
    assert MorphologyProcessor().rewrite(copula).text == "is not"
    plural_copula = RoleFragment(
        text="is", role=SemanticRole.OPERATOR, number=Number.PLURAL, negated=True
    )
    assert MorphologyProcessor().rewrite(plural_copula).text == "are not"


# ── typed clause vocabulary ──────────────────────────────────────────────────────


def test_verb_element_is_a_verb_role_leaf_carrying_the_lemma():
    leaf = Verb("work").as_fragment()
    assert leaf.role is SemanticRole.VERB
    assert leaf.text == "work"


def test_clause_joins_constituents_in_order():
    built = clause(Noun("an Employee"), Verb("work"), Preposition.IN, Noun("a Department"))
    # Pre-morphology the verb leaf still holds the lemma.
    assert flatten_fragment_to_plain_text(built) == "an Employee work in a Department"


# ── negate_clause (feature marking) ──────────────────────────────────────────────


def test_negate_clause_marks_the_verb_head():
    built = clause(Noun("an Employee"), Verb("work"), Preposition.IN, Noun("a Department"))
    negated = negate_clause(built)
    assert negated.parts[1].negated is True


def test_negate_clause_returns_none_without_a_verb_or_copula():
    built = clause(Noun("an Employee"), Noun("a Department"))
    assert negate_clause(built) is None


# ── end-to-end: affirmative and automatically negated predicates ─────────────────


def test_copula_predicate_affirmative_and_negated():
    assert verbalize_expression(IsReachable(variable(Location, []))) == (
        "a Location is reachable"
    )
    assert verbalize_expression(Not(IsReachable(variable(Location, [])))) == (
        "a Location is not reachable"
    )


def test_verb_predicate_affirmative_and_negated_with_do_support():
    employee, department = variable(StaffMember, []), variable(Department, [])
    assert verbalize_expression(WorksIn(employee, department)) == (
        "a StaffMember works in a Department"
    )
    assert verbalize_expression(Not(WorksIn(employee, department))) == (
        "a StaffMember does not work in a Department"
    )


# ── fragments are required ───────────────────────────────────────────────────────


def test_predicate_returning_a_string_template_is_rejected():
    """A hook returning a string (an old-style template) rather than a Fragment is an error."""

    @dataclass(eq=False)
    class SaysHello(Predicate):
        who: Any

        def __call__(self) -> bool:
            return True

        @classmethod
        def _verbalization_fragment_(cls, fields):
            return "{who} says hello"  # a string template — no longer supported

    with pytest.raises(NonFragmentPredicateError):
        verbalize_expression(SaysHello(variable(Location, [])))
