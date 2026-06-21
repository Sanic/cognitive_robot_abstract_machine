from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.fragments.base import (
    PhraseFragment,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Keywords,
    Prepositions,
    Pronouns,
)


def _attribute_fragment(step: PathStep) -> RoleFragment:
    """:return: A role-tagged attribute fragment for *step*."""
    return RoleFragment(
        text=step.name,
        role=SemanticRole.ATTRIBUTE,
        source_reference=step.source_reference,
    )


def _genitive_step(step: PathStep, owner_fragment: Fragment) -> Fragment:
    """:return: *"the <attribute> of <owner>"* — one plain (noun) hop wrapping its owner."""
    return PhraseFragment(
        parts=[
            Articles.THE.as_fragment(),
            _attribute_fragment(step),
            Prepositions.OF.as_fragment(),
            owner_fragment,
        ]
    )


def _relative_clause(
    step: PathStep, owner_fragment: Fragment, owner_number: Number = Number.SINGULAR
) -> Fragment:
    """:return: *"the <Type> <preposition> which <owner> is <participle>"* — one relational hop
    wrapping its owner as a relative clause (the preposition pied-piped before *which*: *"the Robot
    to which a Mission is assigned"*). Keeping the owner the verb's subject means the meaning never
    flips for agentive relations (*"the Person by which a Book is owned"*); the copula agrees with
    the owner's *owner_number* (*"it is"* / *"they are"*)."""
    relation = step.relation
    return PhraseFragment(
        parts=[
            Articles.THE.as_fragment(),
            RoleFragment.for_type(relation.value_type),
            WordFragment(text=relation.preposition),
            Keywords.WHICH.as_fragment(),
            owner_fragment,
            Copulas.for_number(owner_number),
            RoleFragment.for_attribute(
                relation.owner_class, step.name, text=relation.participle
            ),
        ]
    )


def possessive_path(parts: List[PathStep], root_fragment: Fragment) -> Fragment:
    """:return: the navigation read out from the root, hop by hop (parts innermost-first) — a plain
    hop as the genitive *"the <attribute> of <owner>"*, a relational hop as the relative clause
    *"the <Type> <prep> which <owner> is <participle>"*. With only plain hops this is the familiar
    *"the <outer> of the <inner> of <root>"*."""
    owner = root_fragment
    for step in parts:
        owner = (
            _relative_clause(step, owner)
            if step.is_relation
            else _genitive_step(step, owner)
        )
    return owner


def pronominal_path(parts: List[PathStep], subject_number: Number) -> Fragment:
    """:return: the navigation read out with the (elided) root pronominalised — *"its attribute"* /
    *"the attribute of its foo"* for plain hops, and the relative clause *"the <Type> <prep> which
    it is <participle>"* for a relational hop (the innermost hop, adjacent to the elided root, takes
    the pronoun: the possessive *its/their* for a genitive, the nominative *it/they* as the verb's
    subject for a relation). Reuses the same hop builders as :func:`possessive_path`.

    :param parts: The chain hops, innermost-first.
    :param subject_number: The discourse subject's number (its/it singular, their/they plural).
    """
    possessive_pronoun = Pronouns.possessive(subject_number).as_fragment()
    if not parts:
        return possessive_pronoun
    nominative_pronoun = Pronouns.nominative(subject_number).as_fragment()
    owner: Fragment = possessive_pronoun
    for index, step in enumerate(parts):
        if index == 0:
            owner = (
                _relative_clause(step, nominative_pronoun, subject_number)
                if step.is_relation
                else PhraseFragment(
                    parts=[possessive_pronoun, _attribute_fragment(step)]
                )
            )
        else:
            owner = (
                _relative_clause(step, owner)
                if step.is_relation
                else _genitive_step(step, owner)
            )
    return owner
