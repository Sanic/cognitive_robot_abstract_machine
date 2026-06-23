from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Union

from krrood.entity_query_language.verbalization.fragments.base import (
    Fragment,
    PhraseFragment,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.vocabulary.english import Copulas
from krrood.entity_query_language.verbalization.vocabulary.words import (
    PlainWord,
    VocabEnum,
)


class ClauseElement(ABC):
    """One typed part-of-speech constituent of a predicate clause.

    A predicate's ``_verbalization_fragment_`` builds its clause from these elements rather than raw
    fragments, so the author writes the affirmative, present-tense form once and the realisation
    passes inflect it (verb agreement, copula suppletion) and negate it (do-support). The element
    only declares *what part of speech* a word is; how it is realised is the morphology pass's job.
    """

    @abstractmethod
    def as_fragment(self) -> Fragment:
        """:return: the fragment this element contributes to the clause."""


@dataclass(frozen=True)
class Noun(ClauseElement):
    """A noun constituent — an already-rendered field fragment, or a literal noun word."""

    content: Union[str, Fragment]
    """The rendered field fragment (passed through), or a literal noun string."""

    def as_fragment(self) -> Fragment:
        """:return: the wrapped fragment, or a word leaf for a literal string.

        >>> Noun("department").as_fragment().text
        'department'
        """
        if isinstance(self.content, Fragment):
            return self.content
        return WordFragment(text=self.content)


@dataclass(frozen=True)
class Verb(ClauseElement):
    """A lexical verb given as its lemma. The morphology pass realises it present-tense
    (*"work"* → *"works"*) and negates it with do-support (*"does not work"*)."""

    lemma: str
    """The verb's base form (*"work"*, *"contain"*, *"love"*)."""

    def as_fragment(self) -> RoleFragment:
        """:return: a ``VERB``-role leaf carrying the lemma for the morphology pass to inflect.

        >>> Verb("work").as_fragment().role
        <SemanticRole.VERB: 'verb'>
        """
        return RoleFragment(text=self.lemma, role=SemanticRole.VERB)


@dataclass(frozen=True)
class Adjective(ClauseElement):
    """A predicative adjective complement after a copula (*"is **reachable**"*)."""

    word: str
    """The adjective's surface word."""

    def as_fragment(self) -> WordFragment:
        """:return: a plain word leaf for the adjective.

        >>> Adjective("reachable").as_fragment().text
        'reachable'
        """
        return WordFragment(text=self.word)


@dataclass(frozen=True)
class Copula(ClauseElement):
    """The copula *"is"* of a predicative clause — realised for number (*"is"* / *"are"*) and
    negation (*"is not"*) by the morphology pass."""

    def as_fragment(self) -> RoleFragment:
        """:return: the affirmative singular copula leaf the morphology pass inflects.

        >>> Copula().as_fragment().text
        'is'
        """
        return Copulas.IS.as_fragment()


class Preposition(VocabEnum):
    """The prepositions a clause links its constituents with (*"works **in** a department"*)."""

    IN = PlainWord("in")
    ON = PlainWord("on")
    OF = PlainWord("of")
    TO = PlainWord("to")
    BY = PlainWord("by")
    AT = PlainWord("at")
    WITH = PlainWord("with")
    FROM = PlainWord("from")


ClauseConstituent = Union[Fragment, ClauseElement, Preposition]


def clause(*constituents: ClauseConstituent) -> PhraseFragment:
    """
    Build a predicate clause from typed part-of-speech constituents.

    A predicate states its affirmative form once — *"<subject> works in <object>"* —
    ``clause(Noun(subject), Verb("work"), Preposition.IN, Noun(object))`` — and the realisation
    passes handle agreement and negation. A raw :class:`Fragment` is accepted too, so a rendered
    field fragment can be dropped in directly.

    :param constituents: The clause's elements in surface order.
    :return: The inline phrase fragment for the clause.

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text, WordFragment,
    ... )
    >>> flatten_fragment_to_plain_text(
    ...     clause(Noun(WordFragment(text="an Employee")), Verb("work"), Preposition.IN,
    ...            Noun(WordFragment(text="a Department")))
    ... )
    'an Employee work in a Department'
    """
    return PhraseFragment(
        parts=[
            constituent if isinstance(constituent, Fragment) else constituent.as_fragment()
            for constituent in constituents
        ]
    )
