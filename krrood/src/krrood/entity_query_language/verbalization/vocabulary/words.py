"""
Base word-type dataclasses and VocabEnum mixin for EQL verbalization.

PlainWord / RoleWord are the two leaf types.  Fixed-role subclasses of RoleWord
pin _role_ as a ClassVar so callers never re-declare the semantic role.
VocabEnum is a thin Enum mixin that delegates as_fragment() and .text to the
dataclass stored in each member's value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing_extensions import ClassVar

from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (  # noqa: F401  (re-export)
    Glue,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole

# ``Number`` is defined in ``fragments.features`` (below both this lexicon and the fragment
# IR, to avoid a cycle) and re-exported here for the ``for_number`` selectors and importers.


@dataclass(frozen=True)
class PlainWord:
    """
    A neutral word/phrase with no semantic role — renders to
    :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`.
    """

    text: str
    """The raw English text (e.g. ``"of"``, ``"and"``, ``"the"``, ``"ascending"``)."""

    def as_fragment(self) -> WordFragment:
        """
        Convert to a :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`.

        :return: Neutral word fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.WordFragment
        """
        return WordFragment(text=self.text)


@dataclass(frozen=True)
class PunctuationWord(PlainWord):
    """
    A punctuation token carrying an orthographic :class:`Glue` — so the orthography pass
    drops the space adjacent to it and rules emit it as an ordinary, normally-separated token.
    """

    glue: Glue = Glue.NONE
    """Which side hugs its neighbour (``LEFT`` for *","* / *")"*; ``RIGHT`` for *"("*)."""

    def as_fragment(self) -> WordFragment:
        """A role-less :class:`WordFragment` carrying this token's :attr:`glue`."""
        return WordFragment(text=self.text, glue=self.glue)


@dataclass(frozen=True)
class RoleWord:
    """
    A word/phrase carrying a fixed
    :class:`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole`
    — renders to :class:`~krrood.entity_query_language.verbalization.fragments.base.RoleFragment`.

    Subclasses set :attr:`_role_` as a ``ClassVar``; ``ClassVar`` fields are invisible
    to ``@dataclass`` so ``_role_`` never appears in ``__init__``.

    ``frozen=True`` is inherited by subclasses via the ``__setattr__`` guard.

    :cvar _role_: The semantic role for this word type (set by each subclass).
    """

    text: str
    """The raw English text."""

    _role_: ClassVar[SemanticRole]

    def as_fragment(self) -> RoleFragment:
        """
        Convert to a :class:`~krrood.entity_query_language.verbalization.fragments.base.RoleFragment`
        carrying :attr:`_role_`.

        :return: Role-tagged fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.RoleFragment
        """
        return RoleFragment(text=self.text, role=self._role_)


class KeyWord(RoleWord):
    """EQL structure keyword with :attr:`~SemanticRole.KEYWORD` role."""

    _role_ = SemanticRole.KEYWORD


class LogicalWord(RoleWord):
    """Logical connective with :attr:`~SemanticRole.LOGICAL` role."""

    _role_ = SemanticRole.LOGICAL


class ChildForm(Enum):
    """
    How an :class:`AggregationWord` verbalizes its child expression.

    * ``PLURAL`` — the aggregation word already ends with *"of"* (e.g. ``"sum of"``,
      ``"number of"``); the child is rendered in plural form:
      *"sum of amounts of BankTransactions"*.
    * ``SINGULAR_OF`` — the aggregation word has no trailing *"of"* (``"maximum"``,
      ``"minimum"``); a literal *"of"* preposition is inserted and the child is
      rendered via the regular (singular, definite-article) chain verbalization:
      *"maximum of the amount of the amount_details of a BankTransaction"*.
    * ``NONE`` — the aggregation word takes **no** child (``"count of all"``); it renders
      as the bare phrase with no child and no determiner.
    """

    PLURAL = "plural"
    SINGULAR_OF = "singular_of"
    NONE = "none"


@dataclass(frozen=True)
class AggregationWord(RoleWord):
    """
    Aggregation phrase with :attr:`~SemanticRole.AGGREGATION` role.
    """

    _role_ = SemanticRole.AGGREGATION
    child_form: ChildForm = ChildForm.PLURAL
    """Controls how the child expression is verbalized. Defaults to :attr:`ChildForm.PLURAL`."""


class OperatorWord(RoleWord):
    """Comparison operator phrase with :attr:`~SemanticRole.OPERATOR` role."""

    _role_ = SemanticRole.OPERATOR


class PronounWord(RoleWord):
    """
    A coreference pronoun (e.g. *"its"*) standing in for a previously introduced
    variable.  Carries the :attr:`~SemanticRole.VARIABLE` role so it is coloured
    like the variable it refers to.
    """

    _role_ = SemanticRole.VARIABLE


@dataclass(frozen=True)
class OperatorPhrase:
    """
    All eight text variants for one comparison operator, co-located.

    The three boolean flags ``negated``, ``compact``, and ``temporal`` select
    among the eight fields.  Use :meth:`select` to obtain the appropriate
    :class:`OperatorWord` for a given combination.
    """

    standard: str
    """Default form (e.g. ``"is greater than"``)."""

    compact: str
    """Copula-less form used in HAVING clauses (e.g. ``"greater than"``)."""

    negated: str
    """Negated form (e.g. ``"is not greater than"``)."""

    negated_compact: str
    """Negated copula-less form (e.g. ``"not greater than"``)."""

    temporal: str = ""
    """Temporal standard form (e.g. ``"is after"``)."""

    temporal_compact: str = ""
    """Temporal compact form (e.g. ``"after"``)."""

    temporal_negated: str = ""
    """Temporal negated form (e.g. ``"is no later than"``)."""

    temporal_negated_compact: str = ""
    """Temporal negated compact form (e.g. ``"no later than"``)."""

    def select(
        self, *, negated: bool = False, compact: bool = False, temporal: bool = False
    ) -> OperatorWord:
        """
        Select the appropriate :class:`OperatorWord` for the given flag combination.

        Falls back to :attr:`standard` when the corresponding temporal field is empty
        (e.g. for operators without temporal variants).

        :param negated: Use the negated variant.
        :type negated: bool
        :param compact: Use the copula-less variant (for HAVING clauses).
        :type compact: bool
        :param temporal: Use the temporal variant (for datetime comparisons).
        :type temporal: bool
        :return: The appropriate :class:`OperatorWord`.
        :rtype: OperatorWord
        """
        if temporal:
            if negated and compact:
                text = self.temporal_negated_compact
            elif negated:
                text = self.temporal_negated
            elif compact:
                text = self.temporal_compact
            else:
                text = self.temporal
        elif negated and compact:
            text = self.negated_compact
        elif negated:
            text = self.negated
        elif compact:
            text = self.compact
        else:
            text = self.standard
        return OperatorWord(text=text or self.standard)


class VocabEnum(Enum):
    """
    Enum mixin for enums whose member values are :class:`PlainWord` / :class:`RoleWord` instances.

    Delegates :meth:`as_fragment` and :attr:`text` to the dataclass stored in
    ``.value`` so callers never need to write ``.value`` explicitly.

    All namespace enums (:class:`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords`,
    :class:`~krrood.entity_query_language.verbalization.vocabulary.english.Logicals`, etc.)
    inherit this mixin.
    """

    def as_fragment(self) -> VerbFragment:
        """
        Convert the member's value to its
        :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`.

        :return: Fragment representing this vocabulary item.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return self.value.as_fragment()

    @property
    def text(self) -> str:
        """
        The raw English text of this vocabulary item.

        :rtype: str
        """
        return self.value.text
