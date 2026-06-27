from __future__ import annotations

import inspect
from abc import ABC

from typing_extensions import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from krrood.utils import recursive_subclasses

_T = TypeVar("_T")


def most_specific(candidates: Sequence[_T], key: Callable[[_T], Any]) -> Optional[_T]:
    """
    :param candidates: Items already filtered to those that apply.
    :param key: Specificity key; the maximum wins.
    :return: The single most-specific candidate by *key*, or ``None`` when empty.

    >>> most_specific(["a", "abc", "ab"], key=len)
    'abc'
    >>> most_specific([], key=len) is None
    True
    """
    return max(candidates, key=key, default=None)


def mro_depth(cls: type) -> int:
    """
    :param cls: A class.
    :return: Its specificity — deeper in the hierarchy ⇒ more specific (a subclass outranks the
        alternative it refines).

    >>> mro_depth(object)
    1
    >>> mro_depth(bool) > mro_depth(int)
    True
    """
    return len(cls.__mro__)


def concrete_subclasses(base: Type[_T]) -> List[Type[_T]]:
    """
    The single subclass-discovery primitive: every concrete (instantiable) transitive subclass of
    *base*, abstract intermediates excluded. Shared by the ``RULES`` registry (over
    :class:`PhraseRule`) and the :class:`SpecificityRule` families (over each family base), so
    discovery is defined once.

    :param base: The family / rule base class.
    :return: Its concrete transitive subclasses.

    This is the low-level primitive doing the walk: it collects the instantiable forms under
    ``RankingForm`` and drops the abstract base — the raw list :meth:`SpecificityRule.alternatives`
    then exposes per family.

    >>> from krrood.entity_query_language.verbalization.grammar.query.ranking import RankingForm
    >>> sorted(rule.__name__ for rule in concrete_subclasses(RankingForm))
    ['AttributeRankedByForm', 'AttributeSuperlativeForm', 'LeadingRankForm']
    """
    return [
        subclass
        for subclass in recursive_subclasses(base)
        if not inspect.isabstract(subclass)
    ]


class SpecificityRule(ABC):
    """
    A guarded alternative selected by specificity: the shared base of the small rule
    registries (restriction folding, restriction-subject resolution, navigation forms).

    An alternative is a **subclass** that implements an ``applies(...)`` guard (its
    signature is the subfamily's concern) and a payload method; alternatives
    *self-register* as the family's concrete subclasses and are ranked by class specificity —
    a more-derived alternative (one that subclasses another and refines its guard) outranks the
    alternative it refines. Alternatives that are not in a subclass relationship must have
    mutually exclusive guards, so at most one applies (there is no other ordering between them).

    This mirrors ``select`` for :class:`PhraseRule`: precedence comes from the class hierarchy
    (genuine subsumption) or from disjoint guards, never from a hand-assigned number.

    Reference: production-rule selection; the systemic-functional "most delicate system wins"
    principle.
    """

    @classmethod
    def alternatives(cls) -> List[Type[SpecificityRule]]:
        """:return: The concrete alternative subclasses of this family (transitive; abstract
        family bases are excluded).

        This is the family-facing view over :func:`concrete_subclasses`: bound to one base, it is the
        candidate set :meth:`most_applicable` ranks — here the three ``RankingForm`` templates.

        >>> from krrood.entity_query_language.verbalization.grammar.query.ranking import RankingForm
        >>> sorted(rule.__name__ for rule in RankingForm.alternatives())
        ['AttributeRankedByForm', 'AttributeSuperlativeForm', 'LeadingRankForm']
        """
        return concrete_subclasses(cls)

    @classmethod
    def most_applicable(cls, *args: Any) -> Optional[Type[SpecificityRule]]:
        """
        *args* are forwarded verbatim to each alternative's ``applies`` classmethod, so
        the subfamily fixes that signature (e.g. ``(item, subject)``).

        :return: The most-specific alternative whose ``applies(*args)`` holds, or ``None``.

        >>> from krrood.entity_query_language.verbalization.grammar.query.ranking import (
        ...     RankingForm, RankingRequest)
        >>> from krrood.entity_query_language.verbalization.grammar.query.planner import (
        ...     RankingPlan, RankingDirection, RankingKeyRelation)
        >>> plan = RankingPlan(limit_number=3, direction=RankingDirection.DESCENDING,
        ...     relation=RankingKeyRelation.ATTRIBUTE, order_key=None)
        >>> RankingForm.most_applicable(RankingRequest(plan=plan)).__name__
        'AttributeRankedByForm'
        """
        applicable = [alt for alt in cls.alternatives() if alt.applies(*args)]
        return most_specific(applicable, key=mro_depth)
