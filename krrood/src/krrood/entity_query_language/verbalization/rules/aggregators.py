"""
Verbalization rules for aggregation expressions — Count, Sum, Average, Max, Min,
Mode, MultiMode, and CountAll.

:class:`AggregatorRule` handles all standard aggregators via the
``_AGGREGATION_KIND`` lookup table; :class:`CountAllRule` is a more-specific
subclass that renders ``CountAll`` directly as *"count of all"*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from krrood.entity_query_language.operators.aggregators import (
    Aggregator, Count, CountAll, Sum, Average, Max, Min, Mode, MultiMode,
)
from krrood.entity_query_language.verbalization.chain_utils import verbalize_plural
from krrood.entity_query_language.verbalization.fragments.base import PhraseFragment, VerbFragment
from krrood.entity_query_language.verbalization.fragments.factory import phrase
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization.utils import _str
from krrood.entity_query_language.verbalization.vocabulary.english import Aggregations, Articles, Prepositions
from krrood.entity_query_language.verbalization.vocabulary.words import ChildForm

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


_AGGREGATION_KIND: dict[type, Aggregations] = {
    Count:      Aggregations.COUNT,
    Sum:        Aggregations.SUM,
    Average:    Aggregations.AVERAGE,
    Max:        Aggregations.MAX,
    Min:        Aggregations.MIN,
    Mode:       Aggregations.MODE,
    MultiMode:  Aggregations.MULTI_MODE,
}


class AggregatorRule(VerbalizationRule):
    """
    Verbalizes any :class:`~krrood.entity_query_language.operators.aggregators.Aggregator`
    subtype via the ``_AGGREGATION_KIND`` lookup table.

    Produces *"<agg_phrase> <plural_child>"* (e.g. *"sum of tasks"*).
    On second mention inserts *"the"* before the phrase for coreference consistency.
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for any :class:`~krrood.entity_query_language.operators.aggregators.Aggregator`."""
        return isinstance(expr, Aggregator)

    @classmethod
    def transform(cls, expr: Aggregator, ctx: VerbalizationContext, verbalizer: EQLVerbalizer) -> VerbFragment:
        """
        Build *"the <aggregation> <plural_child>"* or *"the <aggregation> of <child>"*.

        All mentions always include the definite article *"the"* — aggregations denote a
        specific computed value and are always definite noun phrases.

        The child form depends on :attr:`~krrood.entity_query_language.verbalization.vocabulary.words.AggregationWord.child_form`:

        * :attr:`~krrood.entity_query_language.verbalization.vocabulary.words.ChildForm.PLURAL` —
          ``verbalize_plural`` produces the plural child, e.g. *"sum of amounts of Xs"*.
        * :attr:`~krrood.entity_query_language.verbalization.vocabulary.words.ChildForm.SINGULAR_OF` —
          the regular chain verbalization is used with an explicit *"of"* separator,
          e.g. *"maximum of the amount of the amount_details of a X"*.

        :param expr: Aggregator expression.
        :param ctx: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :returns: Aggregation phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        agg_kind = _AGGREGATION_KIND[type(expr)]
        agg_word = agg_kind.value
        agg_frag = agg_kind.as_fragment()

        if agg_word.child_form == ChildForm.SINGULAR_OF:
            child_frag = verbalizer.build(expr._child_, ctx)
            result = phrase(Articles.THE.as_fragment(), agg_frag, Prepositions.OF.as_fragment(), child_frag)
        else:
            child_frag = verbalize_plural(expr._child_, ctx, verbalizer.build)
            result = phrase(Articles.THE.as_fragment(), agg_frag, child_frag)

        if expr._id_ not in ctx.seen:
            ctx.seen[expr._id_] = _str(phrase(agg_frag, child_frag))
        return result


class CountAllRule(AggregatorRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.operators.aggregators.CountAll`
    as *"count of all"* (no child expression).

    Takes priority over :class:`AggregatorRule` for ``CountAll`` instances.
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.operators.aggregators.CountAll`."""
        return isinstance(expr, CountAll)

    @classmethod
    def transform(cls, expr: CountAll, ctx: VerbalizationContext, verbalizer: EQLVerbalizer) -> VerbFragment:
        """
        Return the *"count of all"* aggregation fragment directly.

        :param expr: CountAll expression.
        :param ctx: Shared verbalization state (unused).
        :param verbalizer: Parent verbalizer (unused).
        :returns: ``Aggregations.COUNT_ALL.as_fragment()``.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return Aggregations.COUNT_ALL.as_fragment()
