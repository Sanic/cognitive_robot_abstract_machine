"""
Verbalization rules for logical quantifiers — ForAll and Exists.

* :class:`ForAllRule` — *"for all <variables>, <condition>"*.
* :class:`ExistsRule` — *"there exists <variable> such that <condition>"*.

Both inherit from :class:`QuantifierRule`, the abstract base that catches any
:class:`~krrood.entity_query_language.operators.logical_quantifiers.QuantifiedConditional`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll, QuantifiedConditional
from krrood.entity_query_language.verbalization.chain_utils import verbalize_plural
from krrood.entity_query_language.verbalization.fragments.base import PhraseFragment, VerbFragment, WordFragment
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization.vocabulary.english import Keywords, Logicals

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


class QuantifierRule(VerbalizationRule):
    """
    Abstract base rule: catches
    :class:`~krrood.entity_query_language.operators.logical_quantifiers.ForAll` and
    :class:`~krrood.entity_query_language.operators.logical_quantifiers.Exists`.

    Concrete subclasses (:class:`ForAllRule`, :class:`ExistsRule`) handle each
    quantifier type and take priority due to MRO-depth sorting.
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for any :class:`~krrood.entity_query_language.operators.logical_quantifiers.QuantifiedConditional`."""
        return isinstance(expr, QuantifiedConditional)


class ForAllRule(QuantifierRule):
    """
    Verbalizes universal quantification as *"for all <variables>, <condition>"*.

    Variable names are pluralised (e.g. *"for all Robots, …"*).
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.operators.logical_quantifiers.ForAll`."""
        return isinstance(expr, ForAll)

    @classmethod
    def transform(cls, expr: ForAll, ctx: VerbalizationContext, verbalizer: EQLVerbalizer) -> VerbFragment:
        """
        Build *"for all <plural_var>, <condition>"*.

        :param expr: ForAll quantifier expression.
        :param ctx: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :returns: Universal-quantification phrase.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        var_frag = verbalize_plural(expr.variable, ctx, verbalizer.build)
        cond_frag = verbalizer.build(expr.condition, ctx)
        return phrase(Logicals.FOR_ALL.as_fragment(), var_frag, word(","), cond_frag)


class ExistsRule(QuantifierRule):
    """
    Verbalizes existential quantification as *"there exists <variable> such that <condition>"*.
    """

    @classmethod
    def applies(cls, expr, ctx: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.operators.logical_quantifiers.Exists`."""
        return isinstance(expr, Exists)

    @classmethod
    def transform(cls, expr: Exists, ctx: VerbalizationContext, verbalizer: EQLVerbalizer) -> VerbFragment:
        """
        Build *"there exists <variable> such that <condition>"*.

        :param expr: Exists quantifier expression.
        :param ctx: Shared verbalization state.
        :param verbalizer: Parent verbalizer for recursive calls.
        :returns: Existential-quantification phrase.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        var_frag = verbalizer.build(expr.variable, ctx)
        cond_frag = verbalizer.build(expr.condition, ctx)
        return phrase(
            Logicals.THERE_EXISTS.as_fragment(),
            var_frag,
            Keywords.SUCH_THAT.as_fragment(),
            cond_frag,
        )
