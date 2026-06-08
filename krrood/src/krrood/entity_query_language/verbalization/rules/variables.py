"""
Verbalization rules for EQL variable expressions.

* :class:`VariableRule` — *"a Robot"* / *"the Robot"* with article selection.
* :class:`LiteralRule` — plain literal value rendering.
* :class:`ExternallySetVariableRule` — opaque external variable.
* :class:`InstantiatedVariableRule` — *"a TypeName where … such that …"*.
* :class:`InstantiatedVerbalizableRule` — user-supplied verbalization template.

The module-level helpers at the bottom implement the InstantiatedVariable
natural-language rendering: binding building, copula selection, constraint
deferral, and phrase assembly.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from krrood.entity_query_language.core.variable import (
    ExternallySetVariable,
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.predicate import Verbalizable
from krrood.entity_query_language.verbalization.chain_utils import verbalize_plural
from krrood.entity_query_language.verbalization.context import ArticleSelection
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_and,
    PhraseFragment,
    RoleFragment,
    VerbFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import (
    phrase,
    role,
    word,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    Copulas,
    Keywords,
    Prepositions,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


class VariableRule(VerbalizationRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.core.variable.Variable` expressions
    as *"a/an TypeName"* (first mention) or *"the TypeName"* (subsequent mention).

    Uses :meth:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.noun_for_parts`
    for article selection and coreference tracking.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.core.variable.Variable` expressions."""
        return isinstance(expression, Variable)

    @classmethod
    def transform(
        cls,
        expression: Variable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Build *"a/an TypeName"*, *"the TypeName"*, or just *"TypeName N"* based on context.

        :param expression: Variable expression.
        :param context: Shared verbalization state (article selection + coreference).
        :param verbalizer: Parent verbalizer (unused directly).
        :return: Noun phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        article, label = context.noun_for_parts(expression)
        label_fragment = RoleFragment.for_variable(label, expression)
        if article == ArticleSelection.NONE:
            return label_fragment
        if article == ArticleSelection.DEFINITE:
            return phrase(Articles.THE.as_fragment(), label_fragment)
        return phrase(Articles.indefinite(label), label_fragment)


class LiteralRule(VariableRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.core.variable.Literal` expressions
    as a plain semantic-role fragment using :meth:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.type_name_of_value`.

    Takes priority over :class:`VariableRule` because Literal is a Variable subclass.
    The known limitation is that no source link is generated for literal values.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.core.variable.Literal` expressions."""
        return isinstance(expression, Literal)

    @classmethod
    def transform(
        cls,
        expression: Literal,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Build a LITERAL-role fragment from the Python value.

        :param expression: Literal expression.
        :param context: Shared verbalization state (for value rendering).
        :param verbalizer: Parent verbalizer (unused).
        :return: Literal value fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return role(
            context.type_name_of_value(expression._value_), SemanticRole.LITERAL
        )


class ExternallySetVariableRule(VerbalizationRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.core.variable.ExternallySetVariable`
    as *"a/an TypeName"*.

    :class:`~krrood.entity_query_language.core.variable.ExternallySetVariable` is a sibling
    of :class:`~krrood.entity_query_language.core.variable.Variable` (both inherit
    ``CanHaveDomainSource``) so :class:`VariableRule` does not match it.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.core.variable.ExternallySetVariable`."""
        return isinstance(expression, ExternallySetVariable)

    @classmethod
    def transform(
        cls,
        expression: ExternallySetVariable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Build *"a/an TypeName"* without coreference tracking (external variables are opaque).

        :param expression: ExternallySetVariable expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer (unused).
        :return: Noun phrase fragment.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        type_name = (
            expression._type_.__name__
            if getattr(expression, "_type_", None)
            else "variable"
        )
        return phrase(
            Articles.indefinite(type_name), role(type_name, SemanticRole.VARIABLE)
        )


class InstantiatedVariableRule(VerbalizationRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
    in natural form: *"a TypeName where the field of the TypeName is …"*.

    Delegates to :func:`_verbalize_instantiated_natural` which handles constraint
    frames and binding overrides.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` for :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`."""
        return isinstance(expression, InstantiatedVariable)

    @classmethod
    def transform(
        cls,
        expression: InstantiatedVariable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Delegate to :func:`_verbalize_instantiated_natural`.

        :param expression: InstantiatedVariable expression.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer.
        :return: Full natural-language binding phrase.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        return _verbalize_instantiated_natural(expression, context, verbalizer)


class InstantiatedVerbalizableRule(InstantiatedVariableRule):
    """
    Verbalizes :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`
    when its type implements
    :meth:`~krrood.entity_query_language.predicate.Verbalizable._verbalization_template_`.

    Uses the user-supplied format string directly, substituting verbalized child
    values.  Takes priority over :class:`InstantiatedVariableRule`.

    .. note::
        The format string cannot carry semantic roles, so the result is a plain
        :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`.
    """

    @classmethod
    def applies(cls, expression, context: VerbalizationContext) -> bool:
        """Return ``True`` when the InstantiatedVariable type provides a verbalization template."""
        return isinstance(
            expression, InstantiatedVariable
        ) and _has_verbalization_template(expression)

    @classmethod
    def transform(
        cls,
        expression: InstantiatedVariable,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Apply the verbalization template, substituting verbalized child values.

        :param expression: InstantiatedVariable with a Verbalizable type.
        :param context: Shared verbalization state.
        :param verbalizer: Parent verbalizer for verbalizing child expressions.
        :return: Plain word fragment from the formatted template.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        template = expression._type_._verbalization_template_()
        kwargs = {
            name: verbalizer.verbalize(child, context)
            for name, child in expression._child_vars_.items()
        }
        return word(template.format(**kwargs))


# ── Module-level helpers ───────────────────────────────────────────────────────


def _has_verbalization_template(expression: InstantiatedVariable) -> bool:
    """Return ``True`` when *expression*'s type implements :class:`~krrood.entity_query_language.predicate.Verbalizable` and supplies a template."""
    try:
        if isinstance(expression._type_, type) and issubclass(
            expression._type_, Verbalizable
        ):
            expression._type_._verbalization_template_()
            return True
    except NotImplementedError:
        pass
    return False


def _make_field_ref_frag(field_name: str, type_name: str, type_cls) -> VerbFragment:
    """Build a 'the <field> of the <Type>' fragment with proper semantic roles."""
    return PhraseFragment(
        parts=[
            Articles.THE.as_fragment(),
            RoleFragment(text=field_name, role=SemanticRole.ATTRIBUTE),
            Prepositions.OF.as_fragment(),
            Articles.THE.as_fragment(),
            RoleFragment(
                text=type_name,
                role=SemanticRole.VARIABLE,
                source_ref=(
                    SourceRef.for_type(type_cls) if isinstance(type_cls, type) else None
                ),
            ),
        ]
    )


def _copula_and_value(
    field_name: str,
    child_expression,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> tuple[VerbFragment, VerbFragment]:
    """Return (copula_frag, value_frag) choosing singular/plural based on field name."""
    if morphology.is_plural(field_name):
        return Copulas.ARE.as_fragment(), verbalize_plural(
            child_expression, context, verbalizer.build
        )
    return Copulas.IS.as_fragment(), verbalizer.build(child_expression, context)


def _build_bindings(
    expression: InstantiatedVariable,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> tuple[list[VerbFragment], dict[uuid.UUID, VerbFragment]]:
    """Build all binding fragments and collect overrides without registering them.

    Separating value-building from override-registration ensures no binding's value
    is rendered under a sibling binding's override.
    """
    type_name = getattr(expression._type_, "__name__", str(expression._type_))
    binding_frags: list[VerbFragment] = []
    pending_overrides: dict[uuid.UUID, VerbFragment] = {}
    for field_name, child_expression in expression._child_vars_.items():
        field_ref = _make_field_ref_frag(field_name, type_name, expression._type_)
        copula, value = _copula_and_value(
            field_name, child_expression, context, verbalizer
        )
        binding_frags.append(phrase(field_ref, copula, value))
        pending_overrides[child_expression._id_] = field_ref
    return binding_frags, pending_overrides


def _assemble_instantiated_phrase(
    type_name: str,
    expression: InstantiatedVariable,
    binding_frags: list[VerbFragment],
    constraint_frags: list[VerbFragment],
) -> VerbFragment:
    result_parts: list[VerbFragment] = [
        phrase(
            Articles.indefinite(type_name),
            RoleFragment.for_variable(type_name, expression),
        )
    ]
    if binding_frags:
        joined = oxford_and(binding_frags, Conjunctions.AND.as_fragment())
        result_parts.append(
            PhraseFragment(parts=[word(","), Keywords.WHERE.as_fragment(), joined])
        )
    if constraint_frags:
        joined_c = oxford_and(constraint_frags, Conjunctions.AND.as_fragment())
        result_parts.append(
            PhraseFragment(
                parts=[word(","), Keywords.SUCH_THAT.as_fragment(), joined_c]
            )
        )
    return PhraseFragment(parts=result_parts, separator="")


def _verbalize_instantiated_natural(
    expression: InstantiatedVariable,
    context: VerbalizationContext,
    verbalizer: EQLVerbalizer,
) -> VerbFragment:
    type_name = getattr(expression._type_, "__name__", str(expression._type_))
    seen = context.seen_reference(expression)
    if seen is not None:
        return seen
    context.seen[expression._id_] = type_name

    context.push_constraint_frame()
    binding_frags, overrides = _build_bindings(expression, context, verbalizer)
    context.binding_overrides.update(overrides)
    deferred = context.pop_constraint_frame()
    constraint_frags = [
        verbalizer.build(expression, context) for expression in deferred
    ]

    return _assemble_instantiated_phrase(
        type_name, expression, binding_frags, constraint_frags
    )
