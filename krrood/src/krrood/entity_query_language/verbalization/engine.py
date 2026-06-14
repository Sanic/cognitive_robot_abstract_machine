from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Optional, Sequence

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.fragments.base import Fragment
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
    PhraseRule,
    select,
)
from krrood.entity_query_language.verbalization.grammar.english import RULES
from krrood.entity_query_language.verbalization.exceptions import (
    UnverbalizableExpressionError,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.context import MicroplanningServices


def fold(
    node: SymbolicExpression,
    services: MicroplanningServices,
    rules: Optional[Sequence[PhraseRule]] = None,
    number: Number = Number.SINGULAR,
) -> Fragment:
    """
    Verbalize *node* by dispatching it to its matching grammar rule and recursing — the single
    catamorphism (fold) over the EQL expression tree.

    A node carrying a pre-built binding override is returned directly, before any dispatch.
    When no rule covers the node, an ``UnverbalizableExpressionError`` is raised rather than
    degrading silently to the class name.

    The recursion is an F-algebra fold over the EQL algebra, with the grammar as the algebra
    (Meijer, Fokkinga & Paterson 1991, "Functional Programming with Bananas, Lenses, Envelopes
    and Barbed Wire"; Bird & de Moor 1997, "Algebra of Programming").

    :param node: Any EQL expression.
    :param services: The pass-wide microplanning services (and render configuration).
    :param rules: Grammar to dispatch over; defaults to the standard rule set.
    :param number: Grammatical number to build *node* under.
    :return: The fragment for *node*.
    :raises UnverbalizableExpressionError: when no grammar rule covers *node*.
    """
    rules = RULES if rules is None else rules

    node_id = getattr(node, "_id_", None)
    if node_id is not None:
        override = services.binding.binding_overrides.get(node_id)
        if override is not None:
            return override

    context = RuleContext(
        child=lambda child_node, number=Number.SINGULAR: fold(
            child_node, services, rules, number=number
        ),
        services=services,
        number=number,
    )

    rule = select(node, rules, context)
    if rule is None:
        raise UnverbalizableExpressionError(node=node)
    if rule.enters_query_scope:
        # The rule's construct is a query body: everything built inside sees query_depth >= 1,
        # so a nested Entity renders as a noun phrase. Declared on the rule (not pushed by hand
        # in assemblers) so the policy lives in one place. ``when`` already ran outside.
        with services.configuration.query_depth_scope():
            return rule.build(node, context)
    return rule.build(node, context)
