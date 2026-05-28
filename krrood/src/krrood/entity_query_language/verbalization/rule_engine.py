from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from krrood.entity_query_language.verbalization.fragments.base import WordFragment

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.verbalization.context import VerbalizationContext
    from krrood.entity_query_language.verbalization.fragments.base import VerbFragment
    from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer


class VerbalizationRule(ABC):
    """
    Abstract base for a declarative verbalization rule.

    Subclass to declare when a rule applies (:meth:`applies`) and what fragment
    it produces (:meth:`transform`).  The
    :class:`RuleEngine` sorts registered rule classes by MRO depth so that
    more-specific subclasses are always tried before their parents — no priority
    integers are needed.

    All methods are class methods; rules are stateless.

    **Auto-registration:** every concrete (non-abstract) subclass is automatically
    registered via :meth:`__init_subclass__`.  No manual list maintenance is needed —
    just define the class and it will be discovered by :class:`RuleEngine`.
    """

    _registry: list[type[VerbalizationRule]] = []

    def __init_subclass__(cls, **kwargs):
        """Auto-register every concrete (non-abstract) rule subclass."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            VerbalizationRule._registry.append(cls)

    @classmethod
    def registered_rules(cls) -> list[type[VerbalizationRule]]:
        """Return all auto-registered concrete rule classes in definition order."""
        return list(cls._registry)

    @classmethod
    @abstractmethod
    def applies(cls, expr: SymbolicExpression, ctx: VerbalizationContext) -> bool:
        """
        Return ``True`` if this rule can handle *expr*.

        :param expr: EQL expression to test.
        :type expr: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param ctx: Current verbalization state.
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :returns: ``True`` when this rule should be applied to *expr*.
        :rtype: bool
        """

    @classmethod
    @abstractmethod
    def transform(
        cls,
        expr: SymbolicExpression,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Build and return the :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
        for *expr*.

        :param expr: EQL expression to verbalize.
        :type expr: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param ctx: Shared verbalization state (coreference, bindings).
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :param verbalizer: The top-level verbalizer used for recursive sub-expression verbalization.
        :type verbalizer: ~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer
        :returns: Fragment tree representing *expr* in natural language.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """


def _inheritance_depth(cls: type) -> int:
    """MRO depth from :class:`VerbalizationRule` — greater depth = more specific."""
    try:
        return cls.__mro__.index(VerbalizationRule)
    except ValueError:
        return 0


@dataclass
class RuleEngine:
    """
    Applies the first matching :class:`VerbalizationRule` to an expression,
    deepest subclass first.

    On construction the supplied rule classes are sorted by MRO depth
    (``__mro__.index(VerbalizationRule)``, descending) so that subclasses always
    shadow their parents without the caller having to manage ordering.
    """

    _rule_classes: list[type[VerbalizationRule]] = field(repr=False)
    """Unsorted rule classes supplied at construction time (stored for repr)."""

    _rules: list[type[VerbalizationRule]] = field(init=False, repr=False)
    """Rule classes sorted by MRO depth (descending) so subclasses shadow parents."""

    def __post_init__(self) -> None:
        self._rules = sorted(self._rule_classes, key=_inheritance_depth, reverse=True)

    def build(
        self,
        expr: SymbolicExpression,
        ctx: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Dispatch *expr* to the first matching rule and return its fragment.

        Before consulting any rule, checks whether *expr*'s ``_id_`` appears in
        :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.binding_overrides`;
        if so the override fragment is returned immediately.

        Falls back to a plain :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
        bearing ``expr._name_`` when no rule matches.

        :param expr: EQL expression to verbalize.
        :type expr: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param ctx: Shared verbalization state.
        :type ctx: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :param verbalizer: Top-level verbalizer for recursive calls.
        :type verbalizer: ~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer
        :returns: Fragment tree for *expr*.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        var_id = getattr(expr, "_id_", None)
        if var_id is not None and var_id in ctx.binding_overrides:
            return ctx.binding_overrides[var_id]
        for rule_cls in self._rules:
            if rule_cls.applies(expr, ctx):
                return rule_cls.transform(expr, ctx, verbalizer)
        return WordFragment(text=expr._name_)
