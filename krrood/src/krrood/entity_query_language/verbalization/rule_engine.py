from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

from typing_extensions import Type

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

    _registry: List[Type[VerbalizationRule]] = []

    def __init_subclass__(cls, **kwargs):
        """Auto-register every concrete (non-abstract) rule subclass."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            VerbalizationRule._registry.append(cls)

    @classmethod
    def registered_rules(cls) -> List[Type[VerbalizationRule]]:
        """Return all auto-registered concrete rule classes in definition order."""
        return list(cls._registry)

    @classmethod
    @abstractmethod
    def applies(cls, expression: SymbolicExpression, context: VerbalizationContext) -> bool:
        """
        Return ``True`` if this rule can handle *expression*.

        :param expression: EQL expression to test.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Current verbalization state.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :returns: ``True`` when this rule should be applied to *expression*.
        :rtype: bool
        """

    @classmethod
    @abstractmethod
    def transform(
        cls,
        expression: SymbolicExpression,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Build and return the :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
        for *expression*.

        :param expression: EQL expression to verbalize.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state (coreference, bindings).
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :param verbalizer: The top-level verbalizer used for recursive sub-expression verbalization.
        :type verbalizer: ~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer
        :returns: Fragment tree representing *expression* in natural language.
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

    _rule_classes: List[Type[VerbalizationRule]] = field(repr=False)
    """Unsorted rule classes supplied at construction time (stored for repr)."""

    _rules: List[Type[VerbalizationRule]] = field(init=False, repr=False)
    """Rule classes sorted by MRO depth (descending) so subclasses shadow parents."""

    def __post_init__(self) -> None:
        self._rules = sorted(self._rule_classes, key=_inheritance_depth, reverse=True)

    def build(
        self,
        expression: SymbolicExpression,
        context: VerbalizationContext,
        verbalizer: EQLVerbalizer,
    ) -> VerbFragment:
        """
        Dispatch *expression* to the first matching rule and return its fragment.

        Before consulting any rule, checks whether *expression*'s ``_id_`` appears in
        :attr:`~krrood.entity_query_language.verbalization.context.VerbalizationContext.binding_overrides`;
        if so the override fragment is returned immediately.

        Falls back to a plain :class:`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
        bearing ``expression._name_`` when no rule matches.

        :param expression: EQL expression to verbalize.
        :type expression: ~krrood.entity_query_language.core.base_expressions.SymbolicExpression
        :param context: Shared verbalization state.
        :type context: ~krrood.entity_query_language.verbalization.context.VerbalizationContext
        :param verbalizer: Top-level verbalizer for recursive calls.
        :type verbalizer: ~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer
        :returns: Fragment tree for *expression*.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        """
        variable_id = getattr(expression, "_id_", None)
        if variable_id is not None and variable_id in context.binding_overrides:
            return context.binding_overrides[variable_id]
        for rule_cls in self._rules:
            if rule_cls.applies(expression, context):
                return rule_cls.transform(expression, context, verbalizer)
        return WordFragment(text=expression._name_)
