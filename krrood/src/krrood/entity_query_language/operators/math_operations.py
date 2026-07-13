"""
Arithmetic operators for the Entity Query Language.

An arithmetic node (see :mod:`krrood.entity_query_language.operators.arithmetic`) delegates its
computation to the :class:`MathOperator` it carries: each operator owns both its rendered symbol and the
Python callable that performs it, so the node stays decoupled from the concrete operation.
"""

from __future__ import annotations

import numbers
import operator
from dataclasses import dataclass
from enum import Enum

from typing_extensions import Callable


@dataclass(frozen=True)
class MathOperatorSpecification:
    """
    The symbol and callable that make up one :class:`MathOperator`.
    """

    symbol: str
    """The mathematical symbol used when rendering the operator."""
    function: Callable[..., numbers.Number]
    """The callable that performs the operation over already-resolved operand values."""


class MathOperator(Enum):
    """
    An arithmetic operator usable inside a query. Each member carries the symbol it renders as and the
    callable that computes it over already-resolved operand values.
    """

    ADD = MathOperatorSpecification("+", operator.add)
    SUBTRACT = MathOperatorSpecification("-", operator.sub)
    MULTIPLY = MathOperatorSpecification("*", operator.mul)
    DIVIDE = MathOperatorSpecification("/", operator.truediv)
    FLOOR_DIVIDE = MathOperatorSpecification("//", operator.floordiv)
    MODULO = MathOperatorSpecification("%", operator.mod)
    POWER = MathOperatorSpecification("**", operator.pow)
    NEGATE = MathOperatorSpecification("-", operator.neg)

    @property
    def symbol(self) -> str:
        """
        :return: The mathematical symbol used when rendering this operator.
        """
        return self.value.symbol

    @property
    def function(self) -> Callable[..., numbers.Number]:
        """
        :return: The callable that performs this operation over already-resolved operand values.
        """
        return self.value.function
