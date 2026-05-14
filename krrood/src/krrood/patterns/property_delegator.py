from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import cached_property

from typing_extensions import Type, TypeVar

from krrood.class_diagrams.utils import T
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_param


@dataclass
class PropertyDelegator(SubClassSafeGeneric[T], ABC):
    """Delegates properties and methods of a field of type T to self.

    Subclasses specify which field to delegate from by implementing
    ``delegatee_attribute_name()``.  The transformer then generates a
    ``DelegatorFor<FieldType>`` mixin that forwards every attribute of the
    delegatee field onto the delegating class.

    Example::

        @dataclass
        class Engine:
            horsepower: int

        @dataclass
        class Car(PropertyDelegator[Engine]):
            engine: Engine
            color: str

            @classmethod
            def delegatee_attribute_name(cls) -> str:
                return "engine"

        # After transformation Car also inherits DelegatorForEngine, so:
        car = Car(engine=Engine(200), color="red")
        assert car.horsepower == 200   # delegated to car.engine.horsepower
        assert car.color == "red"      # Car's own field, unaffected
    """

    @classmethod
    @abstractmethod
    def delegatee_attribute_name(cls) -> str:
        """Return the name of the dataclass field that holds the delegatee."""
        ...

    @cached_property
    def delegatee(self) -> T:
        """
        Retrieves the role taker instance.

        Uses object.__getattribute__ to avoid triggering __getattr__ recursion.
        """
        attr_name = self.delegatee_attribute_name()
        try:
            return object.__getattribute__(self, attr_name)
        except AttributeError:
            raise AttributeError(f"Delegatee attribute '{attr_name}' not found.")

    @classmethod
    def get_delegatee_type(cls) -> Type[T]:
        """Return the type of the delegatee field."""
        try:
            type_ = next(
                f.type for f in fields(cls) if f.name == cls.delegatee_attribute_name()
            )
        except StopIteration:
            type_ = get_generic_type_param(cls, PropertyDelegator)[0]
        if isinstance(type_, str):
            try:
                type_ = sys.modules[cls.__module__].__dict__[type_]
            except KeyError:
                type_ = eval(type_, sys.modules[cls.__module__].__dict__)
        if isinstance(type_, TypeVar):
            if type_.__bound__ is not None:
                type_ = type_.__bound__
            else:
                raise ValueError(f"TypeVar {type_} has no bound")
        return type_
