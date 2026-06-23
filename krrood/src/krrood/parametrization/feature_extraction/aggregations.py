from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing_extensions import Callable, Optional, Type, Any, ClassVar

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.parametrization.feature_extraction.exceptions import MissingFieldNameError
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from random_events.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.utils import T, recursive_subclasses


def aggregation_statistic(field_name: str) -> Callable[[Callable], Callable]:
    """
    Marks a method as an aggregation statistic for the named exchangeable-part field.

    :param field_name: The name of the exchangeable-part field this statistic aggregates over.
    """

    def decorator(func: Callable) -> Callable:
        AggregationStatistic._aggregation_registry[field_name].append(func)
        return func

    return decorator


@lru_cache(maxsize=None)
def get_aggregation_class(owner: Type) -> Optional[Type[AggregationStatistic]]:
    """
    Returns the most specific :class:`AggregationStatistic` subclass for ``owner``.

    Walks the MRO of ``owner`` from most specific to least specific, returning the
    first subclass of :class:`AggregationStatistic` whose generic ``T`` matches that
    ancestor.  This means that if ``B`` extends ``A`` and only ``AggregationStatistic[A]``
    exists, a lookup for ``B`` will return it.

    :param owner: The domain class to look up.
    :return: The most specific matching subclass, or ``None`` if none has been defined.
    """
    subclasses = list(recursive_subclasses(AggregationStatistic))
    for ancestor in owner.__mro__:
        for subclass in subclasses:
            if subclass.get_generic_type() == ancestor:
                return subclass
    return None


@dataclass
class AggregationStatistic(SubClassSafeGeneric[T]):
    """
    Base class for aggregation statistics over a domain object's exchangeable-part fields.

    Subclasses bind ``T`` to a concrete owner type and declare one or more methods, each
    annotated with :func:`aggregation_statistic`.  Discovery happens automatically via
    :func:`get_aggregation_class` — no explicit registration is required.

    Set :attr:`field_name` to scope :attr:`aggregation_features` and related methods to
    a single exchangeable-part field.

    .. note::
        Each owner class may have at most one ``AggregationStatistic`` subclass, which must
        handle all of its exchangeable-part fields.  Shared logic across owner types should
        be extracted to an intermediate abstract subclass whose concrete children each bind
        their own ``T``.
    """

    instance: T
    """
    The domain object whose statistics are computed.
    """

    field_name: Optional[str] = None
    """
    The exchangeable-part field this instance is scoped to.
    Must be set before accessing :attr:`aggregation_features`.
    """

    _aggregation_registry: ClassVar[dict[str, list[Callable]]] = defaultdict(list)

    @property
    def aggregation_features(self) -> list[Callable]:
        """
        All methods on this class marked with :func:`aggregation_statistic` for :attr:`field_name`.

        :return: The marked callable methods for the scoped field, sorted alphabetically by name.
        :raises MissingFieldNameError: If :attr:`field_name` was not provided.
        """
        if self.field_name is None:
            raise MissingFieldNameError()
        registered = {f.__name__ for f in self._aggregation_registry.get(self.field_name, [])}
        return [
            func
            for _, func in inspect.getmembers(self.__class__, predicate=inspect.isfunction)
            if func.__name__ in registered
        ]

    def symbolic_aggregation_features(self) -> list[MappedVariable]:
        """
        Symbolic variables for statistic methods that aggregate :attr:`field_name`.

        :return: One :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
            per matching statistic method, in alphabetical order.
        """
        aggregation_variable = variable(type(self), [])
        return [getattr(aggregation_variable, func.__name__)() for func in self.aggregation_features]

    def apply_mapping(self) -> list:
        """
        Evaluates every statistic for :attr:`field_name` against this instance.

        :return: One concrete value per matching statistic method, in alphabetical order.
        """
        return [feature.apply_mapping_on_external_root(self) for feature in self.symbolic_aggregation_features()]


def compute_aggregation_statistics(
    domain_object,
    feature_functions: list[MappedVariable],
    latent_variables: list[Variable],
) -> dict[Variable, Any]:
    """
    Evaluate aggregation feature functions against a domain object and map results to latent variables.

    Each feature function is evaluated only if its name matches a latent variable; values outside
    the training domain of their variable are silently skipped to avoid impossible conditioning events.

    :param domain_object: The domain object whose aggregation statistics are computed.
    :param feature_functions: Symbolic feature functions for one exchangeable-part field.
    :param latent_variables: Latent variables that define which statistics are relevant.
    :return: A mapping from matched latent variables to their observed values.
    """
    latent_variable_by_name = {
        latent_variable.name: latent_variable for latent_variable in latent_variables
    }
    aggregation_class = get_aggregation_class(type(domain_object))
    if aggregation_class is None:
        return {}
    aggregation_instance = aggregation_class(instance=domain_object)
    statistics = {}
    for feature_function in feature_functions:
        feature_name = feature_function._name_
        if feature_name not in latent_variable_by_name:
            continue
        value = feature_function.apply_mapping_on_external_root(aggregation_instance)
        latent_variable = latent_variable_by_name[feature_name]
        # ``make_value`` is the random_events API that validates domain membership;
        # it signals an out-of-domain value by raising. There is no non-throwing
        # membership predicate for a raw value against a symbolic domain, so this
        # boundary adapts that exception into a skip.
        try:
            latent_variable.make_value(value)
            statistics[latent_variable] = value
        except (ValueError, TypeError):
            pass
    return statistics
