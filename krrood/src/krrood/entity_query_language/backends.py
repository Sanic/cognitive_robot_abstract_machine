from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
)
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Query
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.parametrization.model_registries import (
    ModelRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import (
    UnderspecifiedParameters,
)

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.
    Query backends are objects that answer queries by different means.
    """

    @abstractmethod
    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """


@dataclass
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.
    These can take any query as input.
    """


@dataclass
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.
    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Match[T]) -> Iterable[T]: ...


@dataclass
class SQLAlchemyBackend(SelectiveBackend):
    """
    A backend that selects elements from a database that is available via SQLAlchemy.
    """

    session_maker: sessionmaker
    """
    The session maker used for the database interactions.
    """

    def evaluate(self, expression: Query) -> Iterable:
        session = self.session_maker()
        translator = eql_to_sql(expression, session)
        yield from translator.evaluate()


@dataclass
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A backend that evaluates elements in this python process. This is just ordinary EQL: each
    expression knows how to evaluate itself natively (queries select, matches generate).
    """

    def evaluate(self, expression: Evaluable) -> Iterable:
        yield from expression._evaluate_natively_()


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model registry.
    """

    model_registry: ModelRegistry = field(default_factory=FullyFactorizedRegistry)
    """
    A model registry that can be used to resolve match statements to probabilistic models.
    """

    number_of_samples: int = field(kw_only=True, default=50)
    """
    The number of samples to generate.
    This is only used if the query does not specify a limit.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

        model = self.model_registry.get_model(parameters)

        # apply conditions from literal assignments to underspecified variables
        conditioned, _ = model.conditional(
            parameters.conditioning_assignments_from_literal_values
        )

        if conditioned is None:
            raise NoSolutionFound(expression.expression)

        # apply conditions from the where statements
        if parameters.truncation_assignments_from_where_conditions:
            truncated, _ = conditioned.truncated(
                parameters.truncation_assignments_from_where_conditions
            )
        else:
            truncated = conditioned

        # apply conditions from variable assignments to underspecified variables
        if parameters.truncation_assignments_from_krrood_variables:
            complete_event = parameters.truncation_assignments_from_krrood_variables[0]
            complete_event.fill_missing_variables(parameters.variables.values())
            for event in parameters.truncation_assignments_from_krrood_variables[1:]:
                complete_event = complete_event.intersection_with(event)
            truncated, _ = conditioned.truncated(complete_event, singleton_allowed=True)

            if truncated is None:
                raise NoSolutionFound(expression.expression)

        number_of_samples = expression.expression._limit_ or self.number_of_samples

        # sample and sort by log likelihood
        samples = truncated.sample(number_of_samples)
        log_likelihoods = truncated.log_likelihood(samples)
        samples = samples[log_likelihoods.argsort()[::-1]]

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.construct_instance_from_model_sample(
                truncated.variables, sample
            )
            yield instance
