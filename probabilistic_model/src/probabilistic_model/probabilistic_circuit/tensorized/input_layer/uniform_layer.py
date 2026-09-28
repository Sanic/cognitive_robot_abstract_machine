from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from random_events.interval import SimpleInterval
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import List, Self

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeValues,
    SampleColumn,
    SampleNodeValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.continuous_layer_with_density import (
    ContinuousLayerWithFiniteSupport,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)


@dataclass(eq=False, repr=False)
class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer of uniform distributions over one continuous variable.
    """

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    def log_probability_density_function_value(self) -> NodeValues:
        """
        :return: The log-density of every node.
        """
        with np.errstate(divide="ignore"):
            return -np.log(self.upper - self.lower)

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        return np.where(
            self.included_condition(values),
            self.log_probability_density_function_value(),
            -np.inf,
        )

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        result = (column - self.lower) / (self.upper - self.lower)
        return np.clip(result, 0.0, 1.0)

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        return self.antiderivative_of_moment_at(
            self.upper, order, center
        ) - self.antiderivative_of_moment_at(self.lower, order, center)

    def antiderivative_of_moment_at(
        self, bound: NodeValues, order: int, center: float
    ) -> NodeValues:
        """
        :param bound: One value per node.
        :param order: The order of the moment.
        :param center: The center of the moment.
        :return: The antiderivative of ``density * (value - center) ** order`` of every
            node at its bound.
        """
        density = np.exp(self.log_probability_density_function_value())
        return density * (bound - center) ** (order + 1) / (order + 1)

    def node_distribution(self, index: int, variable: Variable) -> UniformDistribution:
        return UniformDistribution(
            variable=variable, interval=self.simple_interval_of(index)
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[UniformDistribution]
    ) -> Self:
        interval = np.array(
            [
                [distribution.interval.lower, distribution.interval.upper]
                for distribution in distributions
            ],
            dtype=float,
        )
        bounds = np.array(
            [
                [int(distribution.interval.left), int(distribution.interval.right)]
                for distribution in distributions
            ],
            dtype=np.int64,
        )
        return cls(variable_index, interval, bounds)

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> SampleColumn:
        return np.random.uniform(self.lower[node], self.upper[node], amount)

    def log_truncated_of_non_singleton_interval(
        self, interval: SimpleInterval
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node to a simple interval. A uniform truncated to an interval is
        the uniform over the intersection of the two.

        :param interval: The simple interval, which is not a singleton.
        :return: The uniform layer over the intersections and the log-probability of the
            interval under every node.
        """
        lower, upper = float(interval.lower), float(interval.upper)
        left_bound, right_bound = int(interval.left), int(interval.right)

        cumulative = self.cumulative_distribution_of_nodes_from_column(
            np.array([lower, upper])
        )
        probability = cumulative[1] - cumulative[0]
        alive = probability > 0

        # the bounds of the intersection: the tighter side wins, and where the two
        # bounds coincide the interval is open if either of them is open. Bound.OPEN is
        # the larger value, so that is a maximum.
        own_left, own_right = self.bounds[:, 0], self.bounds[:, 1]
        new_left = np.where(
            self.lower > lower,
            own_left,
            np.where(self.lower < lower, left_bound, np.maximum(own_left, left_bound)),
        )
        new_right = np.where(
            self.upper < upper,
            own_right,
            np.where(
                self.upper > upper, right_bound, np.maximum(own_right, right_bound)
            ),
        )

        # impossible nodes keep their parameters and are dropped by the prune pass
        interval_of_nodes = np.where(
            alive[:, None],
            np.stack([np.maximum(self.lower, lower), np.minimum(self.upper, upper)], 1),
            self.interval,
        )
        bounds_of_nodes = np.where(
            alive[:, None], np.stack([new_left, new_right], axis=1), self.bounds
        )
        log_probabilities = np.where(
            alive, np.log(np.where(alive, probability, 1.0)), -np.inf
        )

        return LayerWithLogProbabilities(
            self.__class__(self.variable, interval_of_nodes, bounds_of_nodes),
            log_probabilities,
        )
