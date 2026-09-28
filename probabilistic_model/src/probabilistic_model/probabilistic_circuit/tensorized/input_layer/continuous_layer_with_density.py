from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from random_events.interval import Bound, Interval, SimpleInterval
from typing_extensions import Dict, List, Self, Type

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIntervalBounds,
    NodeIntervals,
    NodeMask,
    NodeValues,
    SampleColumn,
    SampleNodeMask,
    VariableValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    AbstractContinuousLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    RowGroupedSparseArray,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)


@dataclass(eq=False, repr=False)
class ContinuousLayerWithDensity(AbstractContinuousLayer, ABC):
    """
    Abstract base class for the input layers of continuous distributions that have a
    density.

    A node of such a layer truncated to a composite interval becomes a mixture of its
    truncations to the simple intervals, and a node conditioned on a value or truncated
    to a singleton becomes a Dirac delta.
    """

    @abstractmethod
    def log_truncated_of_non_singleton_interval(
        self, interval: SimpleInterval
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node of this layer to a simple interval that is not a singleton.

        :param interval: The simple interval.
        :return: The truncated layer and the log-probabilities of its nodes.
        """
        raise NotImplementedError

    def type_of_layer_truncated_to_interval(
        self, interval: SimpleInterval
    ) -> Type[Layer]:
        """
        :param interval: A simple interval that is not a singleton.
        :return: The type of the layer :meth:`log_truncated_of_non_singleton_interval`
            returns for that interval.
        """
        return self.__class__

    def type_of_truncated_layer(
        self, assignment: Interval, singleton_allowed: bool
    ) -> Type[Layer]:
        if len(assignment.simple_sets) > 1:
            return SumLayer
        if not assignment.simple_sets:
            return self.__class__
        [interval] = assignment.simple_sets
        if singleton_allowed and interval.is_singleton():
            return DiracDeltaLayer
        return self.type_of_layer_truncated_to_interval(interval)

    def log_truncated_of_assignment(
        self, assignment: Interval, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        pieces = [
            self.log_truncated_of_simple_interval(interval, singleton_allowed)
            for interval in assignment.simple_sets
        ]
        if not pieces:
            return LayerWithLogProbabilities(
                self.__deepcopy__(), np.full(self.number_of_nodes, -np.inf)
            )
        if len(pieces) == 1:
            return pieces[0]
        return self.mixture_of_pieces(pieces)

    def log_truncated_of_simple_interval(
        self, interval: SimpleInterval, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node of this layer to a simple interval.

        :param interval: The simple interval.
        :param singleton_allowed: Whether a singleton interval truncates to a Dirac delta
            rather than to an impossible node.
        :return: The truncated layer and the log-probabilities of its nodes.
        """
        if not (singleton_allowed and interval.is_singleton()):
            return self.log_truncated_of_non_singleton_interval(interval)
        log_likelihood = self.log_likelihood_of_nodes_from_column(
            np.array([interval.lower])
        )[0]
        dirac_delta_layer = DiracDeltaLayer(
            self.variable,
            np.full(self.number_of_nodes, float(interval.lower)),
            np.ones(self.number_of_nodes),
        )
        return LayerWithLogProbabilities(dirac_delta_layer, log_likelihood)

    def mixture_of_pieces(
        self, pieces: List[LayerWithLogProbabilities]
    ) -> LayerWithLogProbabilities:
        """
        Mix the truncations of this layer to the simple intervals of a composite
        interval.

        Node ``i`` of the result mixes node ``i`` of every piece, weighted by the
        probability of that piece. Pieces of the same type are joined into one child
        layer, so the number of layers does not grow with the number of simple
        intervals.

        :param pieces: The truncated layer and the log-probabilities of its nodes, per
            simple interval.
        :return: The mixture and the log-probabilities of its nodes.
        """
        number_of_nodes = self.number_of_nodes
        pieces_by_type: Dict[Type[Layer], List[LayerWithLogProbabilities]] = {}
        for piece in pieces:
            pieces_by_type.setdefault(type(piece.layer), []).append(piece)

        child_layers = []
        log_probabilities_per_child_layer = []
        for layer_type, typed_pieces in pieces_by_type.items():
            child_layers.append(
                layer_type.concatenate([piece.layer for piece in typed_pieces])
            )
            log_probabilities_per_child_layer.extend(
                piece.log_probabilities for piece in typed_pieces
            )

        # the pieces are the columns in order, and node i of every piece sits in row i
        number_of_pieces = len(pieces)
        log_weights = RowGroupedSparseArray.from_entries(
            SparseEntries(
                np.concatenate(log_probabilities_per_child_layer),
                np.tile(np.arange(number_of_nodes), number_of_pieces),
                np.arange(number_of_pieces * number_of_nodes),
            ),
            (number_of_nodes, number_of_pieces * number_of_nodes),
        )

        node_log_probabilities = np.logaddexp.reduce(
            [piece.log_probabilities for piece in pieces], axis=0
        )
        return LayerWithLogProbabilities(
            SumLayer(child_layers, log_weights), node_log_probabilities
        )

    def log_conditional_of_value(self, value: float) -> LayerWithLogProbabilities:
        log_likelihood = self.log_likelihood_of_nodes_from_column(np.array([value]))[0]
        dirac_delta_layer = DiracDeltaLayer(
            self.variable,
            np.full(self.number_of_nodes, float(value)),
            np.exp(log_likelihood),
        )
        return LayerWithLogProbabilities(dirac_delta_layer, log_likelihood)


@dataclass(eq=False, repr=False)
class ContinuousLayerWithFiniteSupport(ContinuousLayerWithDensity, ABC):
    """
    Abstract base class for continuous input layers whose nodes have a finite support.
    """

    interval: NodeIntervals
    """
    The lower and upper bound of the support of every node, shape (#nodes, 2).
    """

    bounds: NodeIntervalBounds
    """
    Whether the lower and upper bound of every node are open or closed, as
    :class:`random_events.interval.Bound` values of shape (#nodes, 2).
    """

    @property
    def lower(self) -> NodeValues:
        """
        :return: The lower bounds of the supports of the nodes.
        """
        return self.interval[:, 0]

    @property
    def upper(self) -> NodeValues:
        """
        :return: The upper bounds of the supports of the nodes.
        """
        return self.interval[:, 1]

    @property
    def left_closed(self) -> NodeMask:
        """
        :return: Whether the lower bound of every node is included.
        """
        return self.bounds[:, 0] == int(Bound.CLOSED)

    @property
    def right_closed(self) -> NodeMask:
        """
        :return: Whether the upper bound of every node is included.
        """
        return self.bounds[:, 1] == int(Bound.CLOSED)

    @property
    def number_of_nodes(self) -> int:
        return len(self.interval)

    def simple_interval_of(self, index: int) -> SimpleInterval:
        """
        :param index: The index of a node.
        :return: The support of that node as simple interval.
        """
        return SimpleInterval.from_data(
            float(self.interval[index, 0]),
            float(self.interval[index, 1]),
            Bound(int(self.bounds[index, 0])),
            Bound(int(self.bounds[index, 1])),
        )

    def validate_own(self):
        if self.interval.shape != self.bounds.shape:
            raise ShapeMismatchError(self.interval.shape, self.bounds.shape)

    def included_condition(self, values: SampleColumn) -> SampleNodeMask:
        """
        Check whether values lie inside the support of every node.

        :param values: The values with shape (#events,).
        :return: A boolean array of shape (#events, #nodes).
        """
        column = np.asarray(values, dtype=float).reshape(-1, 1)

        # these arrays hold one entry per event per node, so the homogeneous cases get
        # their own path rather than evaluating both comparisons and selecting between
        # them
        left_closed = self.left_closed
        if left_closed.all():
            left = self.lower <= column
        elif not left_closed.any():
            left = self.lower < column
        else:
            left = np.where(left_closed, self.lower <= column, self.lower < column)

        right_closed = self.right_closed
        if right_closed.all():
            right = column <= self.upper
        elif not right_closed.any():
            right = column < self.upper
        else:
            right = np.where(right_closed, column <= self.upper, column < self.upper)

        return left & right

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(self.variable, self.interval[mask], self.bounds[mask])

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.interval for layer in layers]),
            np.concatenate([layer.bounds for layer in layers]),
        )

    def apply_translation_own(self, translation: VariableValues):
        self.interval = self.interval + translation[self.variable]

    def apply_scaling_own(self, scaling: VariableValues):
        self.interval = self.interval * scaling[self.variable]

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(self.variable, self.interval.copy(), self.bounds.copy())
        memo[id(self)] = result
        return result
