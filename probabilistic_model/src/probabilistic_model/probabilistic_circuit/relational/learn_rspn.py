from __future__ import annotations

from dataclasses import field

import numpy as np
import pandas as pd
from typing import (
    List,
    Optional,
    Iterable,
    Union,
)

from collections import deque
from jpt.learning.impurity import Impurity

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.parametrization.feature_extractor import FeatureExtractor
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)
from random_events.product_algebra import VariableMap
from random_events.variable import Variable


def learn_probabilistic_circuit(
    instances: List[DataAccessObject],
    targets: Optional[Iterable[Variable]] = field(default=None),
    features: Optional[Iterable[Variable]] = field(default=None),
    min_samples_per_leaf: Union[int, float] = field(default=1),
    min_impurity_improvement: float = field(default=0.0),
    max_leaves: Union[int, float] = field(default=float("inf")),
    max_depth: Union[int, float] = field(default=float("inf")),
    dependencies: Optional[VariableMap] = field(default=None),
    total_samples: int = field(default=1),
    indices: Optional[np.ndarray] = field(default=None),
    impurity: Optional[Impurity] = field(default=None),
    c45queue: deque = field(default_factory=deque),
    keep_sample_indices: bool = field(default=False),
    root: Optional[SumUnit] = field(default=None),
) -> ProbabilisticCircuit:
    """
    Learn a ProbabilisticCircuit from a class and a list of instances.
    :param instances: The instances to learn from.
    :param targets: The variables to optimize for.
    :param features: The variables that are used to craft criteria.
    :param min_samples_per_leaf: The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted as fraction w. r. t. the number of samples total.
    :param min_impurity_improvement: The minimum impurity improvement to create another sum node.
    :param max_leaves: The maximum number of leaves in the tree.
    :param max_depth: The maximum depth of the tree.
    :param dependencies: The dependencies between variables.
    :param total_samples: The total number of samples.
    :param indices: The indices of the samples.
    :param impurity: The impurity object to use.
    :param c45queue: The queue to use for C4.5.
    :param keep_sample_indices: Whether to keep the sample indices.
    :param root: The root of the tree.
    :return: The learned ProbabilisticCircuit.
    """

    extractor = FeatureExtractor(instances)

    if not instances:
        raise ValueError("No instances provided")

    df: pd.DataFrame = extractor.create_dataframe()
    df = extractor.preprocess_dataframe(df)
    df = df.sort_index(axis=1)
    variables = infer_variables_from_dataframe(df)

    jpt = JointProbabilityTree(
        annotated_variables=variables,
        total_samples=total_samples,
        indices=indices,
        impurity=impurity,
        c45queue=c45queue,
        keep_sample_indices=keep_sample_indices,
        root=root,
        min_samples_per_leaf=min_samples_per_leaf,
        min_impurity_improvement=min_impurity_improvement,
        max_leaves=max_leaves,
        max_depth=max_depth,
        dependencies=dependencies,
        features=features,
        targets=targets,
    )
    jpt = jpt.fit(df)
    return jpt
