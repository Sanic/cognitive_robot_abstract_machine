from __future__ import annotations

import abc
import logging
from abc import ABC, abstractproperty, abstractmethod
from abc import ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from itertools import product
from typing import Tuple, List, Dict, TYPE_CHECKING, Type
from uuid import UUID

import numpy as np
import scipy.sparse as sp
from typing_extensions import Self

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import BaseConstraint
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.exceptions import (
    InfeasibleException,
    VelocityLimitUnreachableException,
)
from giskardpy.qp.pos_in_vel_limits import (
    shifted_velocity_profile,
    compute_slowdown_asap_vel_profile,
)
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.decorators import memoize
from giskardpy.utils.math import mpc
from krrood.symbolic_math.symbolic_math import Vector, Matrix, Scalar, FloatVariable
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
import giskardpy.utils.math as gm
from krrood.utils import memoize

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


def max_velocity_from_horizon_and_jerk_qp(
    prediction_horizon: int,
    vel_limit: float,
    acc_limit: float,
    jerk_limit: float,
    dt: float,
    max_derivative: Derivatives,
    solver_class: Type[QPSolver],
):
    upper_limits = (
        (vel_limit,) * prediction_horizon,
        (acc_limit,) * prediction_horizon,
        (jerk_limit,) * prediction_horizon,
    )
    lower_limits = (
        (-vel_limit,) * prediction_horizon,
        (-acc_limit,) * prediction_horizon,
        (-jerk_limit,) * prediction_horizon,
    )
    return mpc(
        upper_limits=upper_limits,
        lower_limits=lower_limits,
        current_values=(0, 0),
        dt=dt,
        ph=prediction_horizon,
        q_weight=(0, 0, 0),
        lin_weight=(-1, 0, 0),
        solver_class=solver_class,
        link_to_current_vel=False,
    )


def _sorter(*args: dict) -> Tuple[List[sm.SymbolicScalar], np.ndarray]:
    """
    Sorts every arg dict individually and then appends all of them.
    :arg args: a bunch of dicts
    :return: list
    """
    result = []
    result_names = []
    for arg in args:
        result.extend(__helper(arg))
        result_names.extend(__helper_names(arg))
    return result, np.array(result_names)


def __helper(param: dict):
    return [x for _, x in sorted(param.items())]


def __helper_names(param: dict):
    return [x for x, _ in sorted(param.items())]


@dataclass(eq=False)
class ProblemDataPart(ABC):
    """
    min_x 0.5*x^T*diag(w)*x + g^T*x
    s.t.  lb <= x <= ub
               Ex = b
        lbA <= Ax <= ubA
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    def __post_init__(self):
        self.control_horizon = (
            self.config.prediction_horizon - self.config.max_derivative + 1
        )

    def __hash__(self):
        return id(self)

    @property
    def number_of_free_variables(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def number_ineq_slack_variables(self):
        return sum(
            self.control_horizon
            for c in self.constraint_collection.velocity_inequality_constraints
        )

    @abc.abstractmethod
    def construct_expression(
        self,
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        pass

    def _remove_columns_columns_where_variables_are_zero(
        self, free_variable_model: sm.Matrix, max_derivative: Derivatives
    ) -> sm.Matrix:
        if np.prod(free_variable_model.shape) == 0:
            return free_variable_model
        column_ids = []
        end = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            last_non_zero_variable = self.config.prediction_horizon - (
                max_derivative - derivative
            )
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.config.prediction_horizon
            column_ids.extend(range(start, end))
        free_variable_model.remove([], column_ids)
        return free_variable_model


@dataclass
class DirectLimits:
    lower_bounds: sm.Vector = field(init=False)
    upper_bounds: sm.Vector = field(init=False)
    quadratic_weights: sm.Vector = field(init=False)
    linear_weights: sm.Vector = field(init=False)

    @classmethod
    def create(
        cls,
        degrees_of_freedom: List[DegreeOfFreedom],
        config: QPControllerConfig,
    ) -> Self:
        pass


@dataclass
class SlackLimits(DirectLimits):
    lower_slack_limits: sm.Vector = field(init=False)
    upper_slack_limits: sm.Vector = field(init=False)
    names_without_slack: List[str] = field(init=False)
    names_slack: List[str] = field(init=False)

    @classmethod
    def from_constraints(
        cls, constraints: list[BaseConstraint], config: QPControllerConfig
    ):
        self = cls()
        num_of_slack_variables = len(constraints)
        self.quadratic_weights = Vector(
            [
                self.normalized_weight(
                    quadratic_weight=c.quadratic_weight,
                    control_horizon=config.velocity_horizon,
                    normalization_number=config.radian_normalization_number,
                )
                for c in constraints
            ]
        )
        self.linear_weights = Vector(
            [
                self.normalized_weight(
                    quadratic_weight=c.linear_weight,
                    control_horizon=config.velocity_horizon,
                    normalization_number=config.radian_normalization_number,
                )
                for c in constraints
            ]
        )
        self.lower_bounds = Vector([-np.inf] * num_of_slack_variables)
        self.upper_bounds = Vector([np.inf] * num_of_slack_variables)
        return self

    def normalized_weight(
        self,
        quadratic_weight: Scalar,
        normalization_number: float,
        control_horizon: int,
    ) -> Scalar:
        return quadratic_weight * (1 / (normalization_number**2 * control_horizon))


@dataclass
class DofLimits(DirectLimits):
    @classmethod
    def create(
        cls,
        degrees_of_freedom: List[DegreeOfFreedom],
        config: QPControllerConfig,
    ) -> DofLimits:
        self = cls()
        self.free_variable_bounds(degrees_of_freedom, config)
        self.init_weights(degrees_of_freedom, config)
        return self

    def construct_expression(
        self,
    ) -> Tuple[sm.Vector, sm.Vector]:
        # derivative model
        lb_params, ub_params = self.free_variable_bounds()
        num_free_variables = sum(len(x) for x in lb_params)

        lb, self.names = _sorter(*lb_params)
        ub, _ = _sorter(*ub_params)
        self.names_without_slack = self.names[:num_free_variables]
        self.names_slack = self.names[num_free_variables:]

        return sm.Vector(lb), sm.Vector(ub)

    # todo memorize
    def b_profile(
        self,
        dof_symbols: DerivativeMap[FloatVariable],
        lower_limits: DerivativeMap[float],
        upper_limits: DerivativeMap[float],
        solver_class,
        dt: float,
        ph: int,
        eps: float = 0.00001,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        vel_limit = upper_limits.velocity
        acc_limit = upper_limits.acceleration
        jerk_limit = upper_limits.jerk
        if lower_limits.position is not None:
            pos_range = upper_limits.position - lower_limits.position
            # reduce vel limit, if it can surpass position limits in one dt
            vel_limit = min(vel_limit * dt, pos_range / 2) / dt
            # %% compute max possible profile
            profile = gm.simple_mpc(
                vel_limit=vel_limit,
                acc_limit=acc_limit,
                jerk_limit=jerk_limit,
                current_vel=vel_limit,
                current_acc=0,
                dt=dt,
                ph=ph,
                q_weight=(0, 0, 0),
                lin_weight=(-1, 0, 0),
                solver_class=solver_class,
            )
            vel_profile_mpc = profile[:ph]
            acc_profile_mpc = profile[ph : ph * 2]
            pos_error_lb = lower_limits.position - dof_symbols.position
            pos_error_ub = upper_limits.position - dof_symbols.position
            # %% limits to profile, if vel integral bigger than remaining distance to pos limits
            pos_vel_profile_lb, _ = shifted_velocity_profile(
                vel_profile=vel_profile_mpc,
                acc_profile=acc_profile_mpc,
                distance=-pos_error_lb,
                dt=dt,
            )
            pos_vel_profile_lb *= -1
            pos_vel_profile_ub, _ = shifted_velocity_profile(
                vel_profile=vel_profile_mpc,
                acc_profile=acc_profile_mpc,
                distance=pos_error_ub,
                dt=dt,
            )
            # %% when limits are violated, compute the max velocity that can be reached in one step from zero and put it as
            # negative limits
            one_step_change_ = jerk_limit * dt**2
            one_step_change_lb = sm.min(
                sm.max(Scalar(0), pos_error_lb / dt), Scalar(one_step_change_)
            )
            one_step_change_lb = sm.limit(one_step_change_lb, -vel_limit, vel_limit)
            one_step_change_ub = sm.max(
                sm.min(Scalar(0), pos_error_ub / dt), -Scalar(one_step_change_)
            )
            one_step_change_ub = sm.limit(one_step_change_ub, -vel_limit, vel_limit)
            pos_vel_profile_lb[0] = sm.if_greater(
                pos_error_lb, 0, one_step_change_lb, copy(pos_vel_profile_lb[0])
            )
            pos_vel_profile_ub[0] = sm.if_less(
                pos_error_ub, 0, one_step_change_ub, copy(pos_vel_profile_ub[0])
            )

            # all 0, unless lower or upper position limits are violated
            goal_profile = sm.max(pos_vel_profile_lb, 0) + sm.min(pos_vel_profile_ub, 0)
            # skip first when lower or upper position limit are violated
            skip_first = sm.logic_or(
                pos_vel_profile_lb[0] >= 0, pos_vel_profile_ub[0] <= 0
            )
        else:
            goal_profile = sm.Vector.zeros(ph)
            pos_vel_profile_ub = sm.Vector.ones(ph) * vel_limit
            pos_vel_profile_lb = -pos_vel_profile_ub
            skip_first = sm.Scalar.const_false()

        acc_profile = sm.Vector.ones(pos_vel_profile_ub.shape[0]) * acc_limit
        jerk_profile = sm.Vector.ones(pos_vel_profile_ub.shape[0]) * jerk_limit

        # vel and acc profile for slowing down asap
        proj_vel_profile, proj_acc_profile, _ = compute_slowdown_asap_vel_profile(
            dof_symbols.velocity,
            dof_symbols.acceleration,
            goal_profile,
            Scalar(jerk_limit),
            Scalar(dt),
            ph,
            skip_first,
        )
        # jerk profile when slowing down without jerk limits
        _, _, proj_jerk_profile_violated = compute_slowdown_asap_vel_profile(
            dof_symbols.velocity,
            dof_symbols.acceleration,
            goal_profile,
            Scalar(np.inf),
            Scalar(dt),
            ph,
            skip_first,
        )
        # check if my projected vel profile violated position limits
        vel_lb_violated = sm.logic_or(
            sm.logic_any(proj_vel_profile < pos_vel_profile_lb - eps),
            sm.abs(proj_vel_profile[-1]) >= eps,
        )
        vel_ub_violated = sm.logic_or(
            sm.logic_any(proj_vel_profile > pos_vel_profile_ub + eps),
            sm.abs(proj_vel_profile[-1]) >= eps,
        )

        # if either lower or upper position limits would get violated, relax jerk constraints to max slow down.
        special_jerk_limits = sm.logic_or(vel_lb_violated, vel_ub_violated)
        # with 3 derivatives, slow down is possible in 3 steps
        jerk_profile[0] = sm.if_else(
            special_jerk_limits,
            sm.max(Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[0])),
            sm.Scalar(jerk_limit),
        )
        jerk_profile[1] = sm.if_else(
            special_jerk_limits,
            sm.max(Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[1])),
            sm.Scalar(jerk_limit),
        )
        jerk_profile[2] = sm.if_else(
            special_jerk_limits,
            sm.max(Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[2])),
            sm.Scalar(jerk_limit),
        )

        pos_vel_profile_lb = sm.min(pos_vel_profile_lb, pos_vel_profile_ub)
        pos_vel_profile_ub = sm.max(pos_vel_profile_lb, pos_vel_profile_ub)
        acc_profile_lb = -acc_profile
        acc_profile_ub = acc_profile
        jerk_profile_lb = sm.min(jerk_profile, -jerk_profile) * dt**2
        jerk_profile_ub = sm.max(jerk_profile, jerk_profile) * dt**2
        return DegreeOfFreedomLimits[sm.Vector](
            lower=DerivativeMap(
                velocity=pos_vel_profile_lb,
                acceleration=acc_profile_lb,
                jerk=jerk_profile_lb,
            ),
            upper=DerivativeMap(
                velocity=pos_vel_profile_ub,
                acceleration=acc_profile_ub,
                jerk=jerk_profile_ub,
            ),
        )

    def find_best_jerk_limit(
        self,
        prediction_horizon: int,
        dt: float,
        target_vel_limit: float,
        solver_class: Type[QPSolver],
        eps: float = 0.0001,
    ) -> float:
        jerk_limit = (4 * target_vel_limit) / dt**2
        upper_bound = jerk_limit
        lower_bound = 0
        best_vel_limit = 0
        best_jerk_limit = 0
        i = -1
        for i in range(100):
            vel_limit = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=prediction_horizon,
                vel_limit=1000,
                acc_limit=np.inf,
                jerk_limit=jerk_limit,
                dt=dt,
                max_derivative=Derivatives.jerk,
                solver_class=solver_class,
            )[0]
            if abs(vel_limit - target_vel_limit) < abs(
                best_vel_limit - target_vel_limit
            ):
                best_vel_limit = vel_limit
                best_jerk_limit = jerk_limit
            if abs(vel_limit - target_vel_limit) < eps:
                break
            if vel_limit > target_vel_limit:
                upper_bound = jerk_limit
                jerk_limit = round((jerk_limit + lower_bound) / 2, 4)
            else:
                lower_bound = jerk_limit
                jerk_limit = round((jerk_limit + upper_bound) / 2, 4)
        logger.debug(
            f"best velocity limit: {best_vel_limit} "
            f"(target = {target_vel_limit}) with jerk limit: {best_jerk_limit} after {i + 1} iterations"
        )
        return best_jerk_limit

    def all_limits(
        self,
        degree_of_freedom: DegreeOfFreedom,
        max_derivative: Derivatives,
        config: QPControllerConfig,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        lower_limits = DerivativeMap()
        upper_limits = DerivativeMap()

        # %% pos limits
        if not degree_of_freedom.has_position_limits():
            lower_limits.position = upper_limits.position = None
        else:
            lower_limits.position = degree_of_freedom.limits.lower.position
            upper_limits.position = degree_of_freedom.limits.upper.position

        # %% vel limits
        lower_limits.velocity = degree_of_freedom.limits.lower.velocity
        upper_limits.velocity = degree_of_freedom.limits.upper.velocity
        if config.prediction_horizon == 1:
            raise NotImplementedError("tell ichumuh you actually need this")

        # %% acc limits
        if degree_of_freedom.limits.lower.acceleration is None:
            lower_limits.acceleration = -np.inf
        else:
            lower_limits.acceleration = degree_of_freedom.limits.lower.acceleration
        if degree_of_freedom.limits.upper.acceleration is None:
            upper_limits.acceleration = np.inf
        else:
            upper_limits.acceleration = degree_of_freedom.limits.upper.acceleration

        # %% jerk limits
        if upper_limits.jerk is None:
            upper_limits.jerk = self.find_best_jerk_limit(
                config.prediction_horizon,
                config.mpc_dt,
                upper_limits.velocity,
                solver_class=config.qp_solver_class,
            )
            lower_limits.jerk = -upper_limits.jerk
        else:
            upper_limits.jerk = degree_of_freedom.limits.upper.jerk
            lower_limits.jerk = degree_of_freedom.limits.lower.jerk

        try:
            return self.b_profile(
                dof_symbols=degree_of_freedom.variables,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                solver_class=config.qp_solver_class,
                dt=config.mpc_dt,
                ph=config.prediction_horizon,
            )
        except InfeasibleException as e:
            max_reachable_vel = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=config.prediction_horizon,
                vel_limit=100,
                acc_limit=upper_limits.acceleration,
                jerk_limit=upper_limits.jerk,
                dt=config.mpc_dt,
                max_derivative=max_derivative,
                solver_class=config.qp_solver_class,
            )[0]
            if max_reachable_vel < upper_limits.velocity:
                error_msg = (
                    f'Free variable "{degree_of_freedom.name}" can\'t reach velocity limit of "{upper_limits.velocity}". '
                    f'Maximum reachable with prediction horizon = "{config.prediction_horizon}", '
                    f'jerk limit = "{upper_limits.jerk}" and dt = "{config.mpc_dt}" is "{max_reachable_vel}".'
                )
                logger.error(error_msg)
                raise VelocityLimitUnreachableException(error_msg)
            else:
                raise

    def free_variable_bounds(
        self,
        degrees_of_freedom: List[DegreeOfFreedom],
        config: QPControllerConfig,
    ):
        max_derivative = config.max_derivative
        lower_bounds = []
        upper_bounds = []
        cache: dict[UUID, DegreeOfFreedomLimits[sm.Vector]] = {}
        for degree_of_freedom in degrees_of_freedom:
            all_limits = self.all_limits(
                degree_of_freedom=degree_of_freedom,
                max_derivative=max_derivative,
                config=config,
            )
            cache[degree_of_freedom.id] = all_limits
        for derivative, t in product(
            [Derivatives.velocity, Derivatives.jerk], range(config.prediction_horizon)
        ):
            if t >= config.prediction_horizon - (max_derivative - derivative):
                continue
            for degree_of_freedom in degrees_of_freedom:
                lower_bound = cache[degree_of_freedom.id].lower[derivative][t]
                upper_bound = cache[degree_of_freedom.id].upper[derivative][t]
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

        self.lower_bounds = sm.Vector(lower_bounds)
        self.upper_bounds = sm.Vector(upper_bounds)

    def init_weights(
        self,
        degrees_of_freedom: List[DegreeOfFreedom],
        config: QPControllerConfig,
    ):
        max_derivative = config.max_derivative
        quadratic_weights = []
        for derivative, t in product(
            [Derivatives.velocity, Derivatives.jerk], range(config.prediction_horizon)
        ):
            if t >= config.prediction_horizon - (max_derivative - derivative):
                continue
            for degree_of_freedom in degrees_of_freedom:
                normalized_weight = self.normalize_dof_weight(
                    limit=degree_of_freedom.limits.upper[derivative],
                    base_weight=config.get_dof_weight(
                        degree_of_freedom.name, derivative
                    ),
                    t=t,
                    derivative=derivative,
                    horizon=config.prediction_horizon - 3,
                    alpha=config.horizon_weight_gain_scalar,
                )
                quadratic_weights.append(normalized_weight)
        self.quadratic_weights = sm.Vector(quadratic_weights)
        self.linear_weights = sm.Vector.zeros(len(quadratic_weights))

    def normalize_dof_weight(
        self, limit, base_weight, t, derivative, horizon, alpha
    ) -> sm.Scalar:
        def linear(x_in: float, weight: float, h: int, alpha: float) -> float:
            start = weight * alpha
            a = (weight - start) / h
            return a * x_in + start

        if limit is None:
            return 0.0
        weight = linear(t, base_weight, horizon, alpha)

        return weight * (1 / limit) ** 2


@dataclass
class QPConstraintComponent(ABC):
    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    strategy: type[ConstraintStrategy]

    matrix: sm.Matrix = field(init=False)
    slack_matrix: sm.Matrix = field(init=False)
    slack_variables: DirectLimits = field(init=False)

    @property
    def number_of_free_variables(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def number_of_velocity_columns(self) -> int:
        return self.number_of_free_variables * (self.config.prediction_horizon - 2)

    @property
    def number_of_jerk_columns(self) -> int:
        return self.number_of_free_variables * self.config.prediction_horizon

    @property
    def position_variables(self) -> Vector:
        return Vector([dof.variables.position for dof in self.degrees_of_freedom])

    @property
    def velocity_variables(self) -> Vector:
        return Vector([dof.variables.velocity for dof in self.degrees_of_freedom])

    @property
    def acceleration_variables(self) -> Vector:
        return Vector([dof.variables.acceleration for dof in self.degrees_of_freedom])

    @property
    @abstractmethod
    def constraint_names(self) -> list[str]: ...


@dataclass
class EqualityConstraint(QPConstraintComponent):
    bounds: Vector = field(init=False)


@dataclass
class InequalityConstraint(QPConstraintComponent):
    lower_bounds: Vector = field(init=False)
    upper_bounds: Vector = field(init=False)


@dataclass
class ConstraintStrategy:
    pass


@dataclass
class IntegralConstraintStrategy(ConstraintStrategy):
    """
    Equality constraints have the form:
    .. math::
        f(q) = b

    where

    .. math::

        target - f = \Delta t \sum_{k=0}^{N-1} J_{f} * sp_k

    ::

        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------|
        |  J1*sp |  J1*sp |  J3*sp | J3*sp  | sp     | sp     |
        |-----------------------------------------------------|
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    config: QPControllerConfig

    @property
    def number_of_free_variables(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def number_of_velocity_columns(self) -> int:
        return self.number_of_free_variables * (self.config.prediction_horizon - 2)

    @property
    def number_of_jerk_columns(self) -> int:
        return self.number_of_free_variables * self.config.prediction_horizon

    @property
    def position_variables(self) -> Vector:
        return Vector([dof.variables.position for dof in self.degrees_of_freedom])

    @property
    def velocity_variables(self) -> Vector:
        return Vector([dof.variables.velocity for dof in self.degrees_of_freedom])

    @property
    def acceleration_variables(self) -> Vector:
        return Vector([dof.variables.acceleration for dof in self.degrees_of_freedom])

    def create_matrix(self, constraints: list[BaseConstraint]) -> Matrix:
        if len(constraints) == 0:
            return sm.Matrix()
        J_eq = (
            sm.Vector([c.expression for c in constraints]).jacobian(
                variables=self.position_variables
            )
            * self.config.mpc_dt
        )
        return sm.hstack(
            [J_eq for _ in range(self.config.velocity_horizon)]
            + [sm.Matrix.zeros(J_eq.shape[0], self.number_of_jerk_columns)]
        )

    def create_slack_matrix(self, constraints: list[BaseConstraint]) -> Matrix:
        if len(constraints) == 0:
            return sm.Matrix()
        return sm.Matrix.diag([self.config.mpc_dt for _ in constraints])

    def create_slack_variables(self, constraints: list[BaseConstraint]) -> DirectLimits:
        return SlackLimits.from_constraints(
            constraints=constraints,
            config=self.config,
        )

    def _apply_cap(self, value: Scalar, dt: float, control_horizon: int) -> Scalar:
        # todo normalization with jacobian???
        return sm.limit(
            value,
            -self.config.radian_normalization_number * dt * control_horizon,
            self.config.radian_normalization_number * dt * control_horizon,
        )

    def capped_bound(
        self, equality_bound: Scalar, dt: float, control_horizon: int
    ) -> Scalar:
        return self._apply_cap(equality_bound, dt, control_horizon)

    def create_bounds(self, bounds: list[Scalar]) -> Vector:
        return Vector(
            [
                self.capped_bound(
                    bound, self.config.mpc_dt, self.config.velocity_horizon
                )
                for bound in bounds
            ]
        )


@dataclass
class InequalityConstraintModel(InequalityConstraint):
    strategy: type[ConstraintStrategy] = IntegralConstraintStrategy

    def __post_init__(self):
        constraints = self.constraint_collection.inequality_constraints
        strategy = self.strategy(self.degrees_of_freedom, self.config)
        self.matrix = strategy.create_matrix(constraints)
        self.slack_matrix = strategy.create_slack_matrix(constraints)
        self.slack_variables = strategy.create_slack_variables(constraints)
        self.lower_bounds = strategy.create_bounds([c.lower_error for c in constraints])
        self.upper_bounds = strategy.create_bounds([c.upper_error for c in constraints])

    @property
    def constraint_names(self) -> list[str]:
        return [c.name for c in self.constraint_collection.inequality_constraints]


@dataclass
class InequalityVelocityQPComponent(ABC):
    def derivative_slack_limits(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, sm.Scalar], Dict[str, sm.Scalar]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.config.prediction_horizon):
            for (
                c
            ) in self.constraint_collection.get_inequality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    lower_slack[f"t{t:03}/{c.name}"] = c.lower_slack_limit
                    upper_slack[f"t{t:03}/{c.name}"] = c.upper_slack_limit
        return lower_slack, upper_slack


@dataclass
class EqualityDerivativeLinkModel(EqualityConstraint):
    r"""
    The constraints produced by this class describe the discrete-time relationships between variables
    in the prediction horizon :math:`N` using a semi-implicit euler integration method:

    .. math::

        v_k = v_{k-1} + a_{k} \, \Delta t

        a_k = a_{k-1} + j_{k} \, \Delta t

    Where v, a and j are velocity, acceleration and jerk, respectively, and k is the time step.
    Acceleration variables are removed using substitution.
    The first two row links the MPC to the current state:

    .. math::

        -v_{current} - a_{current} \, \Delta t = -v_0 + j_0 \, \Delta t^2

        v_{current} = - v_1 + 2 v_0 + j_1 \, \Delta t^2

    Row from 2 until k-2 have this form:

    .. math::

        0 = - v_k + 2 v_{k-1} - v_{k-2} + j_k \, \Delta t^2

    The final two rows have this form:

    .. math::

        0 = 2 v_{k-1} - v_{k-2} + j_k \, \Delta t^2

        0 = - v_{k-2} + j_k \, \Delta t^2

    For a prediciton horizon of 5 with 1 degree of freedom, the matrix looks like this:

    ::

        |  equality_bounds |   |           equality constraint matrix          |   |    v_0    |
        |------------------|   |-----------------------------------------------|   |    v_1    |
        | - v_c - a_c * dt |   | -1  |     |     |  1  |     |     |     |     |   |    v_2    |
        |       v_c        |   |  2  | -1  |     |     |  1  |     |     |     |   | j_0*dt**2 |
        |        0         | = | -1  |  2  | -1  |     |     |  1  |     |     | @ | j_1*dt**2 |
        |        0         |   |     | -1  |  2  |     |     |     |  1  |     |   | j_2*dt**2 |
        |        0         |   |     |     | -1  |     |     |     |     |  1  |   | j_3*dt**2 |
        |------------------|   |-----------------------------------------------|   | j_4*dt**2 |
    """

    strategy: type[ConstraintStrategy] = None

    def __post_init__(self):
        self.create_matrix()
        self.compute_bounds()
        self.slack_matrix = Matrix.zeros(self.matrix.shape[0], 0)

    @property
    def constraint_names(self) -> list[str]:
        names = []
        for k in range(self.config.prediction_horizon):
            for dof in self.degrees_of_freedom:
                names.append(f"{dof.name} k_{k} vel/jerk link")
        return names

    def create_matrix(self):
        matrix = np.zeros(
            (
                self.number_of_jerk_columns,
                self.number_of_velocity_columns + self.number_of_jerk_columns,
            )
        )
        identity = np.eye(self.number_of_velocity_columns)
        velocity_at_k = -identity
        velocity_at_k_minus1 = -identity
        velocity_at_k_minus2 = 2 * identity
        matrix[
            : -self.number_of_free_variables * 2, : self.number_of_velocity_columns
        ] += velocity_at_k
        matrix[
            self.number_of_free_variables : -self.number_of_free_variables,
            : self.number_of_velocity_columns,
        ] += velocity_at_k_minus2
        matrix[
            self.number_of_free_variables * 2 :, : self.number_of_velocity_columns
        ] += velocity_at_k_minus1

        matrix[:, self.number_of_velocity_columns :] = np.eye(
            self.number_of_jerk_columns
        )

        self.matrix = sm.Matrix(matrix)

    def compute_bounds(self):
        self.bounds = sm.Vector.zeros(self.number_of_jerk_columns)
        self.bounds[: self.number_of_free_variables] = (
            -self.velocity_variables - self.acceleration_variables * self.config.mpc_dt
        )
        self.bounds[
            self.number_of_free_variables : self.number_of_free_variables * 2
        ] = self.velocity_variables


@dataclass
class EqualityConstraintModel(EqualityConstraint):
    strategy: type[ConstraintStrategy] = IntegralConstraintStrategy

    def __post_init__(self):
        constraints = self.constraint_collection.equality_constraints
        strategy = self.strategy(self.degrees_of_freedom, self.config)
        self.matrix = strategy.create_matrix(constraints)
        self.slack_matrix = strategy.create_slack_matrix(constraints)
        self.slack_variables = strategy.create_slack_variables(constraints)
        self.bounds = strategy.create_bounds([c.bound for c in constraints])

    @property
    def constraint_names(self) -> list[str]:
        return [c.name for c in self.constraint_collection.equality_constraints]


@dataclass
class EqualityVelocityConstraintModel:
    def eq_derivative_slack_limits(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, sm.Scalar], Dict[str, sm.Scalar]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.config.prediction_horizon):
            for c in self.constraint_collection.get_equality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    lower_slack[f"t{t:03}/{c.name}"] = c.lower_slack_limit[t]
                    upper_slack[f"t{t:03}/{c.name}"] = c.upper_slack_limit[t]
        return lower_slack, upper_slack

    @memoize
    def find_best_jerk_limit(
        self,
        prediction_horizon: int,
        dt: float,
        target_vel_limit: float,
        solver_class: Type[QPSolver],
        eps: float = 0.0001,
    ) -> float:
        jerk_limit = (4 * target_vel_limit) / dt**2
        upper_bound = jerk_limit
        lower_bound = 0
        best_vel_limit = 0
        best_jerk_limit = 0
        i = -1
        for i in range(100):
            vel_limit = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=prediction_horizon,
                vel_limit=1000,
                acc_limit=np.inf,
                jerk_limit=jerk_limit,
                dt=dt,
                max_derivative=Derivatives.jerk,
                solver_class=solver_class,
            )[0]
            if abs(vel_limit - target_vel_limit) < abs(
                best_vel_limit - target_vel_limit
            ):
                best_vel_limit = vel_limit
                best_jerk_limit = jerk_limit
            if abs(vel_limit - target_vel_limit) < eps:
                break
            if vel_limit > target_vel_limit:
                upper_bound = jerk_limit
                jerk_limit = round((jerk_limit + lower_bound) / 2, 4)
            else:
                lower_bound = jerk_limit
                jerk_limit = round((jerk_limit + upper_bound) / 2, 4)
        logger.debug(
            f"best velocity limit: {best_vel_limit} "
            f"(target = {target_vel_limit}) with jerk limit: {best_jerk_limit} after {i + 1} iterations"
        )
        return best_jerk_limit


@dataclass
class Weights(ProblemDataPart):
    """
    order:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        eq integral constraints
        eq vel constraints
        neq integral constraints
        neq vel constraints
    """

    evaluated: bool = field(default=True)

    def construct_expression(self) -> Tuple[sm.Vector, sm.Vector]:
        components = []
        components.extend(self.free_variable_weights_expression())
        components.append(self.equality_weight_expressions())
        components.extend(self.eq_derivative_weight_expressions())
        components.append(self.inequality_weight_expressions())
        components.extend(self.derivative_weight_expressions())
        weights, _ = _sorter(*components)
        quadratic_weights = []
        linear_weights = []
        for quadratic_weight, linear_weight in weights:
            quadratic_weights.append(quadratic_weight)
            linear_weights.append(linear_weight)
        return sm.Vector(quadratic_weights), sm.Vector(linear_weights)

    def derivative_weight_expressions(self) -> List[Dict[str, sm.Scalar]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.config.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.config.prediction_horizon):
                d = Derivatives(d)
                for (
                    c
                ) in self.constraint_collection.get_inequality_constraints_by_derivative(
                    d
                ):
                    if t < self.control_horizon:
                        derivative_constr_weights[f"t{t:03}/{c.name}"] = (
                            c.normalized_weight(),
                            c.linear_weight,
                        )
            params.append(derivative_constr_weights)
        return params

    def eq_derivative_weight_expressions(self) -> List[Dict[str, sm.Scalar]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.config.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.config.prediction_horizon):
                d = Derivatives(d)
                for (
                    c
                ) in self.constraint_collection.get_equality_constraints_by_derivative(
                    d
                ):
                    if t < self.control_horizon:
                        derivative_constr_weights[f"t{t:03}/{c.name}"] = (
                            c.normalized_weight(),
                            c.linear_weight,
                        )
            params.append(derivative_constr_weights)
        return params

    def equality_weight_expressions(self) -> dict:
        error_slack_weights = {
            f"{c.name}/error": (
                c.normalized_weight(self.control_horizon),
                c.linear_weight,
            )
            for c in self.constraint_collection.equality_constraints
        }
        return error_slack_weights

    def inequality_weight_expressions(self) -> dict:
        error_slack_weights = {
            f"{c.name}/error": (
                c.normalized_weight(self.control_horizon),
                c.linear_weight,
            )
            for c in self.constraint_collection.inequality_constraints
        }
        return error_slack_weights

    def get_free_variable_symbols(self, order: Derivatives) -> List[sm.FloatVariable]:
        return _sorter(
            {
                v.variables.position: v.variables.data[order]
                for v in self.degrees_of_freedom
            }
        )[0]


# @dataclass(eq=False)
# class FreeVariableBounds(ProblemDataPart):
#     """
#     order:
#         free_variable_velocity
#         free_variable_acceleration
#         free_variable_jerk
#         eq integral constraints
#         eq vel constraints
#         neq integral constraints
#         neq vel constraints
#     """
#
#     names: np.ndarray = field(default=None)
#     names_without_slack: np.ndarray = field(default=None)
#     names_slack: np.ndarray = field(default=None)
#     names_neq_slack: np.ndarray = field(default=None)
#     names_derivative_slack: np.ndarray = field(default=None)
#     names_eq_slack: np.ndarray = field(default=None)
#     evaluated: bool = field(default=True)


@dataclass
class EqualityBounds(ProblemDataPart):
    """

    order:
        derivative model (optional)
        eq integral constraints
        eq vel constraints
    """

    names: np.ndarray = field(default=None)
    names_equality_constraints: np.ndarray = field(default=None)
    names_derivative_links: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def eq_derivative_constraint_bounds(
        self, derivative: Derivatives
    ) -> Dict[str, sm.Vector]:
        bound = {}
        for t in range(self.config.prediction_horizon):
            for c in self.constraint_collection.get_equality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    bound[f"t{t:03}/{c.name}"] = c.bound[t] * self.config.mpc_dt
        return bound

    def equality_constraint_bounds(self) -> Dict[str, sm.Scalar]:
        return {
            f"{c.name}": c.capped_bound(self.config.mpc_dt, self.control_horizon)
            for c in self.constraint_collection.equality_constraints
        }

    def construct_expression(
        self,
    ) -> sm.Vector:
        """
        explicit no acc
        -vc - ac*dt = -v0 + j0*dt**2
                 vc = -v1 + 2*v0 + j1*dt**2
                  0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 +    0   - vt-2 + jt*dt**2
        :return:
        """
        bounds = []
        # derivative model
        derivative_link = {}
        for t in range(self.config.prediction_horizon):
            for v in self.degrees_of_freedom:
                name = f"t{t:03}/{Derivatives.jerk}/{v.name}/link"
                if t == 0:
                    derivative_link[name] = (
                        -v.variables.velocity
                        - v.variables.acceleration * self.config.mpc_dt
                    )
                elif t == 1:
                    derivative_link[name] = v.variables.velocity
                else:
                    derivative_link[name] = 0
        bounds.append(derivative_link)

        num_derivative_links = sum(len(x) for x in bounds)
        num_derivative_constraints = 0

        # eq integral constraints
        bounds.append(self.equality_constraint_bounds())

        # eq vel constraints
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            bound = self.eq_derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(bound)
            bounds.append(bound)

        bounds, self.names = _sorter(*bounds)
        self.names_derivative_links = self.names[:num_derivative_links]
        # self.names_equality_constraints = self.names[num_derivative_links:]
        return sm.Vector(bounds)


@dataclass
class InequalityBounds(ProblemDataPart):
    """
    order:
        derivative model (optional)
        neq integral constraints
        neq vel constraints
    """

    default_limits: bool = field(default=False)
    names: np.ndarray = field(default=None)
    names_position_limits: np.ndarray = field(default=None)
    names_derivative_links: np.ndarray = field(default=None)
    names_neq_constraints: np.ndarray = field(default=None)
    names_non_position_limits: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def derivative_constraint_bounds(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, sm.Vector], Dict[str, sm.Vector]]:
        lower = {}
        upper = {}
        for t in range(self.config.prediction_horizon):
            for (
                c
            ) in self.constraint_collection.get_inequality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    lower[f"t{t:03}/{c.name}"] = c.lower_limit * self.config.mpc_dt
                    upper[f"t{t:03}/{c.name}"] = c.upper_limit * self.config.mpc_dt
        return lower, upper

    def lower_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.constraint_collection.inequality_constraints:
            if isinstance(constraint.lower_error, float) and np.isinf(
                constraint.lower_error
            ):
                bounds[f"{constraint.name}"] = constraint.lower_error
            else:
                bounds[f"{constraint.name}"] = constraint.capped_lower_error(
                    self.config.mpc_dt, self.control_horizon
                )
        return bounds

    def upper_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.constraint_collection.inequality_constraints:
            if isinstance(constraint.upper_error, float) and np.isinf(
                constraint.upper_error
            ):
                bounds[f"{constraint.name}"] = constraint.upper_error
            else:
                bounds[f"{constraint.name}"] = constraint.capped_upper_error(
                    self.config.mpc_dt, self.control_horizon
                )
        return bounds

    def implicit_pos_model_limits(
        self,
    ) -> Tuple[List[Dict[str, sm.Vector]], List[Dict[str, sm.Vector]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.degrees_of_freedom:
            lb_, ub_ = v.lower_limits.position, v.upper_limits.position
            for t in range(self.config.prediction_horizon - 2):
                ptc = v.variables.position
                lb_jerk[f"t{t:03}/{v.name}/{Derivatives.position}"] = lb_ - ptc
                ub_jerk[f"t{t:03}/{v.name}/{Derivatives.position}"] = ub_ - ptc
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    def implicit_model_limits(
        self,
    ) -> Tuple[List[Dict[str, sm.Vector]], List[Dict[str, sm.Vector]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.degrees_of_freedom:
            lb_, ub_ = self.velocity_limit(v=v, max_derivative=Derivatives.jerk)
            for t in range(self.config.prediction_horizon):
                # if self.config.max_derivative >= Derivatives.acceleration:
                #     a_min = v.get_lower_limit(Derivatives.acceleration)
                #     a_max = v.get_upper_limit(Derivatives.acceleration)
                #     if not ((np.isinf(a_min) or cas.is_inf(a_min)) and (np.isinf(a_max) or cas.is_inf(a_max))):
                #         vtc = v.symbols.velocity
                #         if t == 0:
                #             # vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.config.mpc_dt + a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.config.mpc_dt + a_max
                #         else:
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_max
                if self.config.max_derivative >= Derivatives.jerk:
                    j_min = lb_[self.config.prediction_horizon * 2 + t]
                    j_max = ub_[self.config.prediction_horizon * 2 + t]
                    vtc = v.variables.velocity
                    atc = v.variables.acceleration
                    if t == 0:
                        # vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            vtc / self.config.mpc_dt**2
                            + atc / self.config.mpc_dt
                            + j_min
                        )
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            vtc / self.config.mpc_dt**2
                            + atc / self.config.mpc_dt
                            + j_max
                        )
                    elif t == 1:
                        # (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            -vtc / self.config.mpc_dt**2 + j_min
                        )
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            -vtc / self.config.mpc_dt**2 + j_max
                        )
                    else:
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = j_min
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = j_max
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    def construct_expression(
        self,
    ) -> Tuple[sm.Vector, sm.Vector]:
        lb_params: List[Dict[str, sm.Vector]] = []
        ub_params: List[Dict[str, sm.Vector]] = []

        # neq integral constraints
        lower_inequality_constraint_bounds = self.lower_inequality_constraint_bound()
        lb_params.append(lower_inequality_constraint_bounds)
        ub_params.append(self.upper_inequality_constraint_bound())
        num_neq_constraints = len(lower_inequality_constraint_bounds)

        # neq vel constraints
        num_derivative_constraints = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower, upper = self.derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(lower)
            lb_params.append(lower)
            ub_params.append(upper)

        lbA, self.names = _sorter(*lb_params)
        ubA, _ = _sorter(*ub_params)

        self.names_derivative_links = self.names[:num_derivative_constraints]
        self.names_neq_constraints = self.names[
            num_derivative_constraints + num_neq_constraints :
        ]

        return sm.Vector(lbA), sm.Vector(ubA)


@dataclass
class EqualityModel(ProblemDataPart):
    """
    Format:
        last free variable velocity
        0
        last free variable acceleration
        0
        equality_constraint_bounds
    """

    def equality_constraint_expressions(self) -> List[sm.Matrix]:
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.equality_constraints
            }
        )[0]

    def get_free_variable_symbols(
        self, derivative: Derivatives
    ) -> List[sm.FloatVariable]:
        return _sorter(
            {
                v.variables.position.name: v.variables.data[derivative]
                for v in self.degrees_of_freedom
            }
        )[0]

    def get_eq_derivative_constraint_expressions(self, derivative: Derivatives):
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.derivative_equality_constraints
                if c.derivative == derivative
            }
        )[0]

    def velocity_constraint_model(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(
            self.constraint_collection.velocity_equality_constraints
        ) * (self.config.prediction_horizon - 2)
        if number_of_vel_rows > 0:
            expressions = sm.Matrix(
                self.get_eq_derivative_constraint_expressions(Derivatives.velocity)
            )
            parts = []
            for derivative in Derivatives.range(
                Derivatives.position, self.config.max_derivative - 1
            ):
                if derivative == Derivatives.velocity:
                    continue
                if derivative == Derivatives.acceleration:
                    continue
                J_vel = (
                    expressions.jacobian(
                        variables=self.get_free_variable_symbols(derivative)
                    )
                    * self.config.mpc_dt
                )
                missing_variables = self.config.max_derivative - derivative - 1
                eye = sm.Matrix.eye(self.config.prediction_horizon)[
                    :-2, : self.config.prediction_horizon - missing_variables
                ]
                J_vel_limit_block = eye.kron(J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = sm.hstack(parts)
            num_slack_variables = sum(
                self.control_horizon
                for c in self.constraint_collection.velocity_equality_constraints
            )
            slack_model = sm.Matrix.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    @property
    def number_of_non_slack_columns(self) -> int:
        vel_columns = self.number_of_free_variables * (
            self.config.prediction_horizon - 2
        )
        jerk_columns = self.number_of_free_variables * self.config.prediction_horizon
        result = vel_columns
        result += jerk_columns
        return result

    def derivative_link_model_no_acc(self, max_derivative: Derivatives) -> sm.Matrix:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------|
        | -1  |     |     |dt**2|     |     |     |     |       -vc - ac*dt = -v0 + j0*dt**2
        |  2  | -1  |     |     |dt**2|     |     |     |                vc = -v1 + 2*v0 + j1*dt**2
        | -1  |  2  | -1  |     |     |dt**2|     |     |                 0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
        |     | -1  |  2  |     |     |     |dt**2|     |                 0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
        |     |     | -1  |     |     |     |     |dt**2|                 0 =   0 +    0   - vt-2 + jt*dt**2
        |-----------------------------------------------|
        vt = vt-1 + at * dt     <=>  vt/dt - vt-1/dt = at
        at = at-1 + jt * dt     <=>  at/dt - at-1/dt = jt

        vt = vt-1 + (at-1 + jt * dt) * dt
        vt = vt-1 + at-1*dt + jt * dt**2
        vt = vt-1 + (vt-1/dt - vt-2/dt)*dt + jt * dt**2
        vt = vt-1 + vt-1 - vt-2 + jt*dt**2

        0 = -v1 + 2*v0 - vc + j1*dt**2
        vc = -v1 + 2*v0 + j1*dt**2

        a0/dt - ac/dt = j0
        (v0/dt - vc/dt)/dt - ac/dt = j0
        v0/dt**2 - vc/dt**2 - ac/dt = j0
        -vc/dt**2 - ac/dt = -v0/dt**2 + j0
        -vc - ac*dt = -v0 + j0*dt**2

        v0 = vc + ac*dt + j0*dt**2
        vc = -v1 + 2*v0 + j1*dt**2
        v1 = -vc + 2*v0 + j1*dt**2
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * self.config.prediction_horizon
        model = sm.Matrix.zeros(rows=n_jerk, columns=n_jerk + n_vel)
        pre_previous = -sm.Matrix.eye(n_vel)
        same = pre_previous
        previous = -2 * pre_previous
        j_same = sm.Matrix.eye(n_jerk)  # * self.config.mpc_dt**2
        # j_same = sm.Matrix.eye(n_jerk) * self.config.mpc_dt**2
        model[: -self.number_of_free_variables * 2, :n_vel] += pre_previous
        model[
            self.number_of_free_variables : -self.number_of_free_variables, :n_vel
        ] += previous
        model[self.number_of_free_variables * 2 :, :n_vel] += same
        model[:, n_vel:] = j_same
        return model

    def _remove_rows_columns_where_variables_are_zero(
        self, derivative_link_model: sm.Matrix
    ) -> sm.Matrix:
        if np.prod(derivative_link_model.shape) == 0:
            return derivative_link_model
        row_ids = []
        end = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative - 1
        ):
            last_non_zero_variable = self.config.prediction_horizon - (
                self.config.max_derivative - derivative - 1
            )
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.config.prediction_horizon
            row_ids.extend(range(start, end))
        derivative_link_model.remove(row_ids, [])
        return derivative_link_model

    def equality_constraint_model(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|

        explicit no acc
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------|
        |  J1*sp |  J1*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------|
        """
        if len(self.constraint_collection.equality_constraints) > 0:
            model = sm.Matrix.zeros(
                len(self.constraint_collection.equality_constraints),
                self.number_of_non_slack_columns,
            )
            J_eq = (
                sm.Matrix(self.equality_constraint_expressions()).jacobian(
                    variables=self.get_free_variable_symbols(Derivatives.position)
                )
                * self.config.mpc_dt
            )
            J_hstack = sm.hstack(
                [J_eq for _ in range(self.config.prediction_horizon - 2)]
            )
            # set jacobian entry to 0 if control horizon shorter than prediction horizon
            horizontal_offset = J_hstack.shape[1]
            model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = sm.Matrix.diag(
                [
                    self.config.mpc_dt
                    for _ in self.constraint_collection.equality_constraints
                ]
            )
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    def construct_expression(
        self,
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        max_derivative = Derivatives.velocity
        derivative_link_model = sm.Matrix()
        max_derivative = Derivatives.velocity
        derivative_link_model = self.derivative_link_model_no_acc(
            self.config.max_derivative
        )
        equality_constraint_model, equality_constraint_slack_model = (
            self.equality_constraint_model()
        )
        equality_constraint_model = (
            self._remove_columns_columns_where_variables_are_zero(
                equality_constraint_model, max_derivative
            )
        )
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        model_parts = []
        slack_model_parts = []
        if len(derivative_link_model) > 0:
            model_parts.append(derivative_link_model)
        if len(equality_constraint_model) > 0:
            model_parts.append(equality_constraint_model)
            slack_model_parts.append(equality_constraint_slack_model)
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)

        if len(model_parts) == 0:
            # if there are no eq constraints, make an empty matrix with the right columns to prevent stacking issues
            model = sm.Matrix(np.empty((0, self.number_of_non_slack_columns)))
        else:
            model = sm.vstack(model_parts)
        slack_model = sm.diag_stack(slack_model_parts)

        slack_model = sm.vstack(
            [
                sm.Matrix.zeros(derivative_link_model.shape[0], slack_model.shape[1]),
                slack_model,
            ]
        )
        # model = self._remove_columns_columns_where_variables_are_zero(model, max_derivative)
        return model, slack_model


@dataclass
class InequalityModel(ProblemDataPart):
    """
    order:
        derivative model (optional)
        neq integral constraints
        neq vel constraints
    """

    @property
    def number_of_free_variables(self):
        return len(self.degrees_of_freedom)

    @property
    def number_of_non_slack_columns(self) -> int:
        return (
            self.number_of_free_variables * (self.config.prediction_horizon - 2)
            + self.number_of_free_variables * self.config.prediction_horizon
        )

    @memoize
    def num_position_limits(self):
        return self.number_of_free_variables - self.num_of_continuous_joints()

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.degrees_of_freedom if not v.has_position_limits()])

    def inequality_constraint_expressions(self) -> List[sm.Matrix]:
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.inequality_constraints
            }
        )[0]

    def get_derivative_constraint_expressions(self, derivative: Derivatives):
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.derivative_inequality_constraints
                if c.derivative == derivative
            }
        )[0]

    def get_free_variable_symbols(self, order: Derivatives):
        return _sorter(
            {
                v.variables.position.name: v.variables.data[order]
                for v in self.degrees_of_freedom
            }
        )[0]

    def velocity_constraint_model(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(
            self.constraint_collection.velocity_inequality_constraints
        ) * (self.config.prediction_horizon - 2)
        if number_of_vel_rows > 0:
            expressions = sm.Matrix(
                self.get_derivative_constraint_expressions(Derivatives.velocity)
            )
            parts = []
            for derivative in Derivatives.range(
                Derivatives.position, self.config.max_derivative - 1
            ):
                if derivative == Derivatives.velocity:
                    continue
                J_vel = (
                    expressions.jacobian(
                        variables=self.get_free_variable_symbols(derivative),
                    )
                    * self.config.mpc_dt
                )
                missing_variables = self.config.max_derivative - derivative - 1
                eye = sm.Matrix.eye(self.config.prediction_horizon)[
                    :-2, : self.config.prediction_horizon - missing_variables
                ]
                J_vel_limit_block = eye.kron(J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = sm.hstack(parts)
            num_slack_variables = sum(
                self.control_horizon
                for c in self.constraint_collection.velocity_inequality_constraints
            )
            slack_model = sm.Matrix.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    def inequality_constraint_model(
        self, max_derivative: Derivatives
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|
        """
        if len(self.constraint_collection.inequality_constraints) > 0:
            model = sm.Matrix.zeros(
                len(self.constraint_collection.inequality_constraints),
                self.number_of_non_slack_columns,
            )
            J_neq = (
                sm.Matrix(self.inequality_constraint_expressions()).jacobian(
                    variables=self.get_free_variable_symbols(Derivatives.position),
                )
                * self.config.mpc_dt
            )
            J_hstack = sm.hstack(
                [J_neq for _ in range(self.config.prediction_horizon - 2)]
            )
            # set jacobian entry to 0 if control horizon shorter than prediction horizon
            horizontal_offset = J_hstack.shape[1]
            model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = sm.Matrix.diag(
                [
                    self.config.mpc_dt
                    for _ in self.constraint_collection.inequality_constraints
                ]
            )
            return model, slack_model
        if len(self.constraint_collection.inequality_constraints) > 0:
            model = sm.Matrix.zeros(
                len(self.constraint_collection.inequality_constraints),
                self.number_of_non_slack_columns,
            )
            for derivative in Derivatives.range(
                Derivatives.position, max_derivative - 1
            ):
                J_neq = (
                    sm.Matrix(self.inequality_constraint_expressions()).jacobian(
                        variables=self.get_free_variable_symbols(derivative),
                    )
                    * self.config.mpc_dt
                )
                J_hstack = sm.hstack(
                    [J_neq for _ in range(self.config.prediction_horizon - 2)]
                )
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = sm.Matrix.diag(
                [
                    self.config.mpc_dt
                    for _ in self.constraint_collection.inequality_constraints
                ]
            )
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    def implicit_pos_limits(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        pk = pk-1 + vk*dt
            p0 = pc + v0*dt
            pk/dt - pk-1/dt = vk
        vk = vk-1 + ak*dt
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        Layout for prediction horizon 4
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1*dt  |        |        |        |       pt0 - ptc = vt0*dt
        |  1*dt  |  1*dt  |        |        |       pt1 - ptc = vt0*dt + vt1*dt
        |  1*dt  |  1*dt  |  1*dt  |        |       pt2 - ptc = vt0*dt + vt1*dt + vt2*dt
        |  1*dt  |  1*dt  |  1*dt  |  1*dt  |       pt3 - ptc = vt0*dt + vt1*dt + vt2*dt + vt3*dt
        |-----------------------------------|

        :return:
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        model = sm.Matrix.tri(n_vel) * self.config.mpc_dt
        slack_model = sm.Matrix.zeros(model.shape[0], self.number_ineq_slack_variables)
        return model, slack_model

    def implicit_model(
        self, max_derivative: Derivatives
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        vt0 = vtc + at0 * cdt
        vt1 = vt0 + at1 * mdt
        vt = vt-1 + at * mdt

        at = vt-1 + jt * mdt

        Layout for prediction horizon 6
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1/dt  |        |        |        |               vtc/cdt + at0 = vt0/cdt                   vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
        | -1/dt  |  1/dt  |        |        |                        at1 = (vt1 - vt0)/dt
        |        | -1/dt  |  1/dt  |        |                        at2 = (vt2 - vt1)/mdt
        |        |        | -1/dt  | 1/dt   |                        at3 = (vt3 - vt2)/mdt
        |===================================|
        | 1/dt**2|        |        |        |   vtc/dt**2 + atc/dt + jt0 = vt0/dt**2                vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
        |-2/dt**2| 1/dt**2|        |        |          - vtc/dt**2 + jt1 = (vt1 - 2vt0)/dt**2           (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
        | 1/dt**2|-2/dt**2| 1/dt**2|        |                        jt2 = (vt2 - 2vt1 + vt0)/dt**2
        |        | 1/dt**2|-2/dt**2| 1/dt**2|                        jt3 = (vt3 - 2vt2 + vt1)/dt**2
        |        |        | 1/dt**2|-2/dt**2|                        jt4 = (- 2vt3 + vt2)/dt**2
        |        |        |        | 1/dt**2|                        jt5 = (vt3)/dt**2
        |-----------------------------------|

        :param max_derivative:
        :return:
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * (self.config.prediction_horizon)
        if max_derivative >= Derivatives.acceleration:
            model = sm.Matrix.zeros(rows=n_jerk, columns=n_vel)
            pre_previous = sm.Matrix.eye(n_vel) / self.config.mpc_dt**2
            previous = -2 * sm.Matrix.eye(n_vel) / self.config.mpc_dt**2
            same = pre_previous
            model[: -self.number_of_free_variables * 2, :] += pre_previous
            model[
                self.number_of_free_variables : -self.number_of_free_variables, :
            ] += previous
            model[self.number_of_free_variables * 2 :, :] += same
        else:
            model = sm.Matrix()
        slack_model = sm.Matrix.zeros(model.shape[0], self.number_ineq_slack_variables)
        return model, slack_model

    def construct_expression(
        self,
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        model_parts = []
        slack_model_parts = []

        max_derivative = self.config.max_derivative

        inequality_model, inequality_slack_model = self.inequality_constraint_model(
            max_derivative
        )
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        # neq integral constraints
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)
        # neq vel constraints
        if len(inequality_model) > 0:
            model_parts.append(inequality_model)
            slack_model_parts.append(inequality_slack_model)

        combined_model = sm.vstack(model_parts)
        combined_slack_model = sm.diag_stack(slack_model_parts)
        return combined_model, combined_slack_model


@dataclass
class QPDataSymbolic:
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:
    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Edof x <= bE_dof          (equality constraints)
          Eslack x <= bE_slack        (equality constraints)
          lbA <= Adof x <= ubA_dof  (lower/upper inequality constraints)
          lbA <= Aslack x <= ubA_slack  (lower/upper inequality constraints)
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    quadratic_weights: Vector = field(init=False)
    linear_weights: Vector = field(init=False)

    box_lower_constraints: Vector = field(init=False)
    box_upper_constraints: Vector = field(init=False)

    eq_matrix_dofs: Matrix = field(init=False)
    eq_matrix_slack: Matrix = field(init=False)
    eq_bounds: Vector = field(init=False)
    eq_constraint_names: List[str] = field(init=False)

    neq_matrix_dofs: Matrix = field(init=False)
    neq_matrix_slack: Matrix = field(init=False)
    neq_lower_bounds: Vector = field(init=False)
    neq_upper_bounds: Vector = field(init=False)

    def __post_init__(self):
        direct_limits = DofLimits.create(self.degrees_of_freedom, self.config)
        mpc_model = EqualityDerivativeLinkModel(
            degrees_of_freedom=self.degrees_of_freedom,
            constraint_collection=self.constraint_collection,
            config=self.config,
        )
        eq_constraints = EqualityConstraintModel(
            degrees_of_freedom=self.degrees_of_freedom,
            constraint_collection=self.constraint_collection,
            config=self.config,
        )
        ineq_constraints = InequalityConstraintModel(
            degrees_of_freedom=self.degrees_of_freedom,
            constraint_collection=self.constraint_collection,
            config=self.config,
        )
        self.quadratic_weights = sm.concatenate(
            direct_limits.quadratic_weights,
            eq_constraints.slack_variables.quadratic_weights,
            ineq_constraints.slack_variables.quadratic_weights,
        )
        self.linear_weights = sm.concatenate(
            direct_limits.linear_weights,
            eq_constraints.slack_variables.linear_weights,
            ineq_constraints.slack_variables.linear_weights,
        )
        self.box_lower_constraints = sm.concatenate(
            direct_limits.lower_bounds,
            eq_constraints.slack_variables.lower_bounds,
            ineq_constraints.slack_variables.lower_bounds,
        )
        self.box_upper_constraints = sm.concatenate(
            direct_limits.upper_bounds,
            eq_constraints.slack_variables.upper_bounds,
            ineq_constraints.slack_variables.upper_bounds,
        )

        self.eq_matrix_dofs = sm.vstack([mpc_model.matrix, eq_constraints.matrix])
        self.eq_matrix_slack = sm.diag_stack(
            [mpc_model.slack_matrix, eq_constraints.slack_matrix]
        )
        self.eq_bounds = sm.concatenate(mpc_model.bounds, eq_constraints.bounds)
        self.eq_constraint_names = (
            mpc_model.constraint_names + eq_constraints.constraint_names
        )

        self.neq_matrix_dofs = sm.vstack([ineq_constraints.matrix])
        self.neq_matrix_slack = sm.diag_stack([ineq_constraints.slack_matrix])
        self.neq_lower_bounds = ineq_constraints.lower_bounds
        self.neq_upper_bounds = ineq_constraints.upper_bounds
        self.neq_constraint_names = ineq_constraints.constraint_names

    def __hash__(self):
        return hash(id(self))

    @property
    def num_eq_constraints(self) -> int:
        return len(self.constraint_collection.equality_constraints)

    @property
    def num_neq_constraints(self) -> int:
        return len(self.constraint_collection.inequality_constraints)

    @property
    def num_free_variable_constraints(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def num_eq_slack_variables(self) -> int:
        return self.eq_matrix_slack.shape[1]

    @property
    def num_neq_slack_variables(self) -> int:
        return self.neq_matrix_slack.shape[1]

    @property
    def num_slack_variables(self) -> int:
        return self.num_eq_slack_variables + self.num_neq_slack_variables

    @property
    def num_non_slack_variables(self) -> int:
        return self.quadratic_weights.shape[0] - self.num_slack_variables
