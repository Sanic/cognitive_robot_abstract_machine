from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from copy import copy
from dataclasses import dataclass, field
from itertools import product
from typing import List, TYPE_CHECKING, Type
from uuid import UUID

import giskardpy.utils.math as gm
import krrood.symbolic_math.symbolic_math as sm
import numpy as np
from giskardpy.motion_statechart.data_types import FloatEnum
from giskardpy.qp.exceptions import (
    InfeasibleException,
    VelocityLimitUnreachableException,
)
from giskardpy.qp.pos_in_vel_limits import (
    shifted_velocity_profile,
    compute_slowdown_asap_vel_profile,
)
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.math import mpc
from krrood.symbolic_math.symbolic_math import Vector, Matrix, Scalar, FloatVariable
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from typing_extensions import Self, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


LargeNumber = 1e4


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


@dataclass
class DirectLimits(ABC):
    """
    Represents weights and limits of decision variables in a QP.
    All fields must have the same length.
    Subclasses must implement the `create` factory method, which should be used for creating it.
    """

    lower_bounds: sm.Vector = field(init=False)
    upper_bounds: sm.Vector = field(init=False)
    quadratic_weights: sm.Vector = field(init=False)
    linear_weights: sm.Vector = field(init=False)
    names: list[str] = field(init=False)

    @classmethod
    def create(
        cls,
        degrees_of_freedom: List[DegreeOfFreedom],
        config: QPControllerConfig,
    ) -> Self:
        pass


@dataclass
class SlackLimits(DirectLimits):
    """
    Implements
    """

    @classmethod
    def from_constraints(
        cls, constraints: list[GiskardConstraint], config: QPControllerConfig
    ):
        self = cls()
        num_of_slack_variables = len(constraints)
        self.quadratic_weights = Vector(
            [
                self.normalized_weight(
                    quadratic_weight=c.quadratic_weight,
                    control_horizon=config.velocity_horizon,
                    normalization_number=c.normalization_factor,
                )
                for c in constraints
            ]
        )
        self.linear_weights = Vector(
            [
                self.normalized_weight(
                    quadratic_weight=c.linear_weight,
                    control_horizon=config.velocity_horizon,
                    normalization_number=c.normalization_factor,
                )
                for c in constraints
            ]
        )
        self.lower_bounds = Vector([-LargeNumber] * num_of_slack_variables)
        self.upper_bounds = Vector([LargeNumber] * num_of_slack_variables)
        self.names = [c.name for c in constraints]
        return self

    def normalized_weight(
        self,
        quadratic_weight: Scalar,
        normalization_number: float,
        control_horizon: int,
    ) -> Scalar:
        return quadratic_weight * (
            1 / (sm.Scalar(normalization_number) ** 2 * control_horizon)
        )


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
        self.make_names(degrees_of_freedom, config)
        return self

    def make_names(
        self, degrees_of_freedom: List[DegreeOfFreedom], config: QPControllerConfig
    ):
        self.names = []
        for derivative in ["vel", "jerk"]:
            for k in range(config.prediction_horizon):
                if derivative == "vel" and k > config.prediction_horizon - 3:
                    continue
                for dof in degrees_of_freedom:
                    self.names.append(f"{dof.name}_{derivative}_k_{k}")

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
        degrees_of_freedom: list[DegreeOfFreedom],
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
        degrees_of_freedom: list[DegreeOfFreedom],
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
class EnforcementStrategy(ABC):
    degrees_of_freedom: list[DegreeOfFreedom]
    constraints: list[GiskardConstraint]
    config: QPControllerConfig

    @abstractmethod
    def create_matrix(self) -> Matrix: ...

    @abstractmethod
    def create_slack_matrix(self) -> Matrix: ...

    @abstractmethod
    def create_names(self) -> list[str]: ...

    @abstractmethod
    def create_slack_variables(self) -> DirectLimits: ...

    @abstractmethod
    def create_bounds(
        self, bounds_getter: Callable[GiskardConstraint, Scalar]
    ) -> Vector: ...

    def create_lower_bounds(self) -> Vector:
        for c in self.constraints:
            assert isinstance(c, GiskardInequalityConstraint)
        return self.create_bounds(lambda c: c.lower_bound)

    def create_upper_bounds(self) -> Vector:
        for c in self.constraints:
            assert isinstance(c, GiskardInequalityConstraint)
        return self.create_bounds(lambda c: c.upper_bound)

    def create_equality_bounds(self) -> Vector:
        for c in self.constraints:
            assert isinstance(c, GiskardEqualityConstraint)
        return self.create_bounds(lambda c: c.bound)

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


@dataclass
class IntegralStrategy(EnforcementStrategy):
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

    def create_matrix(self) -> Matrix:
        if len(self.constraints) == 0:
            return sm.Matrix()
        jacobian = (
            sm.Vector([c.expression for c in self.constraints]).jacobian(
                variables=self.position_variables
            )
            * self.config.mpc_dt
        )
        return sm.hstack(
            [jacobian for _ in range(self.config.velocity_horizon)]
            + [sm.Matrix.zeros(jacobian.shape[0], self.number_of_jerk_columns)]
        )

    def create_slack_matrix(self) -> Matrix:
        if len(self.constraints) == 0:
            return sm.Matrix()
        return sm.Matrix.diag([self.config.mpc_dt for _ in self.constraints])

    def create_slack_variables(self) -> DirectLimits:
        return SlackLimits.from_constraints(
            constraints=self.constraints,
            config=self.config,
        )

    def _apply_cap(
        self,
        value: Scalar,
        dt: float,
        normalization_number: float,
        control_horizon: int,
    ) -> Scalar:
        # todo normalization with jacobian???
        return sm.limit(
            value,
            -normalization_number * dt * control_horizon,
            normalization_number * dt * control_horizon,
        )

    def capped_bound(
        self,
        equality_bound: Scalar,
        dt: float,
        normalization_number: float,
        control_horizon: int,
    ) -> Scalar:
        return self._apply_cap(
            equality_bound, dt, normalization_number, control_horizon
        )

    def create_bounds(
        self, bounds_getter: Callable[GiskardConstraint, Scalar]
    ) -> Vector:
        return Vector(
            [
                self.capped_bound(
                    bounds_getter(c),
                    self.config.mpc_dt,
                    c.normalization_factor,
                    self.config.velocity_horizon,
                )
                for c in self.constraints
            ]
        )

    def create_names(self) -> list[str]:
        return [c.name for c in self.constraints]


@dataclass
class VelocityStrategy(EnforcementStrategy):
    """
    The constraint will be applied to the derivative of the expression.
    Position constraints are implemented by constraining the integral of the expressions' derivative over a prediction horizon.
    All other constraints are applied directly to that derivative of the expression.
    As a result, position constraints are cheaper, as they only require a single constraint.
    """

    # normalization_factor: sm.ScalarData = field(kw_only=True)
    """
    This value is important to make constraints with different units comparable.
    The meaning depends on derivative.
    If the derivative is position, the normalization factor is rough velocity with which the expression can change.
    For example:
        - If you have a joint position constraint, the normalization factor should be the joint velocity limit.
        - If you have a cartesian position constraint, the normalization factor should be the cartesian velocity limit.
    For other derivatives, the normalization factor is the same unit as the expression.
    For example:
        - Joint velocity constraint -> joint velocity limit
        - Cartesian velocity constraint -> cartesian velocity limit
    .. Warning: This number is different from the bounds of the expression. 
                If you want to enforce a bound below the actual limit, the normalization factor should still be the true limit.
    In practice, use joint limits from the URDF for joint space constraints and define two values for cartesian constraints:
        - a m/s limit for translation
        - a rad/s value for rotation
    """
    """
    Equality constraints have the form:
    .. math::
        f(q) = b

    where

    .. math::

        target - f = \Delta t \sum_{k=0}^{N-1} J_{f} * sp_k

    ::

        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|
    """

    def create_matrix(self) -> Matrix:
        number_of_vel_rows = len(self.constraints) * (
            self.config.prediction_horizon - 2
        )
        if number_of_vel_rows == 0:
            return sm.Matrix()
        jacobian = (
            sm.Vector([c.expression for c in self.constraints]).jacobian(
                variables=self.position_variables
            )
            * self.config.mpc_dt
        )
        missing_variables = self.config.max_derivative - 1
        eye = sm.Matrix.eye(self.config.prediction_horizon)[
            :-2, : self.config.prediction_horizon - missing_variables
        ]
        J_vel_limit_block = eye.kron(jacobian)

        return sm.hstack(
            [
                J_vel_limit_block,
                sm.Matrix.zeros(
                    J_vel_limit_block.shape[0], self.number_of_jerk_columns
                ),
            ]
        )

    def create_slack_matrix(self) -> Matrix:
        if len(self.constraints) == 0:
            return sm.Matrix()
        num_slack_variables = sum(
            self.config.prediction_horizon - 2 for c in self.constraints
        )
        return sm.Matrix.eye(num_slack_variables) * self.config.mpc_dt

    def create_slack_variables(self) -> DirectLimits:
        lower_slack = []
        upper_slack = []
        quadratic_weights = []
        linear_weights = []
        names = []
        for t in range(self.config.prediction_horizon):
            for c in self.constraints:
                if t < self.config.prediction_horizon - 2:
                    lower_slack.append(c.lower_slack_limit)
                    upper_slack.append(c.upper_slack_limit)
                    quadratic_weights.append(
                        self.normalized_weight(
                            c.quadratic_weight, c.normalization_factor
                        )
                    )
                    linear_weights.append(c.linear_weight)
                    names.append(f"t{t:03}/{c.name}")
        limits = SlackLimits()
        limits.lower_bounds = sm.Vector(lower_slack)
        limits.upper_bounds = sm.Vector(upper_slack)
        limits.quadratic_weights = sm.Vector(quadratic_weights)
        limits.linear_weights = sm.Vector(linear_weights)
        limits.names = names
        return limits

    def normalized_weight(
        self, quadratic_weight: Scalar, normalization_factor
    ) -> Scalar:
        return quadratic_weight * (1 / sm.Scalar(normalization_factor)) ** 2

    def create_bounds(
        self, bounds_getter: Callable[GiskardConstraint, Scalar]
    ) -> Vector:
        bounds2 = []
        for t in range(self.config.control_horizon):
            for c in self.constraints:
                bounds2.append(bounds_getter(c) * self.config.mpc_dt)
        return Vector(bounds2)

    def create_names(self) -> list[str]:
        names = []
        for t in range(self.config.control_horizon):
            for c in self.constraints:
                names.append(f"t{t:03}/{c.name}")
        return names


@dataclass
class SystemDynamicsStrategy(EnforcementStrategy):
    """
    The constraints produced by this class describe the discrete-time relationships between variables
    in the prediction horizon :math:`N` using a semi-implicit euler integration method:

    .. math::

        v_k = v_{k-1} + a_{k} \\, \\Delta t

        a_k = a_{k-1} + j_{k} \\, \\Delta t

    Where v, a and j are velocity, acceleration and jerk, respectively, and k is the time step.
    Acceleration variables are removed using substitution.
    The first two row links the MPC to the current state:

    .. math::

        -v_{current} - a_{current} \\, \\Delta t = -v_0 + j_0 \\, \\Delta t^2

        v_{current} = - v_1 + 2 v_0 + j_1 \\, \\Delta t^2

    Row from 2 until k-2 have this form:

    .. math::

        0 = - v_k + 2 v_{k-1} - v_{k-2} + j_k \\, \\Delta t^2

    The final two rows have this form:

    .. math::

        0 = 2 v_{k-1} - v_{k-2} + j_k \\, \\Delta t^2

        0 = - v_{k-2} + j_k \\, \\Delta t^2

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

    def create_matrix(self) -> Matrix:
        matrix = np.zeros(
            (
                self.number_of_jerk_columns,
                self.number_of_velocity_columns + self.number_of_jerk_columns,
            )
        )
        for k in range(self.config.prediction_horizon):
            row_start = k * self.number_of_free_variables
            row_end = (k + 1) * self.number_of_free_variables

            # velocity at k
            if k < self.config.prediction_horizon - 2:
                col_start = k * self.number_of_free_variables
                col_end = (k + 1) * self.number_of_free_variables
                matrix[row_start:row_end, col_start:col_end] -= np.eye(
                    self.number_of_free_variables
                )

            # velocity at k-1
            if 0 < k < self.config.prediction_horizon - 1:
                col_start = (k - 1) * self.number_of_free_variables
                col_end = k * self.number_of_free_variables
                matrix[row_start:row_end, col_start:col_end] += 2 * np.eye(
                    self.number_of_free_variables
                )

            # velocity at k-2
            if k > 1:
                col_start = (k - 2) * self.number_of_free_variables
                col_end = (k - 1) * self.number_of_free_variables
                matrix[row_start:row_end, col_start:col_end] -= np.eye(
                    self.number_of_free_variables
                )

            # jerk at k
            col_start = self.number_of_velocity_columns + row_start
            col_end = self.number_of_velocity_columns + row_end
            matrix[row_start:row_end, col_start:col_end] += np.eye(
                self.number_of_free_variables
            )

        return sm.Matrix(matrix)

    def create_slack_matrix(self) -> Matrix:
        return sm.Matrix.zeros(self.number_of_jerk_columns, 0)

    def create_names(self) -> list[str]:
        names = []
        for k in range(self.config.prediction_horizon):
            for dof in self.degrees_of_freedom:
                names.append(f"{dof.name} k_{k} vel/jerk link")
        return names

    def create_slack_variables(self) -> DirectLimits:
        return SlackLimits()

    def create_bounds(
        self, bounds: list[Scalar], normalization_numbers: list[float]
    ) -> Vector:
        res = sm.Vector.zeros(self.number_of_jerk_columns)
        res[: self.number_of_free_variables] = (
            -self.velocity_variables - self.acceleration_variables * self.config.mpc_dt
        )
        res[self.number_of_free_variables : self.number_of_free_variables * 2] = (
            self.velocity_variables
        )
        return res


@dataclass(kw_only=True)
class GiskardConstraint(ABC):
    """
    Defines a (slack-relaxed) constraint on expression for a quadratic program.
    """

    name: str

    expression: Scalar

    quadratic_weight: sm.ScalarData

    linear_weight: sm.ScalarData = field(default=0)

    normalization_factor: float
    """
    This value is important to make constraints with different units comparable.
    The meaning depends on derivative.
    If the derivative is position, the normalization factor is rough velocity with which the expression can change.
    For example:
        - If you have a joint position constraint, the normalization factor should be the joint velocity limit.
        - If you have a cartesian position constraint, the normalization factor should be the cartesian velocity limit.
    In practice, use joint limits from the URDF for joint space constraints and define two values for cartesian constraints:
        - a m/s limit for translation
        - a rad/s value for rotation
    """

    enforcement_strategy: type[EnforcementStrategy]
    """
    The strategy used to enforce this constraint within the QP.
    """

    lower_slack_limit: sm.ScalarData = field(default=-LargeNumber)
    upper_slack_limit: sm.ScalarData = field(default=LargeNumber)


@dataclass
class GiskardEqualityConstraint(GiskardConstraint):
    bound: Scalar


@dataclass
class GiskardInequalityConstraint(GiskardConstraint):
    lower_bound: Scalar
    upper_bound: Scalar
