"""
Per-degree-of-freedom decision-variable limits, weights, and the MPC-based velocity profiles
that keep each degree of freedom within its position limits across the prediction horizon.
"""

from __future__ import annotations

import enum
import logging
from copy import copy
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

import giskardpy.utils.math as gm
import krrood.symbolic_math.symbolic_math as sm
import numpy as np
from giskardpy.qp.exceptions import (
    InfeasibleException,
    MismatchedLimitLengthsError,
    VelocityLimitUnreachableException,
)
from giskardpy.qp.pos_in_vel_limits import (
    shifted_velocity_profile,
    compute_immediate_slowdown_profile,
)
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.math import mpc
from krrood.symbolic_math.symbolic_math import Scalar, FloatVariable
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


def max_velocity_from_horizon_and_jerk_qp(
    prediction_horizon: int,
    vel_limit: float,
    acc_limit: float,
    jerk_limit: float,
    dt: float,
    solver_class: type[QPSolver],
):
    """
    Computes the highest velocity reachable within the prediction horizon under the given
    acceleration and jerk limits, by solving an MPC that maximizes velocity without a binding
    velocity limit.
    """
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


NUMBER_OF_JERK_RELAXED_STEPS = 3
"""
Number of initial horizon steps whose jerk limit may be relaxed to keep the position goal
reachable.
"""


class VelocityBoundProfiles(NamedTuple):
    """
    Per-step velocity bounds of a degree of freedom across the prediction horizon, together with
    the goal velocity profile and the flag marking when the first step is already at rest against
    a position limit.
    """

    lower_bound: sm.Vector
    """Lower velocity bound at each step of the prediction horizon."""

    upper_bound: sm.Vector
    """Upper velocity bound at each step of the prediction horizon."""

    goal_profile: sm.Vector
    """Goal velocity profile derived from the lower and upper bounds."""

    skip_first: sm.Scalar
    """Flag marking that the first step is already at rest against a position limit."""


class BoundDirection(enum.Enum):
    """
    Selects whether a velocity bound brakes the degree of freedom against its lower or upper
    position limit.
    """

    LOWER = -1
    """Bound that brakes against the lower position limit."""

    UPPER = 1
    """Bound that brakes against the upper position limit."""

    @property
    def sign(self) -> int:
        """
        Returns the sign that mirrors the upper-bound computation onto the lower bound.
        """
        return self.value


@dataclass
class DirectLimits:
    """
    Represents weights and limits of decision variables in a QP.
    All fields must have the same length.
    """

    lower_bounds: sm.Vector
    """
    Lower box limit of each decision variable.
    """

    upper_bounds: sm.Vector
    """
    Upper box limit of each decision variable.
    """

    quadratic_weights: sm.Vector
    """
    Quadratic objective weight of each decision variable.
    """

    linear_weights: sm.Vector
    """
    Linear objective weight of each decision variable.
    """

    names: list[str]
    """
    Human readable name of each decision variable, used for debugging.
    """

    def __post_init__(self):
        """
        Ensures all bound, weight, and name fields describe the same number of decision variables.
        """
        lengths = {
            "lower_bounds": self.lower_bounds.shape[0],
            "upper_bounds": self.upper_bounds.shape[0],
            "quadratic_weights": self.quadratic_weights.shape[0],
            "linear_weights": self.linear_weights.shape[0],
            "names": len(self.names),
        }
        if len(set(lengths.values())) > 1:
            raise MismatchedLimitLengthsError(field_lengths=lengths)

    @classmethod
    def empty(cls) -> DirectLimits:
        """
        Creates a DirectLimits without any decision variables.
        """
        return cls(
            lower_bounds=sm.Vector([]),
            upper_bounds=sm.Vector([]),
            quadratic_weights=sm.Vector([]),
            linear_weights=sm.Vector([]),
            names=[],
        )


@dataclass
class DegreeOfFreedomLimitProfiler:
    """
    Computes the per-degree-of-freedom velocity, acceleration, and jerk bounds across the
    prediction horizon, including the MPC-based position-aware slowdown profiles.
    """

    qp_controller_config: QPControllerConfig
    """
    Controller configuration providing horizon length, time step, and solver.
    """

    def _compute_position_constrained_velocity_bounds(
        self,
        dof_symbols: DerivativeMap[FloatVariable],
        lower_limits: DerivativeMap[float],
        upper_limits: DerivativeMap[float],
        solver_class: type[QPSolver],
        time_step: float,
        prediction_horizon: int,
    ) -> VelocityBoundProfiles:
        """
        Computes per-step velocity bounds that keep the degree of freedom within its position
        limits, slowing it down early enough to stop before a limit is reached.
        """
        velocity_limit = upper_limits.velocity
        if lower_limits.position is None:
            return self._unconstrained_velocity_bounds(
                velocity_limit, prediction_horizon
            )

        jerk_limit = upper_limits.jerk
        position_range = upper_limits.position - lower_limits.position
        velocity_limit = min(velocity_limit * time_step, position_range / 2) / time_step
        mpc_velocity_profile, mpc_acceleration_profile = (
            self._nominal_mpc_velocity_profile(
                velocity_limit=velocity_limit,
                acceleration_limit=upper_limits.acceleration,
                jerk_limit=jerk_limit,
                time_step=time_step,
                prediction_horizon=prediction_horizon,
                solver_class=solver_class,
            )
        )
        velocity_lower_bound = self._directional_velocity_bound(
            mpc_velocity_profile=mpc_velocity_profile,
            mpc_acceleration_profile=mpc_acceleration_profile,
            position_error=lower_limits.position - dof_symbols.position,
            jerk_limit=jerk_limit,
            velocity_limit=velocity_limit,
            time_step=time_step,
            direction=BoundDirection.LOWER,
        )
        velocity_upper_bound = self._directional_velocity_bound(
            mpc_velocity_profile=mpc_velocity_profile,
            mpc_acceleration_profile=mpc_acceleration_profile,
            position_error=upper_limits.position - dof_symbols.position,
            jerk_limit=jerk_limit,
            velocity_limit=velocity_limit,
            time_step=time_step,
            direction=BoundDirection.UPPER,
        )
        goal_profile = sm.max(velocity_lower_bound, 0) + sm.min(velocity_upper_bound, 0)
        skip_first = sm.logic_or(
            velocity_lower_bound[0] >= 0, velocity_upper_bound[0] <= 0
        )
        return VelocityBoundProfiles(
            lower_bound=velocity_lower_bound,
            upper_bound=velocity_upper_bound,
            goal_profile=goal_profile,
            skip_first=skip_first,
        )

    def _unconstrained_velocity_bounds(
        self, velocity_limit: float, prediction_horizon: int
    ) -> VelocityBoundProfiles:
        """
        Builds flat velocity bounds at the velocity limit for a degree of freedom without
        position limits.
        """
        velocity_upper_bound = sm.Vector.ones(prediction_horizon) * velocity_limit
        velocity_lower_bound = -velocity_upper_bound
        return VelocityBoundProfiles(
            lower_bound=velocity_lower_bound,
            upper_bound=velocity_upper_bound,
            goal_profile=sm.Vector.zeros(prediction_horizon),
            skip_first=sm.Scalar.const_false(),
        )

    def _nominal_mpc_velocity_profile(
        self,
        velocity_limit: float,
        acceleration_limit: float,
        jerk_limit: float,
        time_step: float,
        prediction_horizon: int,
        solver_class: type[QPSolver],
    ) -> tuple[sm.Vector, sm.Vector]:
        """
        Solves an MPC that drives the degree of freedom from full velocity to rest, returning the
        nominal velocity and acceleration braking profiles.
        """
        profile = gm.simple_mpc(
            vel_limit=velocity_limit,
            acc_limit=acceleration_limit,
            jerk_limit=jerk_limit,
            current_vel=velocity_limit,
            current_acc=0,
            dt=time_step,
            ph=prediction_horizon,
            q_weight=(0, 0, 0),
            lin_weight=(-1, 0, 0),
            solver_class=solver_class,
        )
        return (
            profile[:prediction_horizon],
            profile[prediction_horizon : prediction_horizon * 2],
        )

    def _directional_velocity_bound(
        self,
        mpc_velocity_profile: sm.Vector,
        mpc_acceleration_profile: sm.Vector,
        position_error: sm.Scalar,
        jerk_limit: float,
        velocity_limit: float,
        time_step: float,
        direction: BoundDirection,
    ) -> sm.Vector:
        """
        Computes the velocity bound that brakes the degree of freedom against one position limit,
        shifting the nominal braking profile by the remaining distance to that limit and capping
        the first step to a single jerk-limited change.
        """
        sign = direction.sign
        velocity_bound, _ = shifted_velocity_profile(
            velocity_profile=mpc_velocity_profile,
            acceleration_profile=mpc_acceleration_profile,
            distance=sign * position_error,
            delta_time=time_step,
        )
        velocity_bound *= sign
        one_step_change = jerk_limit * time_step**2
        one_step_change_bound = sm.limit(
            position_error / time_step,
            min(0.0, -sign * one_step_change),
            max(0.0, -sign * one_step_change),
        )
        one_step_change_bound = sm.limit(
            one_step_change_bound, -velocity_limit, velocity_limit
        )
        velocity_bound[0] = sm.if_less(
            sign * position_error,
            0,
            one_step_change_bound,
            copy(velocity_bound[0]),
        )
        return velocity_bound

    def compute_horizon_bounds(
        self,
        dof_symbols: DerivativeMap[FloatVariable],
        lower_limits: DerivativeMap[float],
        upper_limits: DerivativeMap[float],
        solver_class: type[QPSolver],
        time_step: float,
        prediction_horizon: int,
        epsilon: float = 0.00001,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        """
        Computes the velocity, acceleration, and jerk bounds for one degree of freedom across the
        whole prediction horizon, relaxing the jerk limit on the first steps when the position
        goal would otherwise be unreachable.
        """
        jerk_limit = upper_limits.jerk
        acceleration_limit = upper_limits.acceleration

        velocity_bounds = self._compute_position_constrained_velocity_bounds(
            dof_symbols=dof_symbols,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            solver_class=solver_class,
            time_step=time_step,
            prediction_horizon=prediction_horizon,
        )
        velocity_lower_bound = velocity_bounds.lower_bound
        velocity_upper_bound = velocity_bounds.upper_bound

        acceleration_profile = (
            sm.Vector.ones(velocity_upper_bound.shape[0]) * acceleration_limit
        )
        jerk_profile = sm.Vector.ones(velocity_upper_bound.shape[0]) * jerk_limit

        projected_velocity_profile, projected_jerk_profile_violated = (
            self._project_velocity_profiles(
                dof_symbols=dof_symbols,
                goal_profile=velocity_bounds.goal_profile,
                jerk_limit=jerk_limit,
                time_step=time_step,
                prediction_horizon=prediction_horizon,
                skip_first=velocity_bounds.skip_first,
            )
        )
        needs_relaxed_jerk_limits = self._detect_velocity_bound_violation(
            projected_velocity_profile=projected_velocity_profile,
            velocity_lower_bound=velocity_lower_bound,
            velocity_upper_bound=velocity_upper_bound,
            epsilon=epsilon,
        )
        self._relax_jerk_on_initial_steps(
            jerk_profile=jerk_profile,
            projected_jerk_profile_violated=projected_jerk_profile_violated,
            needs_relaxed_jerk_limits=needs_relaxed_jerk_limits,
            jerk_limit=jerk_limit,
        )
        return self._assemble_degree_of_freedom_limits(
            velocity_lower_bound=velocity_lower_bound,
            velocity_upper_bound=velocity_upper_bound,
            acceleration_profile=acceleration_profile,
            jerk_profile=jerk_profile,
            time_step=time_step,
        )

    def _project_velocity_profiles(
        self,
        dof_symbols: DerivativeMap[FloatVariable],
        goal_profile: sm.Vector,
        jerk_limit: float,
        time_step: float,
        prediction_horizon: int,
        skip_first: sm.Scalar,
    ) -> tuple[sm.Vector, sm.Vector]:
        """
        Projects the slow-down-as-fast-as-possible velocity profile under the real jerk limit and
        the jerk profile that would be required without a jerk limit.
        """
        projected_velocity_profile, _, _ = compute_immediate_slowdown_profile(
            dof_symbols.velocity,
            dof_symbols.acceleration,
            goal_profile,
            Scalar(jerk_limit),
            Scalar(time_step),
            prediction_horizon,
            skip_first,
        )
        _, _, projected_jerk_profile_violated = compute_immediate_slowdown_profile(
            dof_symbols.velocity,
            dof_symbols.acceleration,
            goal_profile,
            Scalar(np.inf),
            Scalar(time_step),
            prediction_horizon,
            skip_first,
        )
        return projected_velocity_profile, projected_jerk_profile_violated

    def _detect_velocity_bound_violation(
        self,
        projected_velocity_profile: sm.Vector,
        velocity_lower_bound: sm.Vector,
        velocity_upper_bound: sm.Vector,
        epsilon: float,
    ) -> sm.Scalar:
        """
        Detects whether the projected velocity profile leaves the velocity bounds or fails to
        come to rest by the end of the horizon, signalling that the jerk limit must be relaxed.
        """
        velocity_lower_bound_violated = sm.logic_or(
            sm.logic_any(projected_velocity_profile < velocity_lower_bound - epsilon),
            sm.abs(projected_velocity_profile[-1]) >= epsilon,
        )
        velocity_upper_bound_violated = sm.logic_or(
            sm.logic_any(projected_velocity_profile > velocity_upper_bound + epsilon),
            sm.abs(projected_velocity_profile[-1]) >= epsilon,
        )
        return sm.logic_or(velocity_lower_bound_violated, velocity_upper_bound_violated)

    def _relax_jerk_on_initial_steps(
        self,
        jerk_profile: sm.Vector,
        projected_jerk_profile_violated: sm.Vector,
        needs_relaxed_jerk_limits: sm.Scalar,
        jerk_limit: float,
    ) -> None:
        """
        Raises the jerk limit on the first horizon steps to the magnitude required to reach the
        position goal, but only when normal braking would otherwise be insufficient.
        """
        for step in range(NUMBER_OF_JERK_RELAXED_STEPS):
            jerk_profile[step] = sm.if_else(
                needs_relaxed_jerk_limits,
                sm.max(
                    Scalar(jerk_limit), sm.abs(projected_jerk_profile_violated[step])
                ),
                sm.Scalar(jerk_limit),
            )

    def _assemble_degree_of_freedom_limits(
        self,
        velocity_lower_bound: sm.Vector,
        velocity_upper_bound: sm.Vector,
        acceleration_profile: sm.Vector,
        jerk_profile: sm.Vector,
        time_step: float,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        """
        Combines the velocity, acceleration, and jerk profiles into the horizon limits, ensuring
        the lower bound never exceeds the upper bound.
        """
        velocity_lower_bound = sm.min(velocity_lower_bound, velocity_upper_bound)
        velocity_upper_bound = sm.max(velocity_lower_bound, velocity_upper_bound)
        acceleration_lower_bounds = -acceleration_profile
        acceleration_upper_bounds = acceleration_profile
        jerk_lower_bounds = sm.min(jerk_profile, -jerk_profile) * time_step**2
        jerk_upper_bounds = sm.max(jerk_profile, -jerk_profile) * time_step**2
        return DegreeOfFreedomLimits[sm.Vector](
            lower=DerivativeMap(
                velocity=velocity_lower_bound,
                acceleration=acceleration_lower_bounds,
                jerk=jerk_lower_bounds,
            ),
            upper=DerivativeMap(
                velocity=velocity_upper_bound,
                acceleration=acceleration_upper_bounds,
                jerk=jerk_upper_bounds,
            ),
        )

    def find_best_jerk_limit(
        self,
        prediction_horizon: int,
        time_step: float,
        target_velocity_limit: float,
        solver_class: type[QPSolver],
        almost_equal_threshold: float = 0.0001,
    ) -> float:
        """
        Finds the jerk limit under which the controller just barely reaches the target velocity
        within the prediction horizon.
        :param target_velocity_limit: The velocity limit that should be reachable.
        :return: The jerk limit that achieves target_velocity_limit.
        """
        jerk_limit = (4 * target_velocity_limit) / time_step**2
        upper_bound = jerk_limit
        lower_bound = 0
        best_vel_limit = 0
        best_jerk_limit = 0
        for i in range(100):
            vel_limit = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=prediction_horizon,
                vel_limit=1000,
                acc_limit=np.inf,
                jerk_limit=jerk_limit,
                dt=time_step,
                solver_class=solver_class,
            )[0]
            if abs(vel_limit - target_velocity_limit) < abs(
                best_vel_limit - target_velocity_limit
            ):
                best_vel_limit = vel_limit
                best_jerk_limit = jerk_limit
            if abs(vel_limit - target_velocity_limit) < almost_equal_threshold:
                break
            if vel_limit > target_velocity_limit:
                upper_bound = jerk_limit
                jerk_limit = round((jerk_limit + lower_bound) / 2, 4)
            else:
                lower_bound = jerk_limit
                jerk_limit = round((jerk_limit + upper_bound) / 2, 4)
        logger.debug(
            f"best velocity limit: {best_vel_limit} "
            f"(target = {target_velocity_limit}) with jerk limit: {best_jerk_limit} after {i + 1} iterations"
        )
        return best_jerk_limit

    def _resolve_limits(
        self, degree_of_freedom: DegreeOfFreedom
    ) -> tuple[DerivativeMap[float], DerivativeMap[float]]:
        """
        Builds the lower and upper limit maps for a degree of freedom, filling in unbounded
        acceleration limits and computing a jerk limit when none is given.
        """
        qp_controller_config = self.qp_controller_config
        lower_limits = DerivativeMap()
        upper_limits = DerivativeMap()

        if degree_of_freedom.has_position_limits():
            lower_limits.position = degree_of_freedom.limits.lower.position
            upper_limits.position = degree_of_freedom.limits.upper.position
        else:
            lower_limits.position = upper_limits.position = None

        lower_limits.velocity = degree_of_freedom.limits.lower.velocity
        upper_limits.velocity = degree_of_freedom.limits.upper.velocity

        if degree_of_freedom.limits.lower.acceleration is None:
            lower_limits.acceleration = -np.inf
        else:
            lower_limits.acceleration = degree_of_freedom.limits.lower.acceleration
        if degree_of_freedom.limits.upper.acceleration is None:
            upper_limits.acceleration = np.inf
        else:
            upper_limits.acceleration = degree_of_freedom.limits.upper.acceleration

        if degree_of_freedom.limits.upper.jerk is None:
            upper_limits.jerk = self.find_best_jerk_limit(
                qp_controller_config.prediction_horizon,
                qp_controller_config.mpc_dt,
                upper_limits.velocity,
                solver_class=qp_controller_config.qp_solver_class,
            )
            lower_limits.jerk = -upper_limits.jerk
        else:
            upper_limits.jerk = degree_of_freedom.limits.upper.jerk
            lower_limits.jerk = degree_of_freedom.limits.lower.jerk

        return lower_limits, upper_limits

    def compute(
        self,
        degree_of_freedom: DegreeOfFreedom,
    ) -> DegreeOfFreedomLimits[sm.Vector]:
        """
        Computes the horizon bounds for a single degree of freedom, filling in missing
        acceleration and jerk limits and raising when its velocity limit is unreachable.
        """
        qp_controller_config = self.qp_controller_config
        lower_limits, upper_limits = self._resolve_limits(degree_of_freedom)
        try:
            return self.compute_horizon_bounds(
                dof_symbols=degree_of_freedom.variables,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                solver_class=qp_controller_config.qp_solver_class,
                time_step=qp_controller_config.mpc_dt,
                prediction_horizon=qp_controller_config.prediction_horizon,
            )
        except InfeasibleException:
            self._raise_velocity_limit_diagnostic(degree_of_freedom, upper_limits)

    def _raise_velocity_limit_diagnostic(
        self, degree_of_freedom: DegreeOfFreedom, upper_limits: DerivativeMap[float]
    ):
        """
        Re-raises the active infeasibility, replacing it with a
        :class:`VelocityLimitUnreachableException` when the velocity limit cannot be reached
        within the prediction horizon under the jerk limit.
        """
        qp_controller_config = self.qp_controller_config
        max_reachable_vel = max_velocity_from_horizon_and_jerk_qp(
            prediction_horizon=qp_controller_config.prediction_horizon,
            vel_limit=100,
            acc_limit=upper_limits.acceleration,
            jerk_limit=upper_limits.jerk,
            dt=qp_controller_config.mpc_dt,
            solver_class=qp_controller_config.qp_solver_class,
        )[0]
        if max_reachable_vel >= upper_limits.velocity:
            raise
        exception = VelocityLimitUnreachableException(
            degree_of_freedom_name=degree_of_freedom.name,
            velocity_limit=upper_limits.velocity,
            prediction_horizon=qp_controller_config.prediction_horizon,
            jerk_limit=upper_limits.jerk,
            mpc_dt=qp_controller_config.mpc_dt,
            max_reachable_velocity=max_reachable_vel,
        )
        logger.error(str(exception))
        raise exception


@dataclass
class QuadraticProgramDegreeOfFreedomLimits:
    """
    Builds a :class:`DirectLimits` holding the bounds and weights of the robot's free variables
    (velocity and jerk decision variables across the prediction horizon).
    """

    @classmethod
    def create(
        cls,
        degrees_of_freedom: list[DegreeOfFreedom],
        qp_controller_config: QPControllerConfig,
    ) -> DirectLimits:
        """
        Builds the :class:`DirectLimits` for the given degrees of freedom and configuration.
        """
        self = cls()
        lower_bounds, upper_bounds = self.free_variable_bounds(
            degrees_of_freedom, qp_controller_config
        )
        quadratic_weights, linear_weights = self.init_weights(
            degrees_of_freedom, qp_controller_config
        )
        return DirectLimits(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            quadratic_weights=quadratic_weights,
            linear_weights=linear_weights,
            names=self.make_names(degrees_of_freedom, qp_controller_config),
        )

    def active_slots(
        self,
        degrees_of_freedom: list[DegreeOfFreedom],
        qp_controller_config: QPControllerConfig,
    ):
        """
        Yields every active decision variable slot as a ``(derivative, time_step, dof)`` tuple.
        The order defines the layout shared by bounds, weights, and names so they stay aligned.
        """
        max_derivative = qp_controller_config.max_derivative
        for derivative, time_step in product(
            [Derivatives.velocity, Derivatives.jerk],
            range(qp_controller_config.prediction_horizon),
        ):
            if time_step >= qp_controller_config.prediction_horizon - (
                max_derivative - derivative
            ):
                continue
            for degree_of_freedom in degrees_of_freedom:
                yield derivative, time_step, degree_of_freedom

    def make_names(
        self,
        degrees_of_freedom: list[DegreeOfFreedom],
        qp_controller_config: QPControllerConfig,
    ) -> list[str]:
        """
        Creates a debug name for every free variable slot.
        """
        short_label = {Derivatives.velocity: "vel", Derivatives.jerk: "jerk"}
        return [
            f"{dof.name}_{short_label[derivative]}_k_{time_step}"
            for derivative, time_step, dof in self.active_slots(
                degrees_of_freedom, qp_controller_config
            )
        ]

    def free_variable_bounds(
        self,
        degrees_of_freedom: list[DegreeOfFreedom],
        qp_controller_config: QPControllerConfig,
    ) -> tuple[sm.Vector, sm.Vector]:
        """
        Computes the lower and upper box limits of every free variable slot.
        """
        lower_bounds = []
        upper_bounds = []
        profiler = DegreeOfFreedomLimitProfiler(qp_controller_config)
        cache: dict[UUID, DegreeOfFreedomLimits[sm.Vector]] = {}
        for degree_of_freedom in degrees_of_freedom:
            cache[degree_of_freedom.id] = profiler.compute(
                degree_of_freedom=degree_of_freedom,
            )
        for derivative, t, degree_of_freedom in self.active_slots(
            degrees_of_freedom, qp_controller_config
        ):
            lower_bounds.append(cache[degree_of_freedom.id].lower[derivative][t])
            upper_bounds.append(cache[degree_of_freedom.id].upper[derivative][t])

        return sm.Vector(lower_bounds), sm.Vector(upper_bounds)

    def init_weights(
        self,
        degrees_of_freedom: list[DegreeOfFreedom],
        qp_controller_config: QPControllerConfig,
    ) -> tuple[sm.Vector, sm.Vector]:
        """
        Computes the quadratic and linear objective weights of every free variable slot.
        """
        quadratic_weights = []
        for derivative, t, degree_of_freedom in self.active_slots(
            degrees_of_freedom, qp_controller_config
        ):
            normalized_weight = self.normalize_dof_weight(
                variable_limit=degree_of_freedom.limits.upper[derivative],
                base_weight=qp_controller_config.get_dof_weight(
                    degree_of_freedom.name, derivative
                ),
                horizon_index=t,
                total_horizon_length=qp_controller_config.prediction_horizon - 3,
                growth_factor=qp_controller_config.horizon_weight_gain_scalar,
            )
            quadratic_weights.append(normalized_weight)
        return sm.Vector(quadratic_weights), sm.Vector.zeros(len(quadratic_weights))

    def normalize_dof_weight(
        self,
        variable_limit: float | None,
        base_weight: float,
        horizon_index: int,
        total_horizon_length: int,
        growth_factor: float,
    ) -> sm.Scalar:
        """
        Scales a free variable weight by its limit so derivatives become comparable, and ramps it
        over the horizon so later time steps are penalized more.
        """

        def linear(
            horizon_index: float,
            weight: float,
            total_horizon_length: int,
            growth_factor: float,
        ) -> float:
            start = weight * growth_factor
            slope = (weight - start) / total_horizon_length
            return slope * horizon_index + start

        if variable_limit is None:
            return 0.0
        weight = linear(horizon_index, base_weight, total_horizon_length, growth_factor)

        return weight * (1 / variable_limit) ** 2
