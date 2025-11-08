from dataclasses import dataclass
from typing import Union

from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver


@dataclass
class SetPredictionHorizon(MotionStatechartNode):
    prediction_horizon: int

    def __post_init__(self):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        if self.prediction_horizon < 7:
            get_middleware().logwarn("Prediction horizon must be >= 7.")
        god_map.qp_controller.prediction_horizon = self.prediction_horizon

    def __call__(self):
        self.state = True


@dataclass
class SetQPSolver(MotionStatechartNode):
    qp_solver_id: Union[SupportedQPSolver, int]

    def __post_init__(self):
        qp_solver_id = SupportedQPSolver(self.qp_solver_id)
        god_map.qp_controller.set_qp_solver(qp_solver_id)

    def __call__(self):
        self.state = True
