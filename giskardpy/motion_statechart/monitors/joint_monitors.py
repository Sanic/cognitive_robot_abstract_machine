from dataclasses import field, dataclass
from typing import Dict

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection,
    ActiveConnection1DOF,
)


@dataclass
class JointGoalReached(MotionStatechartNode):
    goal_state: Dict[ActiveConnection1DOF, float] = field(kw_only=True)
    threshold: float = 0.01

    def __post_init__(self):
        comparison_list = []
        for connection, goal in self.goal_state.items():
            current = connection.dof.variables.position
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current
            comparison_list.append(cas.abs(error) < self.threshold)
        expression = cas.logic_all(cas.Expression(comparison_list))
        self.observation_expression = expression


@dataclass
class JointPositionAbove(MotionStatechartNode):
    connection: ActiveConnection = field(kw_only=True)
    threshold: float = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.connection, ActiveConnection1DOF):
            raise GoalInitalizationException(
                f"Connection {self.connection} must be of type ActiveConnection1DOF"
            )
        if (
            isinstance(self.connection, RevoluteConnection)
            and self.connection.dof.has_position_limits()
        ):
            raise GoalInitalizationException(
                f"{self.__class__.__name__} does not support joints of type continuous."
            )

        current = self.connection.dof.variables.position
        expression = current > self.threshold
        self.observation_expression = expression
