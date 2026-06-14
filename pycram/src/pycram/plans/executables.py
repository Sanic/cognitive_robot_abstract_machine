from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from typing_extensions import List, Dict, ClassVar, TYPE_CHECKING

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, Task
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.ros_executor import Ros2Executor
from krrood.entity_query_language.factories import evaluate_condition
from pycram.datastructures.enums import ExecutionType
from pycram.exceptions import MotionDidNotFinish, ConditionNotSatisfied
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:

    from plans.condition_nodes import ConditionNode
    from plans.plan_node import MotionNode
    from datastructures.dataclasses import Context

logger = logging.getLogger(__name__)


@dataclass
class Executable:
    """
    Base class for executable units.
    """

    execution_list: List[Executable] = field(default_factory=list)

    context: Context = field(kw_only=True)

    def execute(self) -> None:
        """
        Executes the unit.
        """
        for e in self.execution_list:
            e.execute()


@dataclass
class GiskardExecutable(Executable):
    motion_mappings: Dict[MotionNode, Task] = field(kw_only=True)
    """
    Mapping from the motion nodes of the plan to their giskard tasks, in execution order.
    """

    execution_type: ClassVar[ExecutionType] = None
    """
    The execution type used for all giskard executables, managed by
    :py:class:`pycram.motion_executor.ExecutionEnvironment`.
    """

    @property
    def motion_state_chart(self) -> MotionStatechart:
        """
        Giskard's motion state chart constructed from the motions of this executable.
        """
        motion_state_chart = MotionStatechart()
        sequence_node = Sequence(nodes=list(self.motion_mappings.values()))
        motion_state_chart.add_node(sequence_node)

        motion_state_chart.add_node(EndMotion.when_true(sequence_node))
        return motion_state_chart

    @property
    def is_interrupted(self) -> bool:
        return any(node.is_interrupted for node in self.motion_mappings)

    @property
    def is_paused(self) -> bool:
        return any(node.is_paused for node in self.motion_mappings)

    def execute(self) -> None:
        """
        Builds the motion state chart from the motions and executes it according to the execution type.
        """
        if len(self.motion_mappings) == 0:
            return

        match GiskardExecutable.execution_type:
            case ExecutionType.SIMULATED:
                self._execute_simulation()
            case ExecutionType.REAL:
                self._execute_real()
            case ExecutionType.NO_EXECUTION:
                return
            case _:
                logger.error(
                    f"Unknown execution type: {GiskardExecutable.execution_type}"
                )

    def _execute_simulation(self) -> None:
        """
        Compiles the motion state chart and ticks it in the world of the context until it is done.
        """
        executor = Ros2Executor(
            context=MotionStatechartContext(
                world=self.context.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
            ros_node=self.context.ros_node,
        )
        motion_state_chart = self.motion_state_chart
        executor.compile(motion_state_chart)

        counter = 0
        while counter < len(self.motion_mappings) * 1500:
            if self.is_interrupted:
                return
            elif self.is_paused:
                time.sleep(0.01)
                continue

            executor.tick()
            counter += 1
            if executor.motion_statechart.is_end_motion():
                break

        executor._set_velocity_acceleration_jerk_to_zero()
        executor.motion_statechart.cleanup_nodes(context=executor.context)
        executor.context.cleanup()

        if not executor.motion_statechart.is_end_motion():
            failed_nodes = [
                node
                for node in motion_state_chart.nodes
                if node.life_cycle_state
                not in [LifeCycleValues.DONE, LifeCycleValues.NOT_STARTED]
            ]
            logger.error(f"Failed Nodes: {failed_nodes}")
            raise MotionDidNotFinish(failed_nodes)

    def _execute_real(self) -> None:
        """
        Executes the motion state chart on the real robot via giskard while monitoring for interrupts.
        """
        from giskardpy.middleware.ros2.python_interface import GiskardWrapper

        giskard = GiskardWrapper(self.context.ros_node)

        kill_event = threading.Event()
        interrupt_thread = threading.Thread(
            target=self._monitor_interrupt, args=(giskard, kill_event)
        )
        interrupt_thread.start()

        giskard.execute(self.motion_state_chart)

        kill_event.set()
        interrupt_thread.join()

    def _monitor_interrupt(self, giskard_wrapper, kill_event: threading.Event) -> None:
        while not kill_event.is_set():
            if self.is_paused:
                raise NotImplementedError("Pause not implemented for real execution")
            elif self.is_interrupted:
                giskard_wrapper.cancel_goal_async()
            time.sleep(0.01)


@dataclass
class LanguageExecutable(Executable):

    @property
    def motion_state_chart(self):
        return Parallel(
            nodes=[
                motion.motion_node.designator.motion_chart for motion in self.motions
            ]
        )


@dataclass
class ConditionExecutable(Executable):
    """
    An executable unit for a condition node.
    """

    condition_node: ConditionNode = field(kw_only=True)
    """
    The condition node to execute.
    """

    def execute(self) -> None:
        """
        Executes the condition node.
        """
        if evaluate_condition(self.condition_node.condition):
            return True
        raise ConditionNotSatisfied(
            pre_condition=self.condition_node.pre_condition,
            action=self.condition_node.__class__,
            condition=self.condition_node.condition,
        )


@dataclass
class MotionExecutable(GiskardExecutable): ...


@dataclass
class ModelChangeExecutable(Executable):

    body: Body = field(kw_only=True)

    new_parent: Body = field(kw_only=True)

    def execute(self) -> None:
        obj_transform = self.context.world.compute_forward_kinematics(
            self.new_parent, self.body
        )
        with self.context.world.modify_world():
            self.context.world.remove_connection(self.body.parent_connection)
            connection = Connection6DoF.create_with_dofs(
                parent=self.new_parent, child=self.body, world=self.context.world
            )
            self.context.world.add_connection(connection)
            connection.origin = obj_transform
