from dataclasses import field, dataclass
from typing import Dict, Tuple, Union, Optional

import numpy as np
from line_profiler import profile

from giskardpy.god_map import god_map
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import ObservationState


@dataclass
class CheckMaxTrajectoryLength(MotionStatechartNode):
    length: float

    def __post_init__(self):
        self.observation_expression = god_map.time_symbol > self.length


@dataclass(eq=False, repr=False)
class Print(MotionStatechartNode):
    message: str = ""

    def on_tick(self) -> Union[
        ObservationState.TrinaryFalse,
        ObservationState.TrinaryTrue,
        ObservationState.TrinaryUnknown,
    ]:
        print(self.message)
        return ObservationState.TrinaryTrue


@dataclass
class Sleep(MotionStatechartNode):
    seconds: float
    start_time: Optional[float] = field(default=None, init=False)

    def on_start(self) -> Optional[float]:
        self.start_time = None

    def on_tick(self) -> Optional[float]:
        if self.start_time is None:
            self.start_time = god_map.time
        return god_map.time - self.start_time >= self.seconds


@dataclass
class CollisionMatrixUpdater(MotionStatechartNode):
    new_collision_matrix: Dict[Tuple[str, str], float]

    @profile
    def __call__(self):
        god_map.collision_scene.set_collision_matrix(self.new_collision_matrix)
        god_map.collision_scene.reset_cache()
        self.state = ObservationState.true


@dataclass
class PayloadAlternator(MotionStatechartNode):
    mod: int = 2

    def __call__(self):
        self.state = np.floor(god_map.time) % self.mod == 0


@dataclass
class Counter(MotionStatechartNode):
    number: int
    counter: int = field(default=0, init=False)

    def __call__(self):
        if self.state == ObservationState.unknown:
            self.counter = 0
        if self.counter >= self.number:
            self.state = ObservationState.true
        else:
            self.state = ObservationState.false
        self.counter += 1


@dataclass
class Pulse(MotionStatechartNode):
    after_ticks: int
    true_for_ticks: int = 1
    ticks: int = field(default=0, init=False)

    def __call__(self):
        if self.state == ObservationState.unknown:
            self.counter = 0
        if self.after_ticks <= self.counter <= self.after_ticks + self.true_for_ticks:
            self.state = ObservationState.true
        else:
            self.state = ObservationState.false
        self.counter += 1
