"""
Queries about a robot's past execution behaviour, expressed in the KRROOD Entity Query Language.

The plan is taken directly from the bullet-world demo (coraplex/demos/coraplex_bullet_world_demo/demo.py):
a PR2 parks its arms, raises its torso, then transports three objects (milk, bowl, spoon) to a table.
After execution the plan graph is queried with EQL.

Each query is wrapped in a BehaviourQuery that pairs the natural-language question with the
EQL object so both can be inspected, logged, or evaluated together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import krrood.entity_query_language.factories as eql
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment, TaskStatus
from coraplex.datastructures.grasp import GraspDescription
from coraplex.motion_executor import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionNode, PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl, Spoon, Drawer, Handle,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection


# ---------------------------------------------------------------------------
# BehaviourQuery — bundles a natural-language question with its EQL object
# ---------------------------------------------------------------------------

@dataclass
class BehaviourQuery:
    """A natural-language question paired with the EQL query that answers it."""

    question: str
    query: Any  # krrood SymbolicExpression / Query / Quantifier

    def evaluate(self):
        """Evaluate the query and return results."""
        return self.query.evaluate()

    def __repr__(self) -> str:
        return f"BehaviourQuery({self.question!r})"


# ---------------------------------------------------------------------------
# World and plan setup — verbatim from the bullet-world demo
# ---------------------------------------------------------------------------

_resources = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "resources")

world = setup_world()

spoon = STLParser(os.path.join(_resources, "objects", "spoon.stl")).parse()
bowl = STLParser(os.path.join(_resources, "objects", "bowl.stl")).parse()

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(2.4, 2.2, 1, reference_frame=world.root),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"),
        child=spoon.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(-0.05, -0.05, 0),
    )
    world.merge_world(spoon, connection)

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=None)

with world.modify_world():
    WorldReasoner(world).reason()
    world.add_semantic_annotations([
        Bowl(root=world.get_body_by_name("bowl.stl")),
        Spoon(root=world.get_body_by_name("spoon.stl")),
    ])
    world.add_semantic_annotation_recursively(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )

context.evaluate_conditions = False

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        TransportAction(
            world.get_body_by_name("milk.stl"),
            Pose.from_xyz_rpy(4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
        TransportAction(
            world.get_body_by_name("bowl.stl"),
            Pose.from_xyz_rpy(5, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
        TransportAction(
            world.get_body_by_name("spoon.stl"),
            Pose.from_xyz_rpy(5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                pr2.left_arm.end_effector,
            ),
        ),
    ],
    context=context,
).plan

with simulated_robot:
    plan.perform()


# ---------------------------------------------------------------------------
# EQL variable — all nodes in the executed plan
# ---------------------------------------------------------------------------

def _nodes():
    return eql.variable(PlanNode, domain=plan.plan_graph.nodes())


def _action_nodes():
    return eql.variable(ActionNode, domain=plan.plan_graph.nodes())


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

queries: list[BehaviourQuery] = [

    BehaviourQuery(
        question="What did you just do?",
        query=eql.an(
            eql.entity(_nodes()).where(
                _nodes().is_leaf,
                _nodes().status == TaskStatus.SUCCEEDED,
            )
        ).ordered_by(_nodes().start_time),
    ),

    BehaviourQuery(
        question="Walk me through what you did in order.",
        query=eql.an(
            eql.entity(_nodes()).where(_nodes().status == TaskStatus.SUCCEEDED)
        ).ordered_by(_nodes().start_time),
    ),

    BehaviourQuery(
        question="How long did the whole task take?",
        query=eql.the(eql.entity(_nodes()).where(_nodes().parent == None)),  # noqa: E711
    ),

    BehaviourQuery(
        question="How long did each step take?",
        query=(
            eql.set_of(_nodes(), duration := _nodes().end_time - _nodes().start_time)
            .where(_nodes().end_time != None)  # noqa: E711
            .ordered_by(_nodes().start_time)
        ),
    ),

    BehaviourQuery(
        question="Did anything go wrong?",
        query=eql.an(eql.entity(_nodes()).where(_nodes().status == TaskStatus.FAILED)),
    ),

    BehaviourQuery(
        question="Why did you fail at that step?",
        query=eql.an(eql.entity(_nodes().reason).where(_nodes().status == TaskStatus.FAILED)),
    ),

    BehaviourQuery(
        question="How many times did you retry before giving up?",
        query=eql.count(_nodes()).where(_nodes().status == TaskStatus.FAILED),
    ),

    BehaviourQuery(
        question="Which fallback did you end up using?",
        query=eql.an(
            eql.entity(_nodes()).where(
                _nodes().status == TaskStatus.SUCCEEDED,
                eql.exists(
                    eql.variable(PlanNode, domain=_nodes().left_siblings),
                    lambda s: s.status == TaskStatus.FAILED,
                ),
            )
        ),
    ),

    BehaviourQuery(
        question="Were you ever interrupted? What caused it?",
        query=eql.an(eql.entity(_nodes()).where(_nodes().status == TaskStatus.INTERRUPTED)),
    ),

    BehaviourQuery(
        question="Was there a point where you were paused?",
        query=eql.an(eql.entity(_nodes()).where(_nodes().status == TaskStatus.PAUSE)),
    ),

    BehaviourQuery(
        question="Which step took the longest?",
        query=eql.max(
            _nodes(),
            key=lambda node: (node.end_time - node.start_time).total_seconds()
            if node.end_time is not None else 0.0,
        ),
    ),

    BehaviourQuery(
        question="Were all subtasks successful, or did some fail?",
        query=(
            eql.set_of(_nodes().status, c := eql.count(_nodes()))
            .grouped_by(_nodes().status)
            .ordered_by(c, descending=True)
        ),
    ),

    BehaviourQuery(
        question="What world modifications did you make?",
        query=eql.an(
            eql.entity(_action_nodes().execution_data.added_world_modifications)
            .where(
                _action_nodes().status == TaskStatus.SUCCEEDED,
                _action_nodes().execution_data != None,  # noqa: E711
            )
        ),
    ),

    BehaviourQuery(
        question="What was the state of the world when you started the task?",
        query=eql.the(
            eql.entity(_action_nodes().execution_data.execution_start_world_state)
            .where(
                _action_nodes().parent == None,  # noqa: E711
                _action_nodes().execution_data != None,  # noqa: E711
            )
        ),
    ),

    BehaviourQuery(
        question="What was the state of the world when you finished?",
        query=eql.the(
            eql.entity(_action_nodes().execution_data.execution_end_world_state)
            .where(
                _action_nodes().parent == None,  # noqa: E711
                _action_nodes().execution_data != None,  # noqa: E711
            )
        ),
    ),
]


# ---------------------------------------------------------------------------
# Demo: print each question and evaluate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for bq in queries:
        print(f"\n{'=' * 60}")
        print(f"  Q: {bq.question}")
        print(f"  {'─' * 56}")
        try:
            result = bq.evaluate()
            if hasattr(result, "__iter__"):
                items = list(result)
                if items:
                    for item in items:
                        print(f"    {item}")
                else:
                    print("    (no results)")
            else:
                print(f"    {result}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
