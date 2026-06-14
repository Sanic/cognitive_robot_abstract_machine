from dataclasses import dataclass, field

from krrood.entity_query_language.factories import ConditionType, evaluate_condition
from pycram.exceptions import ConditionNotSatisfied
from pycram.plans.plan_node import PlanNode


@dataclass
class ConditionNode(PlanNode):
    """
    Node representing a pre or post condition of an action
    """

    condition: ConditionType = field(kw_only=True)
    """
    The EQL condition to be evaluated
    """

    pre_condition: bool = field(kw_only=True)
    """
    If this is a pre or post condition
    """

    def notify(self):
        pass
