"""Finite typed operations and schemas shared by HTTP, CLI and MCP transports."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from inspect import getdoc
from typing_extensions import get_type_hints
from typing_extensions import Callable
from scan_refiner.agent.models import AgentOperation
from scan_refiner.agent.interfaces import AgentOperations


# %% Tool contracts
@dataclass(kw_only=True)
class AgentBinding:
    """One explicit operation's request, response and typed implementation."""

    operation: AgentOperation
    """Stable public tool name."""
    request_type: type
    """Only this request DTO may select operation arguments."""
    response_type: type
    """Concrete response contract for transport deserialization."""
    handler: Callable[..., object]
    """Bound implementation with a keyword-only request parameter."""
    description: str
    """Inherited operation contract exposed to the agent host."""

    @classmethod
    def from_handler(
        cls, *, operation: AgentOperation, handler: Callable[..., object]
    ) -> AgentBinding:
        """Discover DTO types once from the explicit typed operation contract."""
        annotations = get_type_hints(handler)
        return cls(
            operation=operation,
            request_type=annotations["request"],
            response_type=annotations["return"],
            handler=handler,
            description=getdoc(handler) or operation.value,
        )


@dataclass(kw_only=True)
class AgentRegistry:
    """Named operations rather than arbitrary method or filesystem access."""

    implementation: AgentOperations
    """Typed local service or HTTP client implementing the same operation contracts."""

    @property
    def bindings(self) -> list[AgentBinding]:
        """Bind the finite tool set using each method's actual request/response annotations."""
        service = self.implementation
        return [
            AgentBinding.from_handler(
                operation=AgentOperation.PROJECT, handler=service.inspect_project
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.BODIES, handler=service.find_bodies
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.REGION, handler=service.create_region
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.SAMPLE, handler=service.inspect_region
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.RENDER, handler=service.render_region
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.REGISTER, handler=service.propose_registration
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.REFINE, handler=service.propose_refinement
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.SUBMIT_CHANGE_SET,
                handler=service.submit_change_set,
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.LIST_CHANGE_SETS,
                handler=service.list_change_sets,
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.COMPUTE_CHANGE_SET_HEATMAP,
                handler=service.compute_change_set_heatmap,
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.DECIDE_CHANGE_SET,
                handler=service.decide_change_set,
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.HEATMAP, handler=service.propose_comparison
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.JOB, handler=service.inspect_job
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.CANCEL, handler=service.cancel_job
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.COMPARE, handler=service.compare_proposal
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.APPLY, handler=service.apply_proposal
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.ACCEPT, handler=service.accept_registration
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.UNDO, handler=service.undo_action
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.REDO, handler=service.redo_action
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.SOURCE_REVIEW, handler=service.review_source
            ),
            AgentBinding.from_handler(
                operation=AgentOperation.EXPORT, handler=service.export_source
            ),
        ]
