"""
Invoke typed agent operations on a running shared refiner without owning project files.
"""

from __future__ import annotations

# %% Imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
from pydantic import TypeAdapter
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.agent.models import AgentOperation
from scan_refiner.agent.client import AgentClient
from scan_refiner.agent.registry import AgentRegistry


# %% Typed operation command
@dataclass(kw_only=True)
class AgentOperationCommand:
    """
    Expose the same finite DTO contracts as the agent and local service.
    """

    def invoke_operation(self) -> None:
        """
        Validate a plain JSON input file and print the typed operation result.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("operation", choices=list(AgentOperation))
        parser.add_argument("--url", required=True)
        parser.add_argument("--request", type=Path)
        arguments = parser.parse_args()
        client = AgentClient(base_url=arguments.url)
        binding = next(
            entry
            for entry in AgentRegistry(implementation=client).bindings
            if entry.operation == arguments.operation
        )
        payload = arguments.request.read_text() if arguments.request else "{}"
        request = TypeAdapter(binding.request_type).validate_json(payload)
        result = binding.handler(request=request)
        print(json.dumps(DataclassJSONSerializer.to_json(instance=result), indent=2))


# %% Command entry point
if __name__ == "__main__":
    AgentOperationCommand().invoke_operation()
