"""
Serve the human workspace and scoped agent operations with one authoritative worker
owner.
"""

from __future__ import annotations

# %% Imports
import argparse
from dataclasses import dataclass
from pathlib import Path
from scan_refiner.agent.models import AgentMode, AgentScope
from scan_refiner.project import RefinementProject
from scan_refiner.web_service import InspectionService


# %% Scoped local listener
@dataclass(kw_only=True)
class AgentWorkspaceCommand:
    """
    Configure agent authority once instead of granting permissions through tool
    arguments.
    """

    def serve_workspace(self) -> None:
        """
        Bind the existing web UI and typed tool API to one loopback listener.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--project", required=True, type=Path)
        parser.add_argument("--port", default=8765, type=int)
        parser.add_argument(
            "--mode", choices=list(AgentMode), default=AgentMode.INSPECT
        )
        parser.add_argument("--body", action="append", default=[])
        parser.add_argument("--maximum-translation-m", type=float, default=0.05)
        parser.add_argument("--maximum-extent-change-m", type=float, default=0.05)
        parser.add_argument("--allow-registration", action="store_true")
        parser.add_argument("--browser")
        arguments = parser.parse_args()
        scope = AgentScope(
            mode=arguments.mode,
            body_names=arguments.body,
            maximum_translation_m=arguments.maximum_translation_m,
            maximum_extent_change_m=arguments.maximum_extent_change_m,
            allow_registration=arguments.allow_registration,
        )
        service = InspectionService(
            project=RefinementProject(directory=arguments.project),
            agent_scope=scope,
            browser_executable=arguments.browser,
        )
        with service.create_server(port=arguments.port) as server:
            print(
                f"http://127.0.0.1:{server.server_port}/ — agent mode: {scope.mode}",
                flush=True,
            )
            server.serve_forever()


# %% Command entry point
if __name__ == "__main__":
    AgentWorkspaceCommand().serve_workspace()
