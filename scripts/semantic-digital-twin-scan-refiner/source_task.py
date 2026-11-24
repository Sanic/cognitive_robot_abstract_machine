"""
Prepare or publish a validated source copy in a bounded standalone process.
"""

from __future__ import annotations

# %% Imports
import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import json
from scan_refiner.process_memory import ProcessMemory
from krrood.adapters.json_serializer import DataclassJSONSerializer
from semantic_digital_twin.utils import suppress_stdout_stderr
from scan_refiner.source_application import SourceApplication
from scan_refiner.project import RefinementProject


# %% Command contract
class SourceOperation(StrEnum):
    """
    Explicit review and publication operations.
    """

    REVIEW = "review"
    PUBLISH = "publish"


@dataclass(kw_only=True)
class SourceTask:
    """
    Bounded CLI entry point shared by local HTTP actions and shell use.
    """

    maximum_memory: int = ProcessMemory().memory_limit_bytes
    """
    Address-space ceiling in bytes for source reloads and mesh conversion.
    """

    def run(self) -> None:
        """
        Print a typed review or completed result without altering project geometry.
        """
        ProcessMemory(memory_limit_bytes=self.maximum_memory).limit_address_space()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--project", type=Path, required=True)
        parser.add_argument("--action", choices=list(SourceOperation), required=True)
        parser.add_argument("--review", default="")
        arguments = parser.parse_args()
        service = SourceApplication(
            project=RefinementProject(directory=arguments.project)
        )
        with suppress_stdout_stderr():
            result = (
                service.prepare()
                if arguments.action == SourceOperation.REVIEW
                else service.publish(identifier=arguments.review)
            )
        print(json.dumps(DataclassJSONSerializer.to_json(instance=result)), flush=True)


# %% Command entry point
if __name__ == "__main__":
    SourceTask().run()
