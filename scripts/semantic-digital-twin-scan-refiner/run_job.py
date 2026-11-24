"""
Execute one service-owned bounded refiner computation.
"""

from __future__ import annotations

# %% Imports
import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from scan_refiner.job_worker import JobWorker
from scan_refiner.jobs import JobManager
from scan_refiner.project import RefinementProject


# %% Worker arguments
class WorkerOption(StrEnum):
    """
    Arguments supplied by the owning local service.
    """

    PROJECT = "--project"
    JOB = "--job"


# %% Process entry
@dataclass(kw_only=True)
class WorkerCommandLine:
    """
    Resolve one persisted computation without importing the rendering application.
    """

    @classmethod
    def run(cls) -> None:
        """
        Execute the existing request inside its configured resource ceiling.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(WorkerOption.PROJECT, type=Path, required=True)
        parser.add_argument(WorkerOption.JOB, required=True)
        arguments = parser.parse_args()
        project = RefinementProject(directory=arguments.project)
        JobWorker(
            project=project,
            directory=JobManager(project=project).directory(identifier=arguments.job),
        ).run()


if __name__ == "__main__":
    WorkerCommandLine.run()
