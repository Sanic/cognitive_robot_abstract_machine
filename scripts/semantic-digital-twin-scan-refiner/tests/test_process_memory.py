"""
Worker memory ceilings cannot exceed an inherited operating-system hard limit.
"""

from __future__ import annotations

# %% Imports
import resource
from unittest.mock import Mock
import pytest
from scan_refiner.process_memory import ProcessMemory
from scan_refiner.job_worker import JobWorker
from scan_refiner.project import RefinementProject
from source_task import SourceTask
from pathlib import Path


# %% Inherited process limits
@pytest.mark.parametrize(
    "hard_limit", [resource.RLIM_INFINITY, 8 * 1024**3, 32 * 1024**3]
)
def test_memory_limit_respects_the_inherited_hard_ceiling(
    monkeypatch: pytest.MonkeyPatch, hard_limit: int
) -> None:
    """
    A capped test launcher must not cause PermissionError in its worker children.
    """
    memory = ProcessMemory(memory_limit_bytes=16 * 1024**3)
    setter = Mock()
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (4 * 1024**3, hard_limit))
    monkeypatch.setattr(resource, "setrlimit", setter)
    memory.limit_address_space()
    expected = (
        memory.memory_limit_bytes
        if hard_limit == resource.RLIM_INFINITY
        else min(memory.memory_limit_bytes, hard_limit)
    )
    assert setter.call_args.args == (resource.RLIMIT_AS, (expected, expected))


# %% Shared default
def test_analysis_and_source_workers_share_the_memory_default(
    small_project: RefinementProject, tmp_path: Path
) -> None:
    """
    Source validation and analysis use the same configured process ceiling.
    """
    assert (
        JobWorker(project=small_project, directory=tmp_path).maximum_memory
        == ProcessMemory().memory_limit_bytes
    )
    assert SourceTask().maximum_memory == ProcessMemory().memory_limit_bytes
