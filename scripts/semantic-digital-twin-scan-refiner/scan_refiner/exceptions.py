"""
Failures that leave scan evidence and project baselines unchanged.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException


# %% Input failures
@dataclass(kw_only=True)
class InvalidEvidence(DataclassException):
    """
    An input cannot support the requested interpretation or operation.
    """

    reason: str
    """
    Specific condition preventing the operation.
    """

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return "Check the input, coordinate units and operation settings."


@dataclass(kw_only=True)
class ExistingProject(DataclassException):
    """
    An initialization target already contains data.
    """

    directory: Path
    """
    Target that initialization must not replace.
    """

    def error_message(self) -> str:
        return f"Project directory already exists: {self.directory}"

    def suggest_correction(self) -> str:
        return "Open the existing project or choose a new directory."
