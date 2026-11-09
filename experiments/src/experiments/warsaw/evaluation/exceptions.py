"""
What goes wrong when a run's results are gathered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% a run that cannot be reported


@dataclass
class RunNotFinishedError(DataclassException):
    """
    Raised when the results of a run that stopped before its end are asked for.
    """

    run_directory: Path
    """
    The directory of the run.
    """

    def error_message(self) -> str:
        return f"'{self.run_directory}' did not finish, so it has no results to report."

    def suggest_correction(self) -> str:
        return (
            "Leave the run out of the comparison, and say in the text that it failed, "
            "or run it again."
        )


# %% a ground truth that cannot be built


@dataclass
class DraftAlreadyWrittenError(DataclassException, FileExistsError):
    """
    Raised when a draft would be written over a file that may hold a person's decisions.
    """

    path: Path
    """
    The file already there.
    """

    def error_message(self) -> str:
        return f"'{self.path}' already exists and may hold decisions made by hand."

    def suggest_correction(self) -> str:
        return "Move the file away or write the new draft somewhere else."


@dataclass
class UndecidedGroundTruthError(DataclassException, ValueError):
    """
    Raised when a ground truth is built from a draft that still leaves something open.
    """

    undecided: list[str]
    """
    What is still to be decided.
    """

    def error_message(self) -> str:
        return f"The draft leaves {len(self.undecided)} open: {self.undecided}."

    def suggest_correction(self) -> str:
        return "Decide each of them in the draft, and build the ground truth again."


@dataclass
class UnknownWholeError(DataclassException, LookupError):
    """
    Raised when a draft says an object belongs to one it does not list.
    """

    part: str
    """
    The object said to belong to it.
    """

    whole: str
    """
    The whole the draft names.
    """

    def error_message(self) -> str:
        return f"{self.part} is said to belong to {self.whole}, which is not listed."

    def suggest_correction(self) -> str:
        return "Name an object the draft lists, or say it has no whole."


@dataclass
class PartHierarchyCycleError(DataclassException, ValueError):
    """
    Raised when a draft's objects belong to one another in a circle.
    """

    objects: list[str]
    """
    The objects on the circle.
    """

    def error_message(self) -> str:
        return f"These objects belong to one another in a circle: {self.objects}."

    def suggest_correction(self) -> str:
        return "A part hierarchy is a tree, so one of them has to have another whole."


# %% spatial comparison inputs


@dataclass
class ComparisonGeometryMissingError(DataclassException, ValueError):
    """
    Raised when a geometric candidate is requested without measured geometry.
    """

    objects: list[str]
    """The objects without bounds or dimensions."""

    def error_message(self) -> str:
        return f"No comparison geometry for {self.objects}."

    def suggest_correction(self) -> str:
        return "Leave objects without geometry unmatched or measure their parts first."
