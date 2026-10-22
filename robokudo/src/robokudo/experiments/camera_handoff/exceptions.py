# %% imports

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% experiment errors


@dataclass
class InvalidExperiment(DataclassException):
    """
    An experiment cannot produce meaningful camera observations.
    """

    reason: str
    """The configuration or scene condition preventing execution."""

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return "Use a populated scene, positive camera distances, and a visible target."


@dataclass
class TargetNotAcquired(DataclassException):
    """
    The head image does not contain one complete target with valid depth.
    """

    def error_message(self) -> str:
        return "The head camera could not acquire a unique target."

    def suggest_correction(self) -> str:
        return (
            "Check head target selection, appearance features, visibility, and depth."
        )


@dataclass
class ClusterPoseMissing(DataclassException):
    """
    A selected object cluster lacks a unique pose and bounding-box annotation.
    """

    hypothesis_id: str
    """Identifier of the cluster without a usable pose."""

    def error_message(self) -> str:
        return f"Cluster {self.hypothesis_id} has no unique pose and bounding box."

    def suggest_correction(self) -> str:
        return "Run ClusterPoseBBAnnotator after tabletop clustering."
