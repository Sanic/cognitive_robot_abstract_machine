"""
Errors in annotator placement within behavior trees.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from robokudo.exceptions import RoboKudoError


# %% Collection reader placement
@dataclass
class InvalidCollectionReaderParent(RoboKudoError):
    """
    A collection reader has no parent sequence to initialize.
    """

    def error_message(self) -> str:
        """
        Identify the invalid reader placement.
        """
        return "The collection reader's parent is not a behavior-tree sequence."

    def suggest_correction(self) -> str:
        """
        Describe valid placement of the reader.
        """
        return "Add the collection reader to a pipeline or sequence before ticking it."
