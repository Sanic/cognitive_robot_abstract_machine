"""
Errors in image-region projection.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% image masks


@dataclass
class ProjectionMaskShapeMismatch(DataclassException):
    """
    A projection mask does not cover its declared image region.
    """

    actual_shape: tuple[int, ...]
    """
    Dimensions of the supplied mask.
    """

    expected_shape: tuple[int, ...]
    """
    Dimensions of the selected image region.
    """

    def error_message(self) -> str:
        return f"Projection mask has shape {self.actual_shape}; expected {self.expected_shape}."

    def suggest_correction(self) -> str:
        return "Supply a mask matching the image or cropped region in the selected coordinates."
