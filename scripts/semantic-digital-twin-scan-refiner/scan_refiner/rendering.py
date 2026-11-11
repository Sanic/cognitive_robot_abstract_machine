"""
Compact browser buffers with a reversible display-only coordinate origin.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np


# %% Display coordinates
@dataclass
class DisplayCoordinates:
    """
    Local float32 display positions paired with their exact scan-frame origin.
    """

    points: np.ndarray
    """Compact render positions, never substituted for source evidence."""
    origin: np.ndarray
    """
    Float64 scan position added to local positions to recover scan coordinates.
    """

    @classmethod
    def from_points(cls, points: np.ndarray) -> DisplayCoordinates:
        """
        Choose a nearby origin before reducing the precision of browser buffers.
        """
        origin = np.median(points, axis=0)
        return cls((points - origin).astype(np.float32), origin)
