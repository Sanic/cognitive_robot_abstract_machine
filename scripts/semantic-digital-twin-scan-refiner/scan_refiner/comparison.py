"""
Metric surface-distance comparison independent of browser rendering.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from enum import StrEnum
from dataclasses import dataclass
import numpy as np
import trimesh


# %% Computation progress
class ComparisonPhase(StrEnum):
    """
    Directions whose work can be reported independently of presentation.
    """

    SCAN_RESIDUALS = "Computing labelled scan-to-triangle distances"
    MODEL_SUPPORT = "Sampling model surfaces and checking nearby scan evidence"


@dataclass(kw_only=True)
class ComparisonProgress(ABC):
    """
    Receive phase progress without coupling numeric queries to job persistence.
    """

    @abstractmethod
    def advance(self, *, phase: ComparisonPhase, completed: int, total: int) -> None:
        """
        Publish measured progress; zero total denotes an indeterminate phase.
        """


# %% Nearest surfaces
@dataclass(kw_only=True)
class SurfaceDistances:
    """
    Nearest-triangle query results in original point order.
    """

    distances: np.ndarray
    """
    Unsigned point-to-surface separations in metres.
    """

    closest_points: np.ndarray
    """
    Closest surface positions expressed in the supplied mesh root frame.
    """

    triangle_indices: np.ndarray
    """
    Indices into the exact mesh supplied to the query.
    """


# %% Distance computation
@dataclass(kw_only=True)
class SurfaceComparison:
    """
    Unsigned point-to-triangle distances in a declared common frame.
    """

    batch_size: int = 2000
    """
    Maximum query points passed to the mesh proximity accelerator at once.
    """

    def scan_to_model(
        self,
        *,
        scan_points: np.ndarray,
        root_mesh: trimesh.Trimesh,
        scan_T_root: np.ndarray,
    ) -> np.ndarray:
        """
        Return metres to the selected surface without interpreting missing coverage.
        """
        return self.nearest(
            scan_points=scan_points, root_mesh=root_mesh, scan_T_root=scan_T_root
        ).distances

    def nearest(
        self,
        *,
        scan_points: np.ndarray,
        root_mesh: trimesh.Trimesh,
        scan_T_root: np.ndarray,
        progress: ComparisonProgress | None = None,
    ) -> SurfaceDistances:
        """
        Return distances and triangle ownership without changing source order.
        """
        root_points = trimesh.transform_points(scan_points, np.linalg.inv(scan_T_root))
        distances = np.empty(len(root_points))
        closest_points = np.empty((len(root_points), 3))
        triangle_indices = np.empty(len(root_points), dtype=np.int64)
        for start in range(0, len(root_points), self.batch_size):
            stop = start + self.batch_size
            (
                closest_points[start:stop],
                distances[start:stop],
                triangle_indices[start:stop],
            ) = trimesh.proximity.closest_point(root_mesh, root_points[start:stop])
            if progress is not None:
                progress.advance(
                    phase=ComparisonPhase.SCAN_RESIDUALS,
                    completed=min(stop, len(root_points)),
                    total=len(root_points),
                )
        return SurfaceDistances(
            distances=distances,
            closest_points=closest_points,
            triangle_indices=triangle_indices,
        )
