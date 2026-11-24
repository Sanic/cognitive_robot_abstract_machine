"""
Automatic upright registration with separated hypotheses for human review.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.records import RegistrationCandidate


# %% Upright rigid placement
@dataclass(kw_only=True)
class UprightPlacement:
    """
    Four permitted placement parameters when both inputs already share gravity.
    """

    yaw: float
    """
    Rotation about the common vertical axis in radians.
    """

    translation: np.ndarray
    """
    Root origin expressed in scan coordinates in metres.
    """

    @property
    def scan_T_root(self) -> np.ndarray:
        """
        Homogeneous root-to-scan transform for column-vector coordinates.
        """
        cosine, sine = np.cos(self.yaw), np.sin(self.yaw)
        transform = np.eye(4)
        transform[:2, :2] = ((cosine, -sine), (sine, cosine))
        transform[:3, 3] = self.translation
        return transform

    def transform(self, *, points: np.ndarray) -> np.ndarray:
        """
        Express root-frame row-array points in scan coordinates.
        """
        return points @ self.scan_T_root[:3, :3].T + self.translation


@dataclass(kw_only=True)
class OccupancyGrid:
    """
    Horizontal occupancy and its scan-frame cell origin.
    """

    values: np.ndarray
    """
    Binary evidence, preventing dense regions from dominating correlation.
    """

    origin: np.ndarray
    """
    Lower occupied XY cell corner in metres.
    """


# %% Registration search
@dataclass(kw_only=True)
class RegistrationSearch:
    """
    Find yaw and translation candidates without requiring clicked landmarks.
    """

    cell_width: float = 0.12
    """
    Coarse XY occupancy resolution in metres.
    """

    yaw_step_degrees: float = 10.0
    """
    Angular interval across a complete rotation before local refinement.
    """

    candidate_count: int = 6
    """
    Number of separated initial hypotheses retained for refinement.
    """

    correspondence_distance: float = 0.45
    """
    Largest model-to-scan separation admitted in local fitting, in metres.
    """

    support_distance: float = 0.08
    """
    Distance defining supported model samples for candidate ranking.
    """

    refinement_iterations: int = 35
    """
    Maximum correspondence updates per hypothesis.
    """

    maximum_grid_cells: int = 1_000_000
    """
    Per-input occupancy allocation limit; crop or coarsen larger sites.
    """

    robust_distance: float = 0.03
    """
    Residual scale for robust local fitting in metres.
    """

    separation_degrees: float = 25.0
    """
    Angular spacing distinguishing initial hypotheses at nearby positions.
    """

    separation_distance: float = 0.5
    """
    XY spacing distinguishing spatial hypotheses with similar orientation.
    """

    convergence_tolerance: float = 0.00001
    """
    Stopping threshold for local parameter updates.
    """

    def search(
        self, *, model_points: np.ndarray, scan_points: np.ndarray
    ) -> list[RegistrationCandidate]:
        """
        Rank upright candidates; returned transforms remain unaccepted proposals.
        """
        for points in (model_points, scan_points):
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
                raise InvalidEvidence(
                    reason="Registration requires at least four XYZ points per input."
                )
            if not np.isfinite(points).all():
                raise InvalidEvidence(reason="Registration evidence must be finite.")
        if not 0 < self.yaw_step_degrees <= 180 or self.cell_width <= 0:
            raise InvalidEvidence(
                reason="Use a positive cell width and yaw interval up to 180 degrees."
            )
        if (
            min(
                self.candidate_count,
                self.refinement_iterations,
                self.maximum_grid_cells,
            )
            <= 0
        ):
            raise InvalidEvidence(reason="Registration budgets must be positive.")
        if (
            min(
                self.correspondence_distance,
                self.support_distance,
                self.robust_distance,
            )
            <= 0
        ):
            raise InvalidEvidence(
                reason="Registration distance thresholds must be positive."
            )
        initial = self.coarse_candidates(
            model_points=model_points, scan_points=scan_points
        )
        tree = cKDTree(scan_points)
        candidates = [
            self.refine(model_points=model_points, scan_tree=tree, placement=placement)
            for placement in initial
        ]
        return sorted(
            candidates, key=lambda item: (-item.support_fraction, item.median_distance)
        )

    def grid(self, *, points: np.ndarray) -> OccupancyGrid:
        """
        Construct a bounded binary grid in the input's own coordinate frame.
        """
        origin = np.floor(points[:, :2].min(axis=0) / self.cell_width) * self.cell_width
        cells = np.floor((points[:, :2] - origin) / self.cell_width).astype(np.int64)
        shape = cells.max(axis=0) + 1
        if np.prod(shape, dtype=np.float64) > self.maximum_grid_cells:
            raise InvalidEvidence(
                reason="Occupancy grid budget exceeded; crop the inputs or increase cell width."
            )
        values = np.zeros(shape, dtype=np.float64)
        values[cells[:, 0], cells[:, 1]] = 1
        return OccupancyGrid(values=values, origin=origin)

    def coarse_candidates(
        self, *, model_points: np.ndarray, scan_points: np.ndarray
    ) -> list[UprightPlacement]:
        """
        Search horizontal translations across the complete yaw range.
        """
        scan_grid = self.grid(points=scan_points)
        vertical_translation = np.quantile(scan_points[:, 2], 0.01) - np.quantile(
            model_points[:, 2], 0.01
        )
        ranked = []
        for degrees in np.arange(0, 360, self.yaw_step_degrees):
            placement = UprightPlacement(
                yaw=float(np.radians(degrees)), translation=np.zeros(3)
            )
            model_grid = self.grid(points=placement.transform(points=model_points))
            correlation = fftconvolve(
                scan_grid.values, model_grid.values[::-1, ::-1], mode="full"
            )
            peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape))
            translation = (
                scan_grid.origin
                + (peak - model_grid.values.shape + 1) * self.cell_width
                - model_grid.origin
            )
            placement.translation = np.r_[translation, vertical_translation]
            ranked.append(
                (float(correlation[tuple(peak)] / model_grid.values.sum()), placement)
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for _, placement in ranked:
            if any(
                abs(
                    np.arctan2(
                        np.sin(placement.yaw - other.yaw),
                        np.cos(placement.yaw - other.yaw),
                    )
                )
                < np.radians(self.separation_degrees)
                and np.linalg.norm(placement.translation[:2] - other.translation[:2])
                < self.separation_distance
                for other in selected
            ):
                continue
            selected.append(placement)
            if len(selected) == self.candidate_count:
                break
        return selected

    def refine(
        self,
        *,
        model_points: np.ndarray,
        scan_tree: cKDTree,
        placement: UprightPlacement,
    ) -> RegistrationCandidate:
        """
        Robustly refine one hypothesis while preserving upright axes and metric scale.
        """
        parameters = np.r_[placement.yaw, placement.translation]
        for _ in range(self.refinement_iterations):
            placed = UprightPlacement(
                yaw=parameters[0], translation=parameters[1:]
            ).transform(points=model_points)
            distances, indices = scan_tree.query(placed, workers=1)
            keep = distances < self.correspondence_distance
            if keep.sum() < 4:
                break
            source, target = model_points[keep], scan_tree.data[indices[keep]]
            result = least_squares(
                lambda values: (
                    UprightPlacement(yaw=values[0], translation=values[1:]).transform(
                        points=source
                    )
                    - target
                ).ravel(),
                parameters,
                loss="soft_l1",
                f_scale=self.robust_distance,
                max_nfev=20,
            )
            movement = np.linalg.norm(result.x - parameters)
            parameters = result.x
            if movement < self.convergence_tolerance:
                break
        transform = UprightPlacement(yaw=parameters[0], translation=parameters[1:])
        distances, _ = scan_tree.query(
            transform.transform(points=model_points), workers=1
        )
        return RegistrationCandidate(
            scan_T_root=transform.scan_T_root.tolist(),
            support_fraction=float(np.mean(distances < self.support_distance)),
            median_distance=float(np.median(distances)),
        )
