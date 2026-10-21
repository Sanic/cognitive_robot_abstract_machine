# %% imports

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from itertools import product

import cv2
import numpy as np

from robokudo.experiments.camera_handoff.exceptions import TargetNotAcquired

# %% image geometry


@dataclass(frozen=True)
class ImageRegion:
    """A pixel rectangle with exclusive right and bottom boundaries."""

    left: int
    """First included image column."""
    top: int
    """First included image row."""
    right: int
    """First excluded image column."""
    bottom: int
    """First excluded image row."""


@dataclass
class CameraObservation:
    """A monocular observation with an optical-frame camera pose."""

    color: np.ndarray
    """Unmodified BGR sensor image."""
    intrinsic_matrix: np.ndarray
    """Pinhole calibration in pixel coordinates."""
    world_T_camera: np.ndarray
    """Transform from the optical frame to the scene frame."""

    def project(self, world_points: np.ndarray) -> np.ndarray:
        """Project world points into pixels; callers must reject nonpositive depth."""
        camera_T_world = np.linalg.inv(self.world_T_camera)
        rotation_vector, _ = cv2.Rodrigues(camera_T_world[:3, :3])
        pixels, _ = cv2.projectPoints(
            np.asarray(world_points, dtype=float),
            rotation_vector,
            camera_T_world[:3, 3],
            self.intrinsic_matrix,
            None,
        )
        return pixels.reshape(-1, 2)

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Recover world points from pixels and projective optical depths."""
        normalized = cv2.undistortPoints(
            np.asarray(pixels, dtype=float).reshape(-1, 1, 2),
            self.intrinsic_matrix,
            None,
        ).reshape(-1, 2)
        camera_points = np.column_stack(
            (normalized, np.ones(len(normalized)))
        ) * np.asarray(depth).reshape(-1, 1)
        return (
            camera_points @ self.world_T_camera[:3, :3].T + self.world_T_camera[:3, 3]
        )


@dataclass
class TargetEstimate:
    """An axis-aligned object estimate in the scene frame."""

    world_center: np.ndarray
    """Estimated object center, independent of the scene ground truth."""
    edge_length: float
    """Known cube edge length when individual dimensions are absent."""
    dimensions: np.ndarray | None = None
    """Individual bounding-box dimensions for a non-cubic target."""

    def front_depth(self, observation: CameraObservation) -> float:
        """Return the optical depth of the nearest bounding-box plane."""
        dimensions = (
            np.full(3, self.edge_length) if self.dimensions is None else self.dimensions
        )
        camera_T_world = np.linalg.inv(observation.world_T_camera)
        camera_center = camera_T_world @ np.append(self.world_center, 1)
        return float(camera_center[2] - np.abs(camera_T_world[2, :3]) @ dimensions / 2)

    def project_region(
        self,
        observation: CameraObservation,
        margin_pixels: int,
        uncertainty_meters: float = 0,
    ) -> ImageRegion | None:
        """Enclose the projected model, rejecting bounds crossing the camera plane."""
        dimensions = (
            np.full(3, self.edge_length) if self.dimensions is None else self.dimensions
        )
        corners = self.world_center + np.array(list(product((-0.5, 0.5), repeat=3))) * (
            dimensions + 2 * uncertainty_meters
        )
        camera_T_world = np.linalg.inv(observation.world_T_camera)
        camera_corners = corners @ camera_T_world[:3, :3].T + camera_T_world[:3, 3]
        if np.any(camera_corners[:, 2] <= 0):
            return None
        pixels = observation.project(corners)
        height, width = observation.color.shape[:2]
        minimum = np.floor(pixels.min(axis=0)).astype(int) - margin_pixels
        maximum = np.ceil(pixels.max(axis=0)).astype(int) + margin_pixels + 1
        left, top = np.maximum(minimum, 0)
        right, bottom = np.minimum(maximum, [width, height])
        if right <= left or bottom <= top:
            return None
        return ImageRegion(int(left), int(top), int(right), int(bottom))


# %% color acquisition and tracking


@dataclass
class TargetMeasurement:
    """A complete connected target patch in an image."""

    center: np.ndarray
    """Patch centroid in full-image pixel coordinates."""
    region: ImageRegion
    """Bounding rectangle of the observed patch."""
    mask: np.ndarray
    """Full-image mask containing only the selected component."""


@dataclass
class ImageTargetTracker(ABC):
    """Associate target observations inside a projected search region."""

    @abstractmethod
    def track(
        self, color: np.ndarray, region: ImageRegion | None
    ) -> TargetMeasurement | None:
        """Return an associated target point, or reject unreliable observations."""
        ...


@dataclass
class ColorTargetTracker(ImageTargetTracker):
    """Associate a unique complete color patch inside a geometric search region."""

    target_color: np.ndarray = field(
        default_factory=lambda: np.array([51, 51, 212], dtype=np.uint8)
    )
    """Expected BGR appearance, refined from the head observation."""
    color_tolerance: int = 35
    """Allowed deviation per color channel."""
    minimum_area: int = 9
    """Minimum connected-component area in pixels."""

    def track(
        self, color: np.ndarray, region: ImageRegion | None
    ) -> TargetMeasurement | None:
        """Reject missing, ambiguous, or boundary-clipped target observations."""
        if region is None:
            return None
        crop = color[region.top : region.bottom, region.left : region.right]
        lower = np.clip(
            self.target_color.astype(int) - self.color_tolerance, 0, 255
        ).astype(np.uint8)
        upper = np.clip(
            self.target_color.astype(int) + self.color_tolerance, 0, 255
        ).astype(np.uint8)
        mask = cv2.inRange(crop, lower, upper)
        count, labels, statistics, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = [
            index
            for index in range(1, count)
            if statistics[index, cv2.CC_STAT_AREA] >= self.minimum_area
        ]
        if len(candidates) != 1:
            return None
        component = candidates[0]
        left, top, width, height, _ = statistics[component]
        if (
            left == 0
            or top == 0
            or left + width == crop.shape[1]
            or top + height == crop.shape[0]
        ):
            return None
        full_mask = np.zeros(color.shape[:2], dtype=bool)
        full_mask[region.top : region.bottom, region.left : region.right] = (
            labels == component
        )
        return TargetMeasurement(
            centroids[component] + [region.left, region.top],
            ImageRegion(
                int(left + region.left),
                int(top + region.top),
                int(left + width + region.left),
                int(top + height + region.top),
            ),
            full_mask,
        )

    def acquire(
        self, observation: CameraObservation, depth: np.ndarray, edge_length: float
    ) -> TargetEstimate:
        """Estimate a cube center from a head view perpendicular to its top face.

        Depth is measured in meters. The cube face must be parallel to the image plane.
        """
        height, width = observation.color.shape[:2]
        measurement = self.track(observation.color, ImageRegion(0, 0, width, height))
        if measurement is None:
            raise TargetNotAcquired()
        valid = measurement.mask & np.isfinite(depth) & (depth > 0)
        if np.count_nonzero(valid) < self.minimum_area:
            raise TargetNotAcquired()
        rows, columns = np.nonzero(valid)
        world_surface = observation.unproject(
            np.column_stack((columns, rows)), depth[valid]
        )
        world_center = (
            np.median(world_surface, axis=0)
            + observation.world_T_camera[:3, 2] * edge_length / 2
        )
        self.target_color = np.median(observation.color[valid], axis=0).astype(np.uint8)
        return TargetEstimate(world_center, edge_length)
