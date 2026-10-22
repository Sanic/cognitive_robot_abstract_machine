# %% imports

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np
from typing_extensions import TYPE_CHECKING

from robokudo.experiments.camera_handoff.exceptions import TargetNotAcquired
from robokudo.experiments.camera_handoff.perception import (
    CameraObservation,
    ImageRegion,
    ImageTargetTracker,
    TargetEstimate,
    TargetMeasurement,
)

if TYPE_CHECKING:
    from robokudo.experiments.camera_handoff.tabletop import SegmentedTarget

# %% hand observations


class HueEncoding(IntEnum):
    """
    Circular hue range in OpenCV's byte HSV images.
    """

    PERIOD = 180


@dataclass
class FeatureObservation:
    """
    A hand-image target measurement and its tracking landmarks.
    """

    measurement: TargetMeasurement
    """
    Associated target outline and alignment point.
    """

    pixels: np.ndarray
    """
    Hand-image landmarks supporting temporal tracking.
    """


@dataclass
class TrackingFrame:
    """
    An accepted hand observation retained for temporal correspondence.
    """

    grayscale: np.ndarray
    """
    Previous hand image in grayscale.
    """

    observation: FeatureObservation
    """
    Target and landmarks measured in that hand image.
    """


# %% initialization and hand-only tracking


@dataclass
class FeatureTargetTracker(ImageTargetTracker):
    """
    Initialize from head geometry and track an outline in hand images.
    """

    minimum_height: float = 0.015
    """
    Minimum foreground height above the known table plane in meters.
    """

    seed_pixel: np.ndarray | None = None
    """
    Head-image point inside the selected object; defaults to the principal point.
    """

    target_hue: int | None = None
    """
    Dominant object hue for initial hand-image foreground separation.
    """

    hue_tolerance: int = 12
    """
    Allowed circular hue deviation in OpenCV HSV units.
    """

    minimum_saturation: int = 60
    """
    Minimum saturation of foreground-colored pixels.
    """

    minimum_value: int = 20
    """
    Minimum brightness of foreground-colored pixels.
    """

    minimum_area: int = 20
    """
    Minimum target silhouette area in pixels.
    """

    closing_width: int = 9
    """
    Morphological closing width in pixels.
    """

    ambiguity_ratio: float = 0.5
    """
    Second-candidate area relative to the largest that rejects acquisition.
    """

    maximum_features: int = 200
    """
    Maximum number of hand-image corners retained for optical flow.
    """

    quality_level: float = 0.01
    """
    Minimum relative corner quality for hand-image landmarks.
    """

    minimum_feature_distance: float = 3.0
    """
    Minimum separation of tracking corners in pixels.
    """

    minimum_matches: int = 6
    """
    Minimum consistent hand-image landmarks for a temporal update.
    """

    reprojection_threshold: float = 3.0
    """
    Maximum temporal geometric residual in pixels.
    """

    flow_tolerance: float = 1.0
    """
    Maximum forward-backward optical-flow error in pixels.
    """

    outline_gain: float = 0.5
    """
    Contribution of the current silhouette center to temporal alignment.
    """

    previous: TrackingFrame | None = None
    """
    Most recent accepted hand observation.
    """

    inlier_count: int = 0
    """
    Number of hand-image landmarks supporting the latest update.
    """

    random_seed: int = 0
    """
    Seed for reproducible temporal geometric rejection.
    """

    def acquire(
        self,
        observation: CameraObservation,
        depth: np.ndarray,
        dimensions: np.ndarray,
        bias: np.ndarray,
        segmented_target: SegmentedTarget | None = None,
    ) -> TargetEstimate:
        """
        Initialize target geometry and coarse color from a selected head object.

        A segmented target supplies measured pose and extents. Without one, the upright
        model dimensions and table plane must be known.
        """
        if segmented_target is not None:
            self.learn_color(observation.color, segmented_target.mask)
            measured = segmented_target.estimate
            return TargetEstimate(
                measured.world_center + bias,
                measured.edge_length,
                measured.dimensions.copy(),
                measured.world_R_target.copy(),
            )
        rows, columns = np.nonzero(depth > 0)
        world_points = observation.unproject(
            np.column_stack((columns, rows)), depth[rows, columns]
        )
        foreground = np.zeros(depth.shape, dtype=np.uint8)
        foreground[rows, columns] = (world_points[:, 2] > self.minimum_height).astype(
            np.uint8
        )
        count, labels, _, _ = cv2.connectedComponentsWithStats(foreground)
        if count <= 1:
            raise TargetNotAcquired()
        seed = (
            observation.intrinsic_matrix[:2, 2]
            if self.seed_pixel is None
            else self.seed_pixel
        )
        seed_column, seed_row = np.rint(seed).astype(int)
        if not 0 <= seed_row < depth.shape[0] or not 0 <= seed_column < depth.shape[1]:
            raise TargetNotAcquired()
        component = labels[seed_row, seed_column]
        if component == 0:
            raise TargetNotAcquired()
        mask = labels == component
        rows, columns = np.nonzero(mask)
        surface = observation.unproject(
            np.column_stack((columns, rows)), depth[rows, columns]
        )
        lower, upper = surface.min(axis=0), surface.max(axis=0)
        center = (lower + upper) / 2
        center[1] = lower[1] + dimensions[1] / 2
        center[2] = upper[2] - dimensions[2] / 2
        self.learn_color(observation.color, mask)
        return TargetEstimate(center + bias, float(max(dimensions)), dimensions)

    def learn_color(self, color: np.ndarray, mask: np.ndarray) -> None:
        """
        Initialize hand foreground appearance from the selected object pixels.
        """
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        colored = (
            mask
            & (hsv[:, :, 1] >= self.minimum_saturation)
            & (hsv[:, :, 2] >= self.minimum_value)
        )
        if not np.any(colored):
            raise TargetNotAcquired()
        self.target_hue = int(
            np.argmax(np.bincount(hsv[:, :, 0][colored], minlength=HueEncoding.PERIOD))
        )
        self.previous = None

    def track(
        self, color: np.ndarray, region: ImageRegion | None
    ) -> TargetMeasurement | None:
        """
        Acquire the hand silhouette and track landmarks only between hand frames.
        """
        grayscale = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        temporal = self.follow_previous(grayscale, region)
        measurement = self.segment(color, region)
        if measurement is None and temporal is not None:
            self.previous = TrackingFrame(grayscale, temporal)
            self.inlier_count = len(temporal.pixels)
            return temporal.measurement
        if measurement is None:
            self.previous = None
            self.inlier_count = 0
            return None
        if temporal is not None:
            measurement.center = (
                self.outline_gain * measurement.center
                + (1 - self.outline_gain) * temporal.measurement.center
            )
        pixels = cv2.goodFeaturesToTrack(
            grayscale,
            maxCorners=self.maximum_features,
            qualityLevel=self.quality_level,
            minDistance=self.minimum_feature_distance,
            mask=measurement.mask.astype(np.uint8),
        )
        if pixels is None:
            self.previous = None
            self.inlier_count = 0
            return measurement
        pixels = pixels.reshape(-1, 2)
        self.inlier_count = len(pixels)
        self.previous = TrackingFrame(
            grayscale, FeatureObservation(measurement, pixels)
        )
        return measurement

    def segment(
        self, color: np.ndarray, region: ImageRegion | None
    ) -> TargetMeasurement | None:
        """
        Locate a unique complete colored silhouette inside the projected gate.
        """
        if region is None or self.target_hue is None:
            return None
        crop = color[region.top : region.bottom, region.left : region.right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        distance = np.abs(hsv[:, :, 0].astype(int) - self.target_hue)
        distance = np.minimum(distance, HueEncoding.PERIOD - distance)
        foreground = (
            (distance <= self.hue_tolerance)
            & (hsv[:, :, 1] >= self.minimum_saturation)
            & (hsv[:, :, 2] >= self.minimum_value)
        ).astype(np.uint8)
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            np.ones((self.closing_width, self.closing_width), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = sorted(
            (
                contour
                for contour in contours
                if cv2.contourArea(contour) >= self.minimum_area
            ),
            key=cv2.contourArea,
            reverse=True,
        )
        if not candidates or (
            len(candidates) > 1
            and cv2.contourArea(candidates[1])
            >= self.ambiguity_ratio * cv2.contourArea(candidates[0])
        ):
            return None
        hull = cv2.convexHull(candidates[0])
        left, top, width, height = cv2.boundingRect(hull)
        if (
            left == 0
            or top == 0
            or left + width == crop.shape[1]
            or top + height == crop.shape[0]
        ):
            return None
        mask = np.zeros(color.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull + np.array([[[region.left, region.top]]]), 1)
        boundary = ImageRegion(
            left + region.left,
            top + region.top,
            left + width + region.left,
            top + height + region.top,
        )
        center = np.array(
            [
                (boundary.left + boundary.right - 1) / 2,
                (boundary.top + boundary.bottom - 1) / 2,
            ]
        )
        return TargetMeasurement(center, boundary, mask.astype(bool))

    def follow_previous(
        self, grayscale: np.ndarray, region: ImageRegion | None
    ) -> FeatureObservation | None:
        """
        Propagate the hand target through forward-backward-verified optical flow.
        """
        if (
            self.previous is None
            or region is None
            or len(self.previous.observation.pixels) < self.minimum_matches
        ):
            return None
        previous_pixels = self.previous.observation.pixels.astype(np.float32).reshape(
            -1, 1, 2
        )
        current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous.grayscale, grayscale, previous_pixels, None
        )
        if current is None:
            return None
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            grayscale, self.previous.grayscale, current, None
        )
        if backward is None:
            return None
        valid = forward_status[:, 0].astype(bool) & backward_status[:, 0].astype(bool)
        valid &= (
            np.linalg.norm(backward[:, 0] - previous_pixels[:, 0], axis=1)
            <= self.flow_tolerance
        )
        pixels = current[:, 0]
        valid &= (
            (pixels[:, 0] > region.left)
            & (pixels[:, 0] < region.right)
            & (pixels[:, 1] > region.top)
            & (pixels[:, 1] < region.bottom)
        )
        if np.count_nonzero(valid) < self.minimum_matches:
            return None
        cv2.setRNGSeed(self.random_seed)
        transform, inliers = cv2.estimateAffinePartial2D(
            previous_pixels[valid, 0],
            pixels[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reprojection_threshold,
        )
        if (
            transform is None
            or inliers is None
            or np.count_nonzero(inliers) < self.minimum_matches
        ):
            return None
        center = transform @ np.append(self.previous.observation.measurement.center, 1)
        if (
            not region.left < center[0] < region.right
            or not region.top < center[1] < region.bottom
        ):
            return None
        mask = cv2.warpAffine(
            self.previous.observation.measurement.mask.astype(np.uint8),
            transform,
            (grayscale.shape[1], grayscale.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        left, top, width, height = cv2.boundingRect(mask)
        measurement = TargetMeasurement(
            center,
            ImageRegion(left, top, left + width, top + height),
            mask.astype(bool),
        )
        return FeatureObservation(
            measurement, pixels[valid][inliers[:, 0].astype(bool)]
        )
