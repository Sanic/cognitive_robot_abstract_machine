"""Utilities for image-driven translation-only pose refinement."""

from __future__ import annotations

import numpy as np
from typing_extensions import Callable


def pose_agreement_score(
    outline_score: float,
    pixel_error: float,
    centroid_scale_px: float,
    centroid_weight: float,
) -> float:
    """Combine silhouette score with centroid closeness into one scalar objective."""
    scale_px = max(float(centroid_scale_px), 1e-3)
    weight = max(float(centroid_weight), 0.0)
    clipped_px = float(np.clip(pixel_error, 0.0, scale_px))
    normalized_penalty = clipped_px / scale_px
    if not np.isfinite(pixel_error):
        normalized_penalty = 1.0
    return float(outline_score) - (weight * normalized_penalty)


def is_refinement_result_better(
    candidate_outline_score: float,
    candidate_pixel_error: float,
    incumbent_outline_score: float,
    incumbent_pixel_error: float,
    centroid_scale_px: float,
    centroid_weight: float,
) -> bool:
    """Rank results by joint silhouette/centroid agreement, then silhouette, then centroid."""
    candidate_agreement = pose_agreement_score(
        outline_score=float(candidate_outline_score),
        pixel_error=float(candidate_pixel_error),
        centroid_scale_px=centroid_scale_px,
        centroid_weight=centroid_weight,
    )
    incumbent_agreement = pose_agreement_score(
        outline_score=float(incumbent_outline_score),
        pixel_error=float(incumbent_pixel_error),
        centroid_scale_px=centroid_scale_px,
        centroid_weight=centroid_weight,
    )
    if candidate_agreement > incumbent_agreement + 1e-6:
        return True
    if candidate_agreement < incumbent_agreement - 1e-6:
        return False

    candidate_score = float(candidate_outline_score)
    incumbent_score = float(incumbent_outline_score)
    if candidate_score > incumbent_score + 1e-6:
        return True
    if candidate_score < incumbent_score - 1e-6:
        return False

    candidate_px = float(candidate_pixel_error)
    incumbent_px = float(incumbent_pixel_error)
    if np.isfinite(candidate_px) and not np.isfinite(incumbent_px):
        return True
    if np.isfinite(candidate_px) and np.isfinite(incumbent_px):
        return candidate_px < incumbent_px - 1e-3
    return False


def estimate_translation_shift_world(
    current_center_world: np.ndarray,
    expected_mask: np.ndarray,
    detected_mask: np.ndarray,
    mask_centroid_fn: Callable[[np.ndarray], np.ndarray | None],
    centroid_shift_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None],
    render_mask_for_center_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_delta_m: float,
    max_step_m: float,
    project_center_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray | None, int, str | None]:
    """Estimate translation step in world frame from centroid error and local Jacobian.

    Returns:
    - shift vector or None
    - number of extra renders used for Jacobian
    - early-exit reason or None on success
    """
    jacobian_render_count = 0
    centroid_shift_px = centroid_shift_fn(expected_mask, detected_mask)
    expected_center_px = mask_centroid_fn(expected_mask)
    if centroid_shift_px is None or expected_center_px is None:
        return None, jacobian_render_count, "centroid"

    delta_m = max(float(jacobian_delta_m), 1e-4)
    jacobian = np.zeros((2, 3), dtype=np.float64)
    for axis in range(3):
        perturbed_center = current_center_world.copy()
        perturbed_center[axis] += delta_m
        if project_center_fn is not None:
            perturbed_center = project_center_fn(perturbed_center)
        perturbed_mask = render_mask_for_center_fn(perturbed_center)
        jacobian_render_count += 1
        perturbed_center_px = mask_centroid_fn(perturbed_mask)
        if perturbed_center_px is None:
            continue
        jacobian[:, axis] = (perturbed_center_px - expected_center_px) / delta_m

    if np.linalg.norm(jacobian) < 1e-8:
        return None, jacobian_render_count, "jacobian"

    shift_world, _, _, _ = np.linalg.lstsq(jacobian, centroid_shift_px, rcond=None)
    if not np.all(np.isfinite(shift_world)):
        return None, jacobian_render_count, "invalid_lstsq"

    capped_max_step_m = max(float(max_step_m), 1e-4)
    step_norm = float(np.linalg.norm(shift_world))
    if step_norm > capped_max_step_m:
        shift_world = shift_world * (capped_max_step_m / step_norm)
    return shift_world, jacobian_render_count, None
