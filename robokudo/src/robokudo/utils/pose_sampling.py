"""Utilities for sampling candidate object poses in 3D."""

from __future__ import annotations

import numpy as np
from typing_extensions import Callable


def create_sampling_rng(
    current_rng: np.random.Generator | None,
    current_seed: int | None,
    configured_seed: int,
) -> tuple[np.random.Generator, int]:
    """Create or reuse RNG based on configured seed."""
    if current_rng is None or current_seed != int(configured_seed):
        if int(configured_seed) >= 0:
            return np.random.default_rng(int(configured_seed)), int(configured_seed)
        return np.random.default_rng(), int(configured_seed)
    return current_rng, int(current_seed)


def sample_random_offset_translation(
    center_world: np.ndarray,
    magnitude_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one isotropic random 3D offset around center."""
    if magnitude_m <= 0.0:
        return center_world.astype(np.float64).copy()
    random_direction = rng.normal(size=3).astype(np.float64)
    direction_norm = float(np.linalg.norm(random_direction))
    if direction_norm < 1e-9:
        random_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        direction_norm = 1.0
    random_direction /= direction_norm
    random_radius = float(rng.uniform(0.0, magnitude_m))
    return center_world.astype(np.float64) + (random_direction * random_radius)


def sample_candidate_centers_free_xyz(
    seed_center_world: np.ndarray,
    sample_count: int,
    include_seed: bool,
    sampling_radius_m: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample candidate centers with unconstrained XYZ perturbations."""
    candidates: list[np.ndarray] = []
    if include_seed:
        candidates.append(seed_center_world.astype(np.float64).copy())

    for _ in range(sample_count):
        candidates.append(
            sample_random_offset_translation(
                center_world=seed_center_world,
                magnitude_m=sampling_radius_m,
                rng=rng,
            )
        )
    return candidates


def sample_candidate_centers_free_xyz_non_intersecting(
    seed_center_world: np.ndarray,
    sample_count: int,
    include_seed: bool,
    sampling_radius_m: float,
    max_trials: int,
    rng: np.random.Generator,
    intersects_world: Callable[[np.ndarray], bool],
) -> list[np.ndarray]:
    """Sample free-XYZ candidates that satisfy an external non-intersection check."""
    candidates: list[np.ndarray] = []
    if include_seed and not intersects_world(seed_center_world):
        candidates.append(seed_center_world.astype(np.float64).copy())

    trials = 0
    while len(candidates) < sample_count and trials < max_trials:
        trials += 1
        candidate_center = sample_random_offset_translation(
            center_world=seed_center_world,
            magnitude_m=sampling_radius_m,
            rng=rng,
        )
        if intersects_world(candidate_center):
            continue
        candidates.append(candidate_center)
    return candidates


def sample_candidate_centers_support_surface(
    seed_center_world: np.ndarray,
    sample_count: int,
    include_seed: bool,
    sampling_radius_m: float,
    rng: np.random.Generator,
    project_to_support_surface: Callable[[np.ndarray], np.ndarray],
    normal_world: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Sample candidate centers in local tangent plane around projected seed center."""
    seed_projected = project_to_support_surface(
        seed_center_world.astype(np.float64).copy()
    )
    if seed_projected.shape[0] < 3:
        raise ValueError("project_to_support_surface must return a 3D point.")

    if normal_world is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = np.asarray(normal_world, dtype=np.float64)
        if normal.shape[0] < 3 or np.linalg.norm(normal[:3]) < 1e-9:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normal = normal[:3]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)

    tangent_a = np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    if np.linalg.norm(tangent_a) < 1e-6:
        tangent_a = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    tangent_a = tangent_a / max(float(np.linalg.norm(tangent_a)), 1e-9)
    tangent_b = np.cross(normal, tangent_a)
    tangent_b = tangent_b / max(float(np.linalg.norm(tangent_b)), 1e-9)

    candidates: list[np.ndarray] = []
    if include_seed:
        candidates.append(seed_projected.copy())

    for _ in range(sample_count):
        radius = float(rng.uniform(0.0, sampling_radius_m))
        angle = float(rng.uniform(-np.pi, np.pi))
        lateral_offset = (
            np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
        ) * radius
        candidate = seed_projected + lateral_offset
        candidates.append(project_to_support_surface(candidate))
    return candidates
