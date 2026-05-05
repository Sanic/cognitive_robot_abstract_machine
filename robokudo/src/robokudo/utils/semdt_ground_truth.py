"""Helpers for reading SemDT ground-truth world data from CAS.

The world reference stored in CAS is an in-process object reference and must be
treated as strictly read-only by consumers.
"""

from __future__ import annotations

import numpy as np

from robokudo.cas import CASViews

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from robokudo.cas import CAS
    from semantic_digital_twin.world import World


def get_ground_truth_world_ref(cas: CAS) -> World | None:
    """Return ground-truth world reference from CAS, or None if unavailable."""
    if cas is None or not cas.contains(CASViews.GROUND_TRUTH_WORLD_REF):
        return None
    return cas.get(CASViews.GROUND_TRUTH_WORLD_REF)


def get_gt_pose_from_runtime_world(cas: CAS, body_name: str) -> np.ndarray | None:
    """Return 4x4 world pose for body_name from runtime world, or None.

    Notes:
    - This function reads current data only; callers should not cache Body refs.
    - Returned matrix is a copy and safe for downstream use.
    """
    name = str(body_name).strip()
    if name == "":
        return None

    world = get_ground_truth_world_ref(cas)
    if world is None:
        return None

    bodies = world.get_bodies_by_name(name)
    if len(bodies) == 0:
        return None

    pose_world = np.asarray(bodies[0].global_pose.to_np(), dtype=np.float64)
    if pose_world.shape != (4, 4) or not np.all(np.isfinite(pose_world)):
        return None
    return pose_world.copy()


def get_runtime_world_ref(cas: CAS) -> World | None:
    """Backward-compatible alias for get_ground_truth_world_ref."""
    return get_ground_truth_world_ref(cas)
