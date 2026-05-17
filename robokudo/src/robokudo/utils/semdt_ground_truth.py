"""Helpers for reading SemDT ground-truth world data from CAS.

The world reference stored in CAS is an in-process object reference and must be
treated as strictly read-only by consumers.
"""

from __future__ import annotations

import numpy as np

from robokudo.cas import CASViews
from semantic_digital_twin.world_description.geometry import Box, Cylinder
from semantic_digital_twin.world_description.world_entity import Body

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


def get_body_from_runtime_world(cas: CAS, body_name: str) -> Body | None:
    """Return body object by name from runtime ground-truth world, or None."""
    name = str(body_name).strip()
    if name == "":
        return None

    world = get_ground_truth_world_ref(cas)
    if world is None:
        return None

    bodies = world.get_bodies_by_name(name)
    if len(bodies) == 0:
        return None
    return bodies[0]


def get_body_pose_world(body: Body) -> np.ndarray | None:
    """Return 4x4 body world pose matrix, or None if invalid."""
    pose_world = np.asarray(body.global_pose.to_np(), dtype=np.float64)
    if pose_world.shape != (4, 4) or not np.all(np.isfinite(pose_world)):
        return None
    return pose_world


def body_world_aabb_at_center(
    body: Body, center_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute world-aligned AABB for body primitive placed at center_world.

    Orientation is taken from the body's current runtime world pose.
    """
    if body.collision is None or len(body.collision) == 0:
        return None
    pose_world = get_body_pose_world(body)
    if pose_world is None:
        return None
    rotation_world = pose_world[:3, :3]
    if not np.all(np.isfinite(rotation_world)):
        return None

    shape = body.collision[0]
    if isinstance(shape, Box):
        scale = shape.scale
        if scale is None:
            return None
        half_extents_local = 0.5 * np.array(
            [float(scale.x), float(scale.y), float(scale.z)], dtype=np.float64
        )
        if np.any(~np.isfinite(half_extents_local)) or np.any(
            half_extents_local <= 0.0
        ):
            return None
        half_extents_world = np.abs(rotation_world) @ half_extents_local
    elif isinstance(shape, Cylinder):
        radius = 0.5 * float(shape.width)
        half_height = 0.5 * float(shape.height)
        if (
            radius <= 0.0
            or half_height <= 0.0
            or not np.all(np.isfinite([radius, half_height]))
        ):
            return None
        axis_world = rotation_world[:, 2].astype(np.float64)
        axis_norm = max(float(np.linalg.norm(axis_world)), 1e-9)
        axis_world = axis_world / axis_norm
        half_extents_world = np.zeros(3, dtype=np.float64)
        for axis_idx in range(3):
            axial_component = abs(float(axis_world[axis_idx]))
            radial_component = float(np.sqrt(max(0.0, 1.0 - (axial_component**2))))
            half_extents_world[axis_idx] = (axial_component * half_height) + (
                radial_component * radius
            )
    else:
        return None

    center = np.asarray(center_world, dtype=np.float64)
    if center.shape[0] < 3 or not np.all(np.isfinite(center[:3])):
        return None
    center = center[:3]
    return center - half_extents_world, center + half_extents_world


def body_support_extent_along_normal(
    body: Body, normal_world: np.ndarray
) -> float | None:
    """Return support-function extent from body center along normal_world."""
    if body.collision is None or len(body.collision) == 0:
        return None
    pose_world = get_body_pose_world(body)
    if pose_world is None:
        return None
    rotation_world = pose_world[:3, :3]
    shape = body.collision[0]
    normal = np.asarray(normal_world, dtype=np.float64)
    if normal.shape[0] < 3 or not np.all(np.isfinite(normal[:3])):
        return None
    normal = normal[:3]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        return None
    normal = normal / normal_norm

    if isinstance(shape, Box):
        scale = shape.scale
        if scale is None:
            return None
        half_extents = 0.5 * np.array(
            [float(scale.x), float(scale.y), float(scale.z)], dtype=np.float64
        )
        if np.any(~np.isfinite(half_extents)) or np.any(half_extents <= 0.0):
            return None
        normal_body = rotation_world.T @ normal
        return float(np.sum(np.abs(normal_body) * half_extents))

    if isinstance(shape, Cylinder):
        radius = 0.5 * float(shape.width)
        half_height = 0.5 * float(shape.height)
        if (
            radius <= 0.0
            or half_height <= 0.0
            or not np.all(np.isfinite([radius, half_height]))
        ):
            return None
        axis_world = rotation_world[:, 2].astype(np.float64)
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm < 1e-9:
            axis_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis_world = axis_world / axis_norm
        axial_component = abs(float(np.dot(normal, axis_world)))
        radial_component = float(np.sqrt(max(0.0, 1.0 - (axial_component**2))))
        return float((axial_component * half_height) + (radial_component * radius))

    return None


def body_aabb_intersects_other_collidable_bodies(
    cas: CAS,
    body: Body,
    center_world: np.ndarray,
    epsilon_m: float = 1e-4,
) -> bool:
    """Check whether body AABB at center_world overlaps any other collidable body AABB."""
    world = get_ground_truth_world_ref(cas)
    if world is None:
        return False

    candidate_bounds = body_world_aabb_at_center(body=body, center_world=center_world)
    if candidate_bounds is None:
        return False
    candidate_min, candidate_max = candidate_bounds

    candidate_bodies = world.get_kinematic_structure_entity_by_type(Body)
    if candidate_bodies is None:
        return False

    for obstacle_body in candidate_bodies:
        if (
            obstacle_body is body
            or obstacle_body.collision is None
            or len(obstacle_body.collision) == 0
        ):
            continue
        obstacle_bb = obstacle_body.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_box()
        obstacle_min = np.array(
            [
                float(obstacle_bb.origin.x + obstacle_bb.min_x),
                float(obstacle_bb.origin.y + obstacle_bb.min_y),
                float(obstacle_bb.origin.z + obstacle_bb.min_z),
            ],
            dtype=np.float64,
        )
        obstacle_max = np.array(
            [
                float(obstacle_bb.origin.x + obstacle_bb.max_x),
                float(obstacle_bb.origin.y + obstacle_bb.max_y),
                float(obstacle_bb.origin.z + obstacle_bb.max_z),
            ],
            dtype=np.float64,
        )
        overlap = np.minimum(candidate_max, obstacle_max) - np.maximum(
            candidate_min, obstacle_min
        )
        if np.all(overlap > float(epsilon_m)):
            return True
    return False
