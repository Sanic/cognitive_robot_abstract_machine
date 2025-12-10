import os
from pathlib import Path

import cv2
import numpy as np
import trimesh

from semantic_digital_twin.adapters.mesh import OBJParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    TransformGeometry,
    CenterLocalGeometryAndPreserveWorldPose,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.utils import InheritanceStructureExporter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# dir_path = "/home/itsme/work/cram_ws/src/cognitive_robot_abstract_machine/semantic_digital_twin/resources/warsaw_data/objects/"
dir_path = "/home/pmania/warsaw/src/cognitive_robot_abstract_machine/semantic_digital_twin/resources/warsaw_data/objects/"
files = [f for f in os.listdir(dir_path) if f.endswith(".obj")]

world = World()
root = Body(name=PrefixedName("root_body"))
with world.modify_world():
    world.add_body(root)
for i, file in enumerate(files):
    obj_world = OBJParser(os.path.join(dir_path, file)).parse()
    with world.modify_world():
        # obj_world.bodies[0].collision[0].override_mesh_with_color(color)
        world.merge_world(obj_world)

pipeline = Pipeline(
    steps=[
        TransformGeometry(
            TransformationMatrix.from_xyz_rpy(roll=np.pi / 2, pitch=0, yaw=0)
        ),
        CenterLocalGeometryAndPreserveWorldPose(),
    ]
)
world = pipeline.apply(world)
output_path = Path("../resources/warsaw_data/json_exports/")
if not output_path.exists():
    output_path.mkdir(parents=True)
world.export_kinematic_structure_tree_to_json(
    Path(os.path.join(output_path, "kinematic_structure.json")),
    include_connections=False,
)

InheritanceStructureExporter(
    SemanticAnnotation, Path(os.path.join(output_path, "semantic_annotations.json"))
).export()

rt = RayTracer(world=world)
scene = rt.scene

# import trimesh
# from semantic_digital_twin.spatial_types import TransformationMatrix

# Add simple sphere
# s = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
# # Set color (RGBA)
# s.visual.vertex_colors = trimesh.visual.color.to_rgba([255, 80, 80, 255])
#
# T = TransformationMatrix.from_xyz_rpy(x=1.0, y=0.5, z=0.8).to_np()
# scene.add_geometry(
#     s, node_name="debug_sphere_temp", parent_node_name="world", transform=T
# )


####
test_fov = [60, 45]  # horizontal, vertical degrees
min_standoff_distance = 1.5
max_view_distance = 5.0
# The frustum culling distance should be larger than the maximum camera placement distance
# so that visibility evaluation is not clipped too early.
frustum_culling_max_distance = 8.0
# Camera height filter: poses outside this [min, max] world Z interval are rejected early
# Defaults allow all heights; set to finite values to enable filtering
min_camera_height = 0.5  # -float('inf')
max_camera_height = 2.0  # float('inf')
###


import numpy as np
import trimesh
from dataclasses import dataclass
import time
from trimesh.scene.cameras import Camera


@dataclass
class CameraPose:
    """
    Small container for a camera and its pose in the world.

    The transform maps the camera frame to the world frame.
    """

    camera: Camera
    transform: np.ndarray
    node_name: str
    geometry_key: str


# These are just defaults. Make sure to change the values way below in the instantiation
@dataclass
class ConeSamplingConfig:
    """
    Configuration for sampling camera view directions on a cone around the mean normal.

    The cone is a spherical cap with the axis aligned to the object's mean normal and
    a half-angle of ``cone_half_angle_deg``.
    """

    cone_half_angle_deg: float = 25
    samples_per_cone: int = 12
    roll_samples: int = 1
    seed: int | None = None
    fit_method: str = "footprint_2d"  # or "spherical"
    # If True, reject poses whose directional viewing cones overlap existing ones (per object)
    reject_overlapping_cones: bool = True  # True
    # Small angular margin added to the cone-overlap test (degrees)
    cone_overlap_margin_deg: float = 5.0


@dataclass
class Timer:
    """
    Simple context manager to measure elapsed wall time.

    Prints a line when exiting the context with the elapsed time in milliseconds.
    """

    label: str

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        print(f"[TIMING] {self.label}: {dt_ms:.2f} ms")


def timeit(label: str | None = None):
    """
    Decorator to print how long a function call took.

    If ``label`` is None, the function's __name__ is used.
    """

    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                name = label if label is not None else fn.__name__
                print(f"[TIMING] {name}: {dt_ms:.2f} ms")

        _wrapped.__name__ = fn.__name__
        _wrapped.__doc__ = fn.__doc__
        _wrapped.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
        return _wrapped

    return _decorator


def _is_height_allowed(z_world: float, min_h: float, max_h: float) -> bool:
    """
    Return True if the world Z coordinate lies within [min_h, max_h].
    """
    return (z_world >= min_h) and (z_world <= max_h)


@timeit("compute_fit_distance_spherical_bound")
def compute_fit_distance_spherical_bound(
    radius_world: float, tx: float, ty: float, margin: float
) -> float:
    """
    Compute a conservative camera standoff distance using a spherical bound.

    The method treats the object as a sphere with radius equal to the maximum
    distance from the centroid to the transformed AABB corners and computes the
    distance required to fit the sphere within both horizontal and vertical
    fields of view.
    """
    return margin * max(
        (radius_world / tx) if tx > 1e-12 else np.inf,
        (radius_world / ty) if ty > 1e-12 else np.inf,
    )


@timeit("compute_fit_distance_footprint_2d")
def compute_fit_distance_footprint_2d(
    corners_world: np.ndarray,
    centroid_world: np.ndarray,
    view_normal_world: np.ndarray,
    tx: float,
    ty: float,
    margin: float,
) -> float:
    """
    Compute camera standoff distance from the 2D image-plane footprint.

    The method builds a temporary camera frame oriented by the view normal and
    projects the object’s world-space AABB corners into this frame. The in-plane
    half-extents along X and Y determine the minimum distance to fit within the
    horizontal and vertical FOV, ignoring thickness along the view axis.
    """
    z_cam = -view_normal_world
    x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.cross(np.array([0.0, 1.0, 0.0]), z_cam)
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-12)
    y_cam = np.cross(z_cam, x_cam)
    offsets_world = corners_world - centroid_world
    R_world_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    offsets_cam = offsets_world @ R_world_cam
    max_abs_x = float(np.max(np.abs(offsets_cam[:, 0]))) if offsets_cam.size else 0.0
    max_abs_y = float(np.max(np.abs(offsets_cam[:, 1]))) if offsets_cam.size else 0.0

    dx = max_abs_x / (tx if tx > 1e-12 else np.inf)
    dy = max_abs_y / (ty if ty > 1e-12 else np.inf)
    return margin * max(dx, dy)


@timeit("add_mean_normal_lines_and_cameras")
def add_mean_normal_lines_and_cameras(
    scene,
    normal_length: float | None = None,
    marker_height: float = 0.15,
    *,
    min_standoff: float | None = None,
    max_distance: float | None = None,
    fov_deg_xy: tuple[float, float] | None = None,
    fit_method: str = "footprint_2d",  # "spherical" is more conservative
    min_camera_height_override: float | None = None,
    max_camera_height_override: float | None = None,
) -> list[CameraPose]:
    import numpy as np
    import trimesh
    from trimesh.scene.cameras import Camera

    def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return v
        return v / n

    def look_at_transform(
        origin: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None
    ) -> np.ndarray:
        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0])
        f = _safe_normalize(target - origin)  # forward (world)
        z_cam = -f  # camera -Z
        x_cam = np.cross(up_hint, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            up_hint = np.array([0.0, 1.0, 0.0])
            x_cam = np.cross(up_hint, z_cam)
        x_cam = _safe_normalize(x_cam)
        y_cam = _safe_normalize(np.cross(z_cam, x_cam))
        T = np.eye(4)
        T[:3, 0] = x_cam
        T[:3, 1] = y_cam
        T[:3, 2] = z_cam
        T[:3, 3] = origin
        return T

    line_color = np.array([255, 32, 32, 255], dtype=np.uint8)

    # We will create one Camera instance per generated pose so callers can keep
    # and modify them independently later on.
    # This template defines default parameters for each instance.
    def _new_camera_instance() -> Camera:
        return Camera(resolution=(640, 480), fov=test_fov)

    # Resolve configuration with sensible defaults from module-level settings
    # normal_length (if provided) is treated as the minimum standoff distance for backwards compatibility
    effective_min_standoff = (
        min_standoff
        if min_standoff is not None
        else (normal_length if normal_length is not None else 1.0)
    )
    effective_max_distance = (
        max_distance if max_distance is not None else max_view_distance
    )
    fov_xy = (
        fov_deg_xy
        if fov_deg_xy is not None
        else (float(test_fov[0]), float(test_fov[1]))
    )

    # Precompute FOV tangents
    fovx_deg, fovy_deg = float(fov_xy[0]), float(fov_xy[1])
    tx = np.tan(np.deg2rad(fovx_deg) * 0.5)
    ty = np.tan(np.deg2rad(fovy_deg) * 0.5)

    margin_factor = 1.05  # small safety margin to ensure full coverage

    # Resolve height constraints
    allowed_min_h = (
        float(min_camera_height_override)
        if min_camera_height_override is not None
        else float(min_camera_height)
    )
    allowed_max_h = (
        float(max_camera_height_override)
        if max_camera_height_override is not None
        else float(max_camera_height)
    )

    camera_poses: list[CameraPose] = []

    # use module-level distance calculators defined above

    for node_name in scene.graph.nodes_geometry:
        _, gkey = scene.graph[node_name]
        geom = scene.geometry[gkey]
        if not isinstance(geom, trimesh.Trimesh):
            continue

        T_node, _ = scene.graph.get(
            frame_to=node_name, frame_from=scene.graph.base_frame
        )
        R = T_node[:3, :3]
        t = T_node[:3, 3]

        if geom.faces.shape[0] == 0:
            continue
        face_normals = geom.face_normals
        if hasattr(geom, "area_faces") and geom.area_faces is not None:
            w = geom.area_faces.reshape(-1, 1)
            n_local = (face_normals * w).sum(axis=0)
        else:
            n_local = face_normals.mean(axis=0)
        n_local = _safe_normalize(n_local)
        if np.linalg.norm(n_local) < 1e-12:
            continue

        c_local = geom.center_mass if geom.is_volume else geom.centroid
        c_world = (R @ c_local) + t
        n_world = _safe_normalize(R @ n_local)
        if np.linalg.norm(n_world) < 1e-12:
            continue

        # Compute transformed local AABB corners used by fit-distance methods
        corners_local = np.array(
            [
                [geom.bounds[0][0], geom.bounds[0][1], geom.bounds[0][2]],
                [geom.bounds[0][0], geom.bounds[0][1], geom.bounds[1][2]],
                [geom.bounds[0][0], geom.bounds[1][1], geom.bounds[0][2]],
                [geom.bounds[0][0], geom.bounds[1][1], geom.bounds[1][2]],
                [geom.bounds[1][0], geom.bounds[0][1], geom.bounds[0][2]],
                [geom.bounds[1][0], geom.bounds[0][1], geom.bounds[1][2]],
                [geom.bounds[1][0], geom.bounds[1][1], geom.bounds[0][2]],
                [geom.bounds[1][0], geom.bounds[1][1], geom.bounds[1][2]],
            ],
            dtype=float,
        )
        corners_world = (R @ corners_local.T).T + t
        # Pick fit-distance method
        if fit_method.lower() in {"footprint", "2d", "plane", "footprint_2d"}:
            fit_distance = compute_fit_distance_footprint_2d(
                corners_world=corners_world,
                centroid_world=c_world,
                view_normal_world=n_world,
                tx=tx,
                ty=ty,
                margin=margin_factor,
            )
        else:
            # radius is the max distance from centroid to any corner
            radius_world = float(
                np.max(np.linalg.norm(corners_world - c_world, axis=1))
            )
            fit_distance = compute_fit_distance_spherical_bound(
                radius_world=radius_world, tx=tx, ty=ty, margin=margin_factor
            )

        # Final standoff distance with min and max bounds
        standoff_distance = max(effective_min_standoff, fit_distance)
        if effective_max_distance is not None:
            standoff_distance = min(standoff_distance, effective_max_distance)

        p0 = c_world
        p1 = c_world + standoff_distance * n_world

        # Early reject by camera height (use z of camera origin p1)
        if not _is_height_allowed(float(p1[2]), allowed_min_h, allowed_max_h):
            continue

        path = trimesh.load_path(np.vstack([p0, p1]))
        path.colors = np.tile(line_color, (len(path.entities), 1))
        line_node_name = f"normal_line__{node_name}__{gkey}"
        scene.add_geometry(
            path,
            node_name=line_node_name,
            parent_node_name=scene.graph.base_frame,
            transform=np.eye(4),
        )

        T_cam = look_at_transform(origin=p1, target=p0)

        # Create a distinct camera instance for this pose
        pose_cam = _new_camera_instance()
        cam_marker = trimesh.creation.camera_marker(
            pose_cam, marker_height=marker_height
        )
        cam_node_name = f"camera_marker__{node_name}__{gkey}"
        scene.add_geometry(
            cam_marker,
            node_name=cam_node_name,
            parent_node_name=scene.graph.base_frame,
            transform=T_cam,
        )

        camera_poses.append(
            CameraPose(
                camera=pose_cam, transform=T_cam, node_name=node_name, geometry_key=gkey
            )
        )

    return camera_poses


# Generate visualization and collect camera instances and their poses

generated_camera_poses = add_mean_normal_lines_and_cameras(
    scene,
    marker_height=min_standoff_distance,
    min_standoff=min_standoff_distance,
    max_distance=max_view_distance,
    fov_deg_xy=(float(test_fov[0]), float(test_fov[1])),
    # fit_method="footprint_2d",
)

# Set the z_far distance of all cameras to a fixed value used by frustum culling.
# This is intentionally larger than the maximum camera placement distance so that
# culling/visibility evaluation does not prematurely clip distant objects.
for pose in generated_camera_poses:
    pose.camera.z_far = frustum_culling_max_distance  # meters

# -----------------------------------------------------------------------------
# Camera origin spheres (visual markers placed at each generated camera origin)
# -----------------------------------------------------------------------------


@timeit("add_camera_origin_spheres")
def add_camera_origin_spheres(
    scene: trimesh.Scene,
    camera_poses: list[CameraPose],
    radius: float = 0.05,
    color_rgba: tuple[int, int, int, int] | None = None,
    *,
    occlusion_check: bool = False,
    samples_per_mesh: int = 32,
    visibility_threshold: float = 0.8,
) -> None:
    """
    Add a small sphere at each camera origin and color it by how many bodies
    are fully inside the camera frustum.

    If ``color_rgba`` is None, a two-color gradient is used (blue → red), where
    blue means few objects fully visible and red means many objects fully visible.

    :param scene: The scene to which the spheres are added.
    :param camera_poses: List of camera poses whose origins are to be marked.
    :param radius: Radius of the marker spheres in meters.
    :param color_rgba: If provided, overrides data-driven colors with this RGBA.
    """

    def _count_fully_visible_bodies(pose: CameraPose) -> int:
        # Use frustum culling to get fully inside nodes, then map to unique body indices
        try:
            fully_inside, _ = frustum_cull_scene(
                scene,
                pose.camera,
                pose.transform,
                require_full_visibility=True,
                occlusion_check=occlusion_check,
                samples_per_mesh=samples_per_mesh,
                visibility_threshold=visibility_threshold,
            )
        except NameError:
            # Fallback if frustum function not yet defined
            fully_inside = set()

        # Prefer RayTracer mapping if available
        body_indices = set()
        mapping = None
        try:
            mapping = rt.scene_to_index  # type: ignore[name-defined]
        except Exception:
            mapping = None

        if mapping is not None:
            for node in fully_inside:
                if node in mapping:
                    body_indices.add(mapping[node])
        else:
            # Fallback: deduplicate by body name prefix before '_collision_'
            for node in fully_inside:
                if "_collision_" in node:
                    body_indices.add(node.split("_collision_")[0])

        return len(body_indices)

    # Compute counts per pose for normalization
    counts = [
        _count_fully_visible_bodies(pose) if color_rgba is None else 0
        for pose in camera_poses
    ]
    max_count = max(counts) if counts else 0

    def _val_to_rgba(v: float) -> tuple[int, int, int, int]:
        """
        Map a normalized value v in [0,1] to a color using a two-color gradient
        from blue (low) to red (high).
        """
        v = float(max(0.0, min(1.0, v)))
        low = np.array([0, 0, 255, 255], dtype=float)  # blue
        high = np.array([255, 0, 0, 255], dtype=float)  # red
        rgba = (1.0 - v) * low + v * high
        return tuple(int(round(c)) for c in rgba)

    for idx, pose in enumerate(camera_poses):
        # Determine color
        if color_rgba is None:
            v = (counts[idx] / max_count) if max_count > 0 else 0.0
            rgba = _val_to_rgba(v)
        else:
            rgba = color_rgba

        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.visual.vertex_colors = trimesh.visual.color.to_rgba(rgba)

        # Use a unique node name per pose to avoid collisions when multiple
        # camera poses originate from the same mesh/node (e.g., cone sampling).
        node_name = (
            f"camera_origin_sphere__{pose.node_name}__{pose.geometry_key}__{idx}"
        )
        scene.add_geometry(
            sphere,
            node_name=node_name,
            parent_node_name=scene.graph.base_frame,
            transform=pose.transform,
        )


# Note: spheres are placed after frustum utilities are defined (see below) so
# they can be colored by visibility counts.

# -----------------------------------------------------------------------------
# Frustum culling utilities (Option 2: camera-space FOV angle tests)
# -----------------------------------------------------------------------------


def _aabb_corners(bounds: np.ndarray) -> np.ndarray:
    """
    Return the eight corners of an axis aligned bounding box.
    """
    mn, mx = bounds
    xs = [mn[0], mx[0]]
    ys = [mn[1], mx[1]]
    zs = [mn[2], mx[2]]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def camera_view_matrix(T_cam_world: np.ndarray) -> np.ndarray:
    """
    Compute the view matrix V that maps world -> camera.

    Assumes T_cam_world maps camera coordinates to world coordinates and
    that the camera looks along its local negative Z axis.
    """
    return np.linalg.inv(T_cam_world)


@timeit("frustum_cull_scene")
def frustum_cull_scene(
    scene: trimesh.Scene,
    camera: Camera,
    T_cam_world: np.ndarray,
    require_full_visibility: bool = True,
    eps: float = 1e-6,
    *,
    occlusion_check: bool = False,
    samples_per_mesh: int = 32,
    visibility_threshold: float = 0.8,
) -> tuple[set[str], set[str]]:
    """
    Test scene meshes against the camera frustum using camera-space FOV tests.

    Convention: camera looks along negative Z. Points in front have z_cam < 0.
    Returns two sets of node names: fully_inside and partially_inside.

    If ``occlusion_check`` is True, candidate meshes are validated by ray casting
    from the camera origin to samples on the mesh. For a mesh to count as fully
    visible, at least ``visibility_threshold`` fraction of the sampled points
    must be the first hit in the scene.
    """
    V = camera_view_matrix(T_cam_world)

    # FOV to tangents
    fovx_deg, fovy_deg = float(camera.fov[0]), float(camera.fov[1])
    tx = np.tan(np.deg2rad(fovx_deg) * 0.5)
    ty = np.tan(np.deg2rad(fovy_deg) * 0.5)
    z_near = float(camera.z_near)
    z_far = float(camera.z_far)

    fully_inside: set[str] = set()
    partially_inside: set[str] = set()

    # Prepare ray engine if we do occlusion
    ray_engine = scene.to_mesh().ray if occlusion_check else None

    for node_name in scene.graph.nodes_geometry:
        _, gkey = scene.graph[node_name]
        geom = scene.geometry[gkey]
        if not isinstance(geom, trimesh.Trimesh):
            continue

        # Local AABB corners -> world
        corners_local = _aabb_corners(geom.bounds)
        T_node_world, _ = scene.graph.get(
            frame_to=node_name, frame_from=scene.graph.base_frame
        )
        corners_world = (T_node_world[:3, :3] @ corners_local.T).T + T_node_world[:3, 3]

        # World -> camera
        N = len(corners_world)
        pts_h = np.hstack([corners_world, np.ones((N, 1))])
        pc = (V @ pts_h.T).T  # (N,4)
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]

        # Camera looks along -Z; distance in front:
        dz = -z

        inside_mask = (
            (dz >= z_near - eps)
            & (dz <= z_far + eps)
            & (np.abs(x) <= dz * tx + eps)
            & (np.abs(y) <= dz * ty + eps)
        )

        all_inside = bool(np.all(inside_mask))
        any_inside = bool(np.any(inside_mask))

        candidate_full = False
        candidate_partial = False
        if require_full_visibility:
            candidate_full = all_inside
        else:
            candidate_partial = any_inside

        if not (candidate_full or candidate_partial):
            continue

        if occlusion_check and ray_engine is not None:
            # Sample points on the mesh surface in local frame
            try:
                n_samples = max(8, int(samples_per_mesh))
                pts_local = geom.sample(n_samples)
                if pts_local.shape[0] == 0:
                    # fall back to AABB corners and centroid
                    aabb = _aabb_corners(geom.bounds)
                    centroid = geom.centroid.reshape(1, 3)
                    pts_local = np.vstack([aabb, centroid])
            except Exception:
                aabb = _aabb_corners(geom.bounds)
                centroid = geom.centroid.reshape(1, 3)
                pts_local = np.vstack([aabb, centroid])

            # Transform samples to world
            pts_world = (T_node_world[:3, :3] @ pts_local.T).T + T_node_world[:3, 3]

            # Build rays from camera origin
            origin = T_cam_world[:3, 3]
            dirs = pts_world - origin
            dists = np.linalg.norm(dirs, axis=1)
            valid = dists > eps
            if not np.any(valid):
                continue
            dirs = dirs[valid] / dists[valid][:, None]
            pts_world = pts_world[valid]
            dists = dists[valid]

            # Intersect
            try:
                loc, idx_ray, _ = ray_engine.intersects_location(
                    ray_origins=np.repeat(origin[None, :], len(dirs), axis=0),
                    ray_directions=dirs,
                    multiple_hits=False,
                )
            except Exception:
                loc = np.empty((0, 3))
                idx_ray = np.empty((0,), dtype=int)

            # Count rays where first hit distance matches the sampled point distance
            visible = 0
            if len(idx_ray) > 0:
                hit_d = np.linalg.norm(loc - origin, axis=1)
                # Map back to corresponding target distances
                target_d = dists[idx_ray]
                # If the first intersection is at the same distance (+/- tol), it is not occluded
                visible = int(np.sum(np.abs(hit_d - target_d) <= 1e-3))
            # consider rays that did not hit anything as not visible
            visibility_ratio = visible / max(1, len(dirs))

            if require_full_visibility:
                candidate_full = candidate_full and (
                    visibility_ratio >= visibility_threshold
                )
                if not candidate_full:
                    continue
            else:
                # For partial case, require that at least one sample is visible
                candidate_partial = candidate_partial and (visibility_ratio > 0.0)
                if not candidate_partial:
                    continue

        if candidate_full:
            fully_inside.add(node_name)
        elif candidate_partial:
            partially_inside.add(node_name)

    return fully_inside, partially_inside


import numpy as np

# -----------------------------------------------------------------------------
# Cone-based camera pose generation around object normals
# -----------------------------------------------------------------------------


@timeit("generate_cone_view_poses")
def generate_cone_view_poses(
    scene: trimesh.Scene,
    *,
    cone: ConeSamplingConfig,
    min_standoff: float,
    max_distance: float,
    frustum_culling_max_distance: float,
    fov_deg_xy: tuple[float, float],
    visualize: bool = True,
    marker_height: float = 0.15,
    min_camera_height_override: float | None = None,
    max_camera_height_override: float | None = None,
) -> list[CameraPose]:
    """
    Generate additional camera poses by sampling directions within a cone
    around each mesh's mean normal and placing cameras that look at the
    mesh centroid. Returns a list of CameraPose.
    """
    from trimesh.scene.cameras import Camera as TrimeshCamera

    def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return v
        return v / n

    def look_at_transform(
        origin: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None
    ) -> np.ndarray:
        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0])
        f = _safe_normalize(target - origin)
        z_cam = -f
        x_cam = np.cross(up_hint, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            up_hint = np.array([0.0, 1.0, 0.0])
            x_cam = np.cross(up_hint, z_cam)
        x_cam = _safe_normalize(x_cam)
        y_cam = _safe_normalize(np.cross(z_cam, x_cam))
        T = np.eye(4)
        T[:3, 0] = x_cam
        T[:3, 1] = y_cam
        T[:3, 2] = z_cam
        T[:3, 3] = origin
        return T

    def _new_camera_instance() -> TrimeshCamera:
        return TrimeshCamera(
            resolution=(640, 480), fov=(float(fov_deg_xy[0]), float(fov_deg_xy[1]))
        )

    # FOV tangents
    fovx_deg, fovy_deg = float(fov_deg_xy[0]), float(fov_deg_xy[1])
    tx = np.tan(np.deg2rad(fovx_deg) * 0.5)
    ty = np.tan(np.deg2rad(fovy_deg) * 0.5)
    margin = 1.05

    alpha = np.deg2rad(float(cone.cone_half_angle_deg))
    cos_alpha = float(np.cos(alpha))

    # Directional cone-overlap helper: given two forward unit vectors f1 and f2,
    # and a camera half-FOV (use the larger of hfov/2, vfov/2), determine if the
    # viewing cones overlap significantly (Stage 1 directional test).
    half_fov_rad = max(np.deg2rad(fovx_deg) * 0.5, np.deg2rad(fovy_deg) * 0.5)
    overlap_margin_rad = np.deg2rad(
        float(getattr(cone, "cone_overlap_margin_deg", 5.0))
    )

    def cones_overlap_dir(f1: np.ndarray, f2: np.ndarray) -> bool:
        # Angle between forward vectors
        c = float(np.clip(np.dot(f1, f2), -1.0, 1.0))
        # If theta <= half_fov1 + half_fov2 + margin, cones overlap
        # Since both use same intrinsics here, this becomes theta <= 2*half_fov + margin
        # Compare using cosine to avoid acos when possible
        theta_thresh = 2.0 * half_fov_rad + overlap_margin_rad
        # For small counts this acos is fine and clearer
        theta = float(np.arccos(c))
        return theta <= theta_thresh

    def fibonacci_cap(k: int, K: int) -> tuple[float, float]:
        phi_g = (np.sqrt(5.0) - 1.0) / 2.0
        u = (k + 0.5) / K
        v = (k * phi_g) % 1.0
        cos_theta = (1.0 - u) + u * cos_alpha
        theta = float(np.arccos(cos_theta))
        phi = float(2.0 * np.pi * v)
        return theta, phi

    poses: list[CameraPose] = []

    # Resolve height constraints once
    allowed_min_h = (
        float(min_camera_height_override)
        if min_camera_height_override is not None
        else float(min_camera_height)
    )
    allowed_max_h = (
        float(max_camera_height_override)
        if max_camera_height_override is not None
        else float(max_camera_height)
    )

    for node_name in scene.graph.nodes_geometry:
        _, gkey = scene.graph[node_name]
        geom = scene.geometry[gkey]
        if not isinstance(geom, trimesh.Trimesh) or geom.faces.shape[0] == 0:
            continue

        T_node, _ = scene.graph.get(
            frame_to=node_name, frame_from=scene.graph.base_frame
        )
        R = T_node[:3, :3]
        t = T_node[:3, 3]

        # Mean normal in world
        face_normals = geom.face_normals
        if hasattr(geom, "area_faces") and geom.area_faces is not None:
            w = geom.area_faces.reshape(-1, 1)
            n_local = (face_normals * w).sum(axis=0)
        else:
            n_local = face_normals.mean(axis=0)
        n_world = _safe_normalize(R @ _safe_normalize(n_local))
        if np.linalg.norm(n_world) < 1e-12:
            continue

        c_local = geom.center_mass if geom.is_volume else geom.centroid
        c_world = (R @ c_local) + t

        corners_local = _aabb_corners(geom.bounds)
        corners_world = (R @ corners_local.T).T + t

        # Tangent basis around normal
        up = np.array([0.0, 0.0, 1.0])
        t1 = np.cross(up, n_world)
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(np.array([0.0, 1.0, 0.0]), n_world)
        t1 = _safe_normalize(t1)
        t2 = _safe_normalize(np.cross(n_world, t1))

        K = max(1, int(cone.samples_per_cone))
        # Keep accepted forward directions for this object to reject overlapping cones
        accepted_forward_dirs: list[np.ndarray] = []
        RS = max(1, int(cone.roll_samples))
        for k in range(K):
            theta, phi = fibonacci_cap(k, K)
            d = (
                (np.sin(theta) * np.cos(phi)) * t1
                + (np.sin(theta) * np.sin(phi)) * t2
                + (np.cos(theta)) * n_world
            )
            d = _safe_normalize(d)

            # Fit distance selection
            if str(cone.fit_method).lower() in {
                "footprint",
                "2d",
                "plane",
                "footprint_2d",
            }:
                fit_d = compute_fit_distance_footprint_2d(
                    corners_world=corners_world,
                    centroid_world=c_world,
                    view_normal_world=d,
                    tx=tx,
                    ty=ty,
                    margin=margin,
                )
            else:
                radius_world = float(
                    np.max(np.linalg.norm(corners_world - c_world, axis=1))
                )
                fit_d = compute_fit_distance_spherical_bound(
                    radius_world=radius_world, tx=tx, ty=ty, margin=margin
                )

            standoff = max(float(min_standoff), float(fit_d))
            standoff = min(standoff, float(max_distance))

            p_cam = c_world + standoff * d
            # Early height filter: skip pose generation if camera Z not in range
            if not _is_height_allowed(float(p_cam[2]), allowed_min_h, allowed_max_h):
                continue
            T_cam = look_at_transform(origin=p_cam, target=c_world)

            # Directional cone overlap rejection (per object)
            if getattr(cone, "reject_overlapping_cones", True):
                # Forward vector is along -Z of camera pose
                fwd = -T_cam[:3, 2]
                fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
                # If overlaps with any accepted direction, reject this sample
                overlapped = any(
                    cones_overlap_dir(fwd, a) for a in accepted_forward_dirs
                )
                if overlapped:
                    continue
                accepted_forward_dirs.append(fwd)

            for r in range(RS):
                T_pose = T_cam.copy()
                if RS > 1:
                    psi = 2.0 * np.pi * (r / float(RS))
                    Rz = trimesh.transformations.rotation_matrix(
                        psi, direction=[0, 0, -1]
                    )
                    T_pose = T_pose @ Rz

                cam = _new_camera_instance()
                cam.z_near = 0.1
                cam.z_far = float(frustum_culling_max_distance)

                poses.append(
                    CameraPose(
                        camera=cam,
                        transform=T_pose,
                        node_name=node_name,
                        geometry_key=gkey,
                    )
                )

                if visualize:
                    path = trimesh.load_path(np.vstack([c_world, p_cam]))
                    scene.add_geometry(
                        path,
                        node_name=f"cone_line__{node_name}__{gkey}__{k}_{r}",
                        parent_node_name=scene.graph.base_frame,
                        transform=np.eye(4),
                    )
                    marker = trimesh.creation.camera_marker(
                        cam, marker_height=marker_height
                    )
                    scene.add_geometry(
                        marker,
                        node_name=f"cone_cam__{node_name}__{gkey}__{k}_{r}",
                        parent_node_name=scene.graph.base_frame,
                        transform=T_pose,
                    )

    return poses


# Generate additional cone-based viewpoints and merge them with the existing poses
cone_cfg = ConeSamplingConfig(
    cone_half_angle_deg=75,
    samples_per_cone=20,
    roll_samples=1,
    fit_method="footprint_2d",
)

_extra_cone_poses = generate_cone_view_poses(
    scene,
    cone=cone_cfg,
    min_standoff=min_standoff_distance,
    max_distance=max_view_distance,
    frustum_culling_max_distance=frustum_culling_max_distance,
    fov_deg_xy=(float(test_fov[0]), float(test_fov[1])),
    visualize=True,
    marker_height=0.15,
)
generated_camera_poses.extend(_extra_cone_poses)

# After defining frustum utilities, place spheres colored by visibility counts
# Enable occlusion checking so colors reflect actually visible bodies, not just frustum inclusion

add_camera_origin_spheres(
    scene,
    generated_camera_poses,
    radius=0.08,
    occlusion_check=True,
    samples_per_mesh=64,
    visibility_threshold=0.8,
)

number_of_bodies = 4  # <-- group size you want


def make_opencv_palette(n: int):
    """
    Generate n distinct-ish colors using an OpenCV colormap.
    Returns a list of Color(r, g, b) with r,g,b in [0,1].
    """
    if n <= 0:
        return []

    # values in [0, 255]
    values = np.linspace(0, 255, n, dtype=np.uint8)
    # shape (n,1) so applyColorMap gives (n,1,3)
    gray = values.reshape(-1, 1)

    # pick any nice colormap you like:
    #   COLORMAP_TURBO (if available), COLORMAP_VIRIDIS, COLORMAP_JET, etc.
    cmap = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

    palette = []
    for i in range(n):
        b, g, r = cmap[i, 0]  # OpenCV is BGR, uint8
        palette.append(Color(r / 255.0, g / 255.0, b / 255.0))
    return palette


# --- collect bodies once ---
bodies = list(world.bodies_with_enabled_collision)

# Optional: store original state so we can reset between groups.
# You’ll need to adapt this to your actual API.
original_state = {}
for body in bodies:
    visuals = body.collision[0].mesh.visual.copy()
    # Replace 'current_color' with whatever your API uses.
    # If you don't have a color property, you might skip this
    # and just "overwrite" each iteration instead.
    original_state[body.id] = {
        "mesh_visuals": visuals,
        # "color": getattr(coll, "color", None),   # example, if it exists
        # store anything else you need to restore
    }


def reset_scene_visuals():
    """
    Reset bodies to their original look.
    You must fill in the restore logic for your engine.
    """
    for body in bodies:
        visuals = original_state[body.id]["mesh_visuals"]
        body.collision[0].mesh.visual = visuals


rotate = trimesh.transformations.rotation_matrix(
    angle=np.radians(-90.0), direction=[0, 1, 0]
)
rotate_x = trimesh.transformations.rotation_matrix(
    angle=np.radians(180.0), direction=[1, 0, 0]
)

camera_poses = []

camera_pose1 = TransformationMatrix.from_xyz_rpy(
    x=-3, y=0, z=2.5, roll=-np.pi / 2, pitch=np.pi / 4, yaw=0
).to_np()

camera_poses.append(camera_pose1 @ rotate_x @ rotate)

# camera_pose2 = TransformationMatrix.from_xyz_rpy(
#     x=3, y=0, z=2.5, roll=-np.pi / 2, pitch=np.pi / 4, yaw=np.pi
# ).to_np()
#
# camera_poses.append(camera_pose2 @ rotate_x @ rotate)
#
# camera_pose3 = TransformationMatrix.from_xyz_rpy(
#     x=0, y=-3.5, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=np.pi / 2
# ).to_np()
#
# camera_poses.append(camera_pose3 @ rotate_x @ rotate)
#
# camera_pose4 = TransformationMatrix.from_xyz_rpy(
#     x=0, y=3.5, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=-np.pi / 2
# ).to_np()
#
# camera_poses.append(camera_pose4 @ rotate_x @ rotate)

output_path = Path("../resources/warsaw_data/scene_images/")

if not output_path.exists():
    output_path.mkdir(parents=True)

scene.camera.fov = [60, 45]  # horizontal, vertical degrees

for j, pose in enumerate(camera_poses):

    scene.graph[scene.camera.name] = pose
    scene.show()

    # png = scene.save_image(resolution=(1024, 768), visible=True)

    # with open(os.path.join(output_path, f"original_render_{j}.png"), "wb") as f:
    #     f.write(png)
#
# # --- iterate over groups of bodies ---
# for i, start in enumerate(range(0, len(bodies), number_of_bodies)):
#     group = bodies[start : start + number_of_bodies]
#
#     # reset everything to default look
#     reset_scene_visuals()
#
#     # create palette for this group
#     palette = make_opencv_palette(len(group))
#
#     # apply colors only to the current group; others keep texture
#     for body, color in zip(group, palette):
#         body.collision[0].override_mesh_with_color(color)
#
#     # (Re)build ray tracer / scene if needed
#     rt = RayTracer(world=world)
#     scene = rt.scene
#
#     scene.camera.fov = [60, 45]  # horizontal, vertical degrees
#
#     for j, pose in enumerate(camera_poses):
#
#         scene.graph[scene.camera.name] = pose
#
#         png = scene.save_image(resolution=(1024, 768), visible=True)
#
#         with open(os.path.join(output_path, f"group_{i}_render_{j}.png"), "wb") as f:
#             f.write(png)
