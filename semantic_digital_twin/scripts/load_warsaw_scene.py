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

from dataclasses import dataclass
from typing import Optional
import numpy as np
import trimesh
from trimesh.scene.cameras import Camera


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
# Debug reporting: when enabled, print which cameras see which objects, and
# per object, which cameras can see it.
debug_visibility_reports = True
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

    def is_similar_to(
        self,
        other: "CameraPose",
        pos_threshold_m: float = 0.20,
        ang_threshold_deg: float = 10.0,
    ) -> bool:
        """
        Return ``True`` if this pose is too similar to ``other``.

        Similarity requires both:
        - the Euclidean distance between camera origins is below ``pos_threshold_m``;
        - the angular distance between viewing directions is below ``ang_threshold_deg``.

        Viewing direction follows the convention used in this module: cameras look
        along their local negative Z axis, so the world forward is ``-R[:, 2]`` of
        the camera-to-world transform. The angular distance is computed with
        ``atan2(||u×v||, u·v)`` for numerical stability.
        """

        return are_camera_poses_too_similar(
            self,
            other,
            pos_threshold_m=pos_threshold_m,
            ang_threshold_deg=ang_threshold_deg,
        )


def _safe_normalize_dir(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector; return the input if its norm is near zero.
    """

    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _camera_forward_world(T_cam_world: np.ndarray) -> np.ndarray:
    """
    Compute the camera's viewing direction in world coordinates.

    The camera looks along its local negative Z axis.
    """

    return _safe_normalize_dir(-T_cam_world[:3, 2])


def _angle_between_dirs(u: np.ndarray, v: np.ndarray) -> float:
    """
    Robust unsigned angle (radians) between two direction vectors.

    Uses ``atan2(||u×v||, u·v)`` to avoid precision loss near 0 and π.
    """

    u_n = _safe_normalize_dir(u)
    v_n = _safe_normalize_dir(v)
    cross_mag = float(np.linalg.norm(np.cross(u_n, v_n)))
    dot_uv = float(np.clip(np.dot(u_n, v_n), -1.0, 1.0))
    return float(np.arctan2(cross_mag, dot_uv))


def are_camera_poses_too_similar(
    a: CameraPose,
    b: CameraPose,
    *,
    pos_threshold_m: float = 0.20,
    ang_threshold_deg: float = 10.0,
) -> bool:
    """
    Check whether two camera poses are too similar.

    Two poses are considered similar if both their positional distance and their
    angular distance (between viewing directions) are below the given thresholds.
    The angle is computed using ``atan2`` on cross/dot of the forward vectors to
    ensure numerical stability. Thresholds are configurable; defaults are 0.20 m
    and 10 degrees.
    """

    pa = a.transform[:3, 3]
    pb = b.transform[:3, 3]
    pos_dist = float(np.linalg.norm(pa - pb))

    fa = _camera_forward_world(a.transform)
    fb = _camera_forward_world(b.transform)
    ang_rad = _angle_between_dirs(fa, fb)
    ang_deg = float(np.rad2deg(ang_rad))

    return (pos_dist < float(pos_threshold_m)) and (ang_deg < float(ang_threshold_deg))


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
    reject_overlapping_cones: bool = False  # True
    # Small angular margin added to the cone-overlap test (degrees)
    cone_overlap_margin_deg: float = 5.0


# @dataclass
# class Timer:
#     """
#     Simple context manager to measure elapsed wall time.
#
#     Prints a line when exiting the context with the elapsed time in milliseconds.
#     """
#
#     label: str
#
#     def __enter__(self):
#         self._t0 = time.perf_counter()
#         return self
#
#     def __exit__(self, exc_type, exc, tb):
#         dt_ms = (time.perf_counter() - self._t0) * 1000.0
#         print(f"[TIMING] {self.label}: {dt_ms:.2f} ms")


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
    """
    Create visualization lines along mean normals and place cameras that look at
    object centroids from an appropriate standoff distance.

    The function delegates the work to a focused object that encapsulates all
    steps: configuration resolution, geometry analysis, standoff computation,
    scene updates, and pose collection.
    """

    @dataclass
    class MeanNormalCameraConfig:
        """
        Configuration for camera placement around scene geometries.
        """

        marker_height: float
        min_standoff: float
        max_distance: Optional[float]
        fov_deg_xy: tuple[float, float]
        fit_method: str
        min_height: float
        max_height: float
        margin_factor: float
        line_color: np.ndarray

        def fov_tangents(self) -> tuple[float, float]:
            fx, fy = float(self.fov_deg_xy[0]), float(self.fov_deg_xy[1])
            return float(np.tan(np.deg2rad(fx) * 0.5)), float(
                np.tan(np.deg2rad(fy) * 0.5)
            )

    class MeanNormalCameraPlacer:
        """
        Analyze each mesh, compute mean normal and place a camera facing the mesh.
        """

        def __init__(self, config: MeanNormalCameraConfig):
            self.config = config

        @staticmethod
        def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            n = float(np.linalg.norm(v))
            if n < eps:
                return v
            return v / n

        @staticmethod
        def _look_at_transform(
            origin: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None
        ) -> np.ndarray:
            if up_hint is None:
                up_hint = np.array([0.0, 0.0, 1.0])
            f = MeanNormalCameraPlacer._safe_normalize(target - origin)
            z_cam = -f
            x_cam = np.cross(up_hint, z_cam)
            if np.linalg.norm(x_cam) < 1e-6:
                up_hint = np.array([0.0, 1.0, 0.0])
                x_cam = np.cross(up_hint, z_cam)
            x_cam = MeanNormalCameraPlacer._safe_normalize(x_cam)
            y_cam = MeanNormalCameraPlacer._safe_normalize(np.cross(z_cam, x_cam))
            T = np.eye(4)
            T[:3, 0] = x_cam
            T[:3, 1] = y_cam
            T[:3, 2] = z_cam
            T[:3, 3] = origin
            return T

        @staticmethod
        def _new_camera_instance() -> Camera:
            return Camera(resolution=(640, 480), fov=test_fov)

        def _compute_weighted_mean_normal(self, geom: trimesh.Trimesh) -> np.ndarray:
            if geom.faces.shape[0] == 0:
                return np.zeros(3, dtype=float)
            face_normals = geom.face_normals
            if hasattr(geom, "area_faces") and geom.area_faces is not None:
                w = geom.area_faces.reshape(-1, 1)
                n_local = (face_normals * w).sum(axis=0)
            else:
                n_local = face_normals.mean(axis=0)
            return self._safe_normalize(n_local)

        def _corners_world(
            self, geom: trimesh.Trimesh, R: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
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
            return (R @ corners_local.T).T + t

        def _compute_standoff(
            self, corners_world: np.ndarray, c_world: np.ndarray, n_world: np.ndarray
        ) -> float:
            tx, ty = self.config.fov_tangents()
            if self.config.fit_method.lower() in {
                "footprint",
                "2d",
                "plane",
                "footprint_2d",
            }:
                fit_distance = compute_fit_distance_footprint_2d(
                    corners_world=corners_world,
                    centroid_world=c_world,
                    view_normal_world=n_world,
                    tx=tx,
                    ty=ty,
                    margin=self.config.margin_factor,
                )
            else:
                radius_world = float(
                    np.max(np.linalg.norm(corners_world - c_world, axis=1))
                )
                fit_distance = compute_fit_distance_spherical_bound(
                    radius_world=radius_world,
                    tx=tx,
                    ty=ty,
                    margin=self.config.margin_factor,
                )

            standoff = max(self.config.min_standoff, fit_distance)
            if self.config.max_distance is not None:
                standoff = min(standoff, self.config.max_distance)
            return float(standoff)

        def process_node(
            self, scene: trimesh.Scene, node_name: str
        ) -> list[CameraPose]:
            poses: list[CameraPose] = []
            _, gkey = scene.graph[node_name]
            geom = scene.geometry[gkey]
            if not isinstance(geom, trimesh.Trimesh):
                return poses

            T_node, _ = scene.graph.get(
                frame_to=node_name, frame_from=scene.graph.base_frame
            )
            R = T_node[:3, :3]
            t = T_node[:3, 3]

            n_local = self._compute_weighted_mean_normal(geom)
            if float(np.linalg.norm(n_local)) < 1e-12:
                return poses

            c_local = geom.center_mass if geom.is_volume else geom.centroid
            c_world = (R @ c_local) + t
            n_world = self._safe_normalize(R @ n_local)
            if float(np.linalg.norm(n_world)) < 1e-12:
                return poses

            corners_world = self._corners_world(geom, R, t)
            standoff_distance = self._compute_standoff(corners_world, c_world, n_world)

            p0 = c_world
            p1 = c_world + standoff_distance * n_world

            if not _is_height_allowed(
                float(p1[2]), self.config.min_height, self.config.max_height
            ):
                return poses

            # Add normal line visualization
            path = trimesh.load_path(np.vstack([p0, p1]))
            path.colors = np.tile(self.config.line_color, (len(path.entities), 1))
            line_node_name = f"normal_line__{node_name}__{gkey}"
            scene.add_geometry(
                path,
                node_name=line_node_name,
                parent_node_name=scene.graph.base_frame,
                transform=np.eye(4),
            )

            # Add camera marker
            T_cam = self._look_at_transform(origin=p1, target=p0)
            pose_cam = self._new_camera_instance()
            cam_marker = trimesh.creation.camera_marker(
                pose_cam, marker_height=self.config.marker_height
            )
            cam_node_name = f"camera_marker__{node_name}__{gkey}"
            scene.add_geometry(
                cam_marker,
                node_name=cam_node_name,
                parent_node_name=scene.graph.base_frame,
                transform=T_cam,
            )

            poses.append(
                CameraPose(
                    camera=pose_cam,
                    transform=T_cam,
                    node_name=node_name,
                    geometry_key=gkey,
                )
            )
            return poses

    # Resolve configuration using module-level defaults
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

    config = MeanNormalCameraConfig(
        marker_height=marker_height,
        min_standoff=float(effective_min_standoff),
        max_distance=(
            float(effective_max_distance)
            if effective_max_distance is not None
            else None
        ),
        fov_deg_xy=(float(fov_xy[0]), float(fov_xy[1])),
        fit_method=str(fit_method),
        min_height=float(allowed_min_h),
        max_height=float(allowed_max_h),
        margin_factor=1.05,
        line_color=np.array([255, 32, 32, 255], dtype=np.uint8),
    )

    placer = MeanNormalCameraPlacer(config)
    all_poses: list[CameraPose] = []
    for node in scene.graph.nodes_geometry:
        all_poses.extend(placer.process_node(scene, node))
    return all_poses


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

    # Delegate to split steps for clarity and reuse.
    cfg = VisibilityComputationConfig(
        occlusion_check=occlusion_check,
        samples_per_mesh=samples_per_mesh,
        visibility_threshold=visibility_threshold,
    )
    counts = compute_visible_bodies_per_pose(
        scene=scene, camera_poses=camera_poses, config=cfg
    )
    CameraOriginSpheresVisualizer().render(
        scene=scene,
        camera_poses=camera_poses,
        counts=counts,
        radius=radius,
        color_rgba=color_rgba,
    )


@dataclass
class VisibilityComputationConfig:
    """
    Options controlling visibility computation of camera poses.

    The occlusion and sampling options are forwarded to the frustum and
    optional visibility checks when evaluating mesh visibility per pose.
    """

    occlusion_check: bool = False
    samples_per_mesh: int = 32
    visibility_threshold: float = 0.8


@timeit("compute_visible_bodies_per_pose")
def compute_visible_bodies_per_pose(
    scene: trimesh.Scene,
    camera_poses: list[CameraPose],
    config: VisibilityComputationConfig | None = None,
) -> list[int]:
    """
    Compute, for each camera pose, how many bodies are fully visible.

    Returns a list of counts aligned with ``camera_poses``. Visibility is
    measured by frustum classification and optional occlusion checks.
    Bodies are derived from scene node names using a `RayTracer` index mapping
    when available; otherwise, geometry node names are deduplicated by the
    prefix before ``"_collision_"``.
    """

    cfg = config or VisibilityComputationConfig()

    def _count_for_pose(pose: CameraPose) -> int:
        try:
            fully_inside, _ = frustum_cull_scene(
                scene,
                pose.camera,
                pose.transform,
                require_full_visibility=True,
                occlusion_check=cfg.occlusion_check,
                samples_per_mesh=cfg.samples_per_mesh,
                visibility_threshold=cfg.visibility_threshold,
            )
        except NameError:
            fully_inside = set()

        # Prefer RayTracer mapping if available
        body_indices: set[object] = set()
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

    return [_count_for_pose(p) for p in camera_poses]


def _value_to_rgba(v: float) -> tuple[int, int, int, int]:
    """
    Convert a normalized value in [0, 1] to an RGBA color.

    Uses a simple two-color gradient: blue (low) to red (high).
    """

    v = float(max(0.0, min(1.0, v)))
    low = np.array([0, 0, 255, 255], dtype=float)  # blue
    high = np.array([255, 0, 0, 255], dtype=float)  # red
    rgba = (1.0 - v) * low + v * high
    return tuple(int(round(c)) for c in rgba)


class CameraOriginSpheresVisualizer:
    """
    Render spheres at camera origins, colored by per-pose counts.

    If a fixed ``color_rgba`` is provided, it overrides data-driven colors.
    """

    def render(
        self,
        scene: trimesh.Scene,
        camera_poses: list[CameraPose],
        counts: list[int],
        *,
        radius: float = 0.05,
        color_rgba: tuple[int, int, int, int] | None = None,
    ) -> None:
        if len(counts) != len(camera_poses):
            raise ValueError("counts must be aligned with camera_poses")

        max_count = max(counts) if counts else 0

        for idx, pose in enumerate(camera_poses):
            if color_rgba is None:
                value = (counts[idx] / max_count) if max_count > 0 else 0.0
                rgba = _value_to_rgba(value)
            else:
                rgba = color_rgba

            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            sphere.visual.vertex_colors = trimesh.visual.color.to_rgba(rgba)

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
# Camera pose scoring and greedy non-maximum suppression (selection)
# -----------------------------------------------------------------------------


@dataclass
class CameraSelectionConfig:
    """
    Configuration for greedy non maximum suppression of camera poses.

    Poses are considered duplicates if both their positional distance and
    angular difference (between viewing directions) are below the thresholds.
    """

    pos_threshold_m: float = 0.20
    ang_threshold_deg: float = 10.0
    max_poses: int | None = None


def score_camera_poses_by_visible_bodies(
    scene: trimesh.Scene,
    camera_poses: list[CameraPose],
    visibility_config: VisibilityComputationConfig | None = None,
) -> list[int]:
    """
    Return a score for each pose: the number of fully visible bodies.

    The computation is delegated to :func:`compute_visible_bodies_per_pose`.
    """

    return compute_visible_bodies_per_pose(scene, camera_poses, visibility_config)


@timeit("compute_visible_body_indices_per_pose")
def compute_visible_body_indices_per_pose(
    scene: trimesh.Scene,
    camera_poses: list[CameraPose],
    config: VisibilityComputationConfig | None = None,
) -> list[set[object]]:
    """
    Compute, for each camera pose, the set of visible body identifiers.

    Bodies are identified using the same mapping semantics as
    :func:`compute_visible_bodies_per_pose`:
    - Prefer :class:`RayTracer`'s ``scene_to_index`` mapping if available.
    - Fallback to deduplicating geometry node names by the prefix before
      ``"_collision_"``.
    """

    cfg = config or VisibilityComputationConfig()

    def _bodies_for_pose(pose: CameraPose) -> set[object]:
        try:
            fully_inside, _ = frustum_cull_scene(
                scene,
                pose.camera,
                pose.transform,
                require_full_visibility=True,
                occlusion_check=cfg.occlusion_check,
                samples_per_mesh=cfg.samples_per_mesh,
                visibility_threshold=cfg.visibility_threshold,
            )
        except NameError:
            fully_inside = set()

        # Prefer RayTracer mapping if available
        mapping = None
        try:
            mapping = rt.scene_to_index  # type: ignore[name-defined]
        except Exception:
            mapping = None

        body_indices: set[object] = set()
        if mapping is not None:
            for node in fully_inside:
                if node in mapping:
                    body_indices.add(mapping[node])
        else:
            for node in fully_inside:
                if "_collision_" in node:
                    body_indices.add(node.split("_collision_")[0])

        return body_indices

    return [_bodies_for_pose(p) for p in camera_poses]


def greedy_select_by_score_similarity_and_novelty(
    camera_poses: list[CameraPose],
    scores: list[int],
    body_sets: list[set[object]],
    cfg: CameraSelectionConfig | None = None,
) -> list[CameraPose]:
    """
    Greedily select camera poses by score, similarity suppression, and novelty.

    The algorithm maintains a set of bodies already covered by selected poses
    and only accepts a candidate if it adds at least one previously unseen body
    and is not too similar to any already selected pose according to ``cfg``.
    """

    if not (len(camera_poses) == len(scores) == len(body_sets)):
        raise ValueError("camera_poses, scores, and body_sets must be aligned")

    cfg = cfg or CameraSelectionConfig()

    order = list(range(len(camera_poses)))
    order.sort(key=lambda i: int(scores[i]), reverse=True)

    selected: list[CameraPose] = []
    covered: set[object] = set()

    for i in order:
        cand = camera_poses[i]

        # Similarity suppression
        is_similar = False
        for sel in selected:
            if sel.is_similar_to(
                cand,
                pos_threshold_m=cfg.pos_threshold_m,
                ang_threshold_deg=cfg.ang_threshold_deg,
            ):
                is_similar = True
                break
        if is_similar:
            continue

        # Novelty requirement: must add at least one new body
        new_bodies = body_sets[i] - covered
        if len(new_bodies) == 0:
            continue

        selected.append(cand)
        covered.update(body_sets[i])

        if cfg.max_poses is not None and len(selected) >= int(cfg.max_poses):
            break

    return selected


# -----------------------------------------------------------------------------
# Debug visibility reports (camera→objects and object→cameras)
# -----------------------------------------------------------------------------


@dataclass
class DebugVisibilityReportConfig:
    """
    Configuration for printing debug visibility reports.

    When enabled, the script prints two reports:
    - for each camera pose, the list of objects it can fully see;
    - for each object, the list of camera poses that can see it.
    """

    enabled: bool = False
    show_camera_to_objects: bool = True
    show_object_to_cameras: bool = True


def _make_body_label_resolver() -> dict[object, str]:
    """
    Build a mapping from internal body identifiers to human‑readable labels.

    This prefers the :class:`RayTracer`'s ``scene_to_index`` mapping when
    available by constructing an index→name map using the prefix before
    ``"_collision_"``. If absent, an empty dict is returned and callers should
    treat identifiers as display strings directly.
    """

    try:
        # rt is expected to be a module‑level RayTracer instance if created
        mapping = rt.scene_to_index  # type: ignore[name-defined]
    except Exception:
        mapping = None

    id_to_label: dict[object, str] = {}
    if mapping is None:
        return id_to_label

    # mapping: node_name(str) -> index(object)
    # Build reverse by choosing a stable, readable base name per index
    for node_name, idx in mapping.items():
        base = (
            node_name.split("_collision_")[0]
            if "_collision_" in node_name
            else node_name
        )
        # only set first time to keep deterministic, earlier items take precedence
        if idx not in id_to_label:
            id_to_label[idx] = base
    return id_to_label


def _camera_label(pose: CameraPose, idx: int) -> str:
    """
    Create a concise camera identifier for debug output.

    The label includes the index and the originating node/geometry keys.
    """

    return f"{idx}:{pose.node_name}/{pose.geometry_key}"


def print_camera_to_bodies_report(
    camera_poses: list[CameraPose],
    body_sets: list[set[object]],
) -> None:
    """
    Print a report listing, for each camera, the objects it can fully see.
    """

    if len(camera_poses) != len(body_sets):
        raise ValueError("body_sets must be aligned with camera_poses")

    id_to_label = _make_body_label_resolver()

    print("[DEBUG] Camera → Objects visibility report:")
    for i, pose in enumerate(camera_poses):
        cam_label = _camera_label(pose, i)
        # Map identifiers to strings
        names = []
        for bid in body_sets[i]:
            if bid in id_to_label:
                names.append(id_to_label[bid])
            else:
                # bid may already be a string base name
                names.append(str(bid))
        names_sorted = sorted(set(names))
        print(
            f"  - {cam_label}: {', '.join(names_sorted) if names_sorted else '(none)'}"
        )


def print_body_to_cameras_report(
    camera_poses: list[CameraPose],
    body_sets: list[set[object]],
) -> None:
    """
    Print a report listing, for each object, which cameras can fully see it.
    """

    if len(camera_poses) != len(body_sets):
        raise ValueError("body_sets must be aligned with camera_poses")

    id_to_label = _make_body_label_resolver()

    body_to_cams: dict[str, list[str]] = {}
    for i, pose in enumerate(camera_poses):
        cam_label = _camera_label(pose, i)
        for bid in body_sets[i]:
            name = id_to_label.get(bid, str(bid))
            body_to_cams.setdefault(name, []).append(cam_label)

    print("[DEBUG] Object → Cameras visibility report:")
    for body_name in sorted(body_to_cams.keys()):
        cams_sorted = sorted(body_to_cams[body_name])
        print(f"  - {body_name}: {', '.join(cams_sorted) if cams_sorted else '(none)'}")


def _all_body_identifiers(scene: trimesh.Scene) -> set[object]:
    """
    Return the set of all body identifiers present in the scene.

    The identifier type matches what is used in visibility body sets:
    - If a :class:`RayTracer` mapping is available, use its indices.
    - Otherwise, collapse geometry node names by the prefix before
      ``"_collision_"`` to represent logical objects.
    """

    # Prefer RayTracer mapping if available
    try:
        mapping = rt.scene_to_index  # type: ignore[name-defined]
    except Exception:
        mapping = None

    if mapping is not None:
        # mapping: node_name -> index (identifier)
        return set(mapping.values())

    # Fallback: collect base names from scene nodes
    all_ids: set[object] = set()
    try:
        node_names = list(scene.graph.nodes_geometry)
    except Exception:
        node_names = []
    for node in node_names:
        if isinstance(node, str) and "_collision_" in node:
            all_ids.add(node.split("_collision_")[0])
    return all_ids


def print_uncovered_bodies_report(
    scene: trimesh.Scene,
    body_sets: list[set[object]],
) -> None:
    """
    Print the list of objects that are not covered by any of the given
    camera body sets.

    This should be called after NMS selection to reveal which scene objects
    remain unseen by all selected poses.
    """

    all_ids = _all_body_identifiers(scene)
    covered: set[object] = set()
    for s in body_sets:
        covered.update(s)

    missing = all_ids - covered

    id_to_label = _make_body_label_resolver()
    names = [id_to_label.get(bid, str(bid)) for bid in missing]
    names_sorted = sorted(set(names))

    print("[DEBUG] Objects not fully covered by any selected camera:")
    if not names_sorted:
        print("  - (all objects are covered)")
    else:
        for name in names_sorted:
            print(f"  - {name}")


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
    Test scene meshes against the camera frustum using camera-space tests.

    The function delegates the workflow to small, focused classes:
    configuration, projection, frustum classification, and an optional
    visibility checker for occlusion validation.

    Returns two sets of node names: fully_inside and partially_inside.
    """
    from dataclasses import dataclass

    # ------------------------------------------------------------------
    # Data and helpers
    # ------------------------------------------------------------------
    @dataclass
    class FrustumCullConfig:
        """Configuration for frustum culling and visibility validation.

        The camera is assumed to look along its negative Z axis.
        """

        fov_deg_xy: tuple[float, float]
        z_near: float
        z_far: float
        require_full_visibility: bool
        eps: float
        occlusion_check: bool
        samples_per_mesh: int
        visibility_threshold: float

        def tangents(self) -> tuple[float, float]:
            fx, fy = float(self.fov_deg_xy[0]), float(self.fov_deg_xy[1])
            return float(np.tan(np.deg2rad(fx) * 0.5)), float(
                np.tan(np.deg2rad(fy) * 0.5)
            )

    class CameraFrustum:
        """Encapsulates FOV and near/far clipping classification in camera space."""

        def __init__(self, cfg: FrustumCullConfig):
            self.cfg = cfg
            self.tx, self.ty = cfg.tangents()

        def classify_corners(
            self, x: np.ndarray, y: np.ndarray, z: np.ndarray
        ) -> tuple[bool, bool]:
            dz = -z  # points in front have positive dz
            inside = (
                (dz >= self.cfg.z_near - self.cfg.eps)
                & (dz <= self.cfg.z_far + self.cfg.eps)
                & (np.abs(x) <= dz * self.tx + self.cfg.eps)
                & (np.abs(y) <= dz * self.ty + self.cfg.eps)
            )
            return bool(np.all(inside)), bool(np.any(inside))

    class NodeProjector:
        """Projects node-local AABB corners to camera space."""

        def __init__(self, scene_: trimesh.Scene, V_world_cam: np.ndarray):
            self.scene = scene_
            self.V = V_world_cam

        def corners_world(
            self, node_name: str, geom: trimesh.Trimesh
        ) -> tuple[np.ndarray, np.ndarray]:
            corners_local = _aabb_corners(geom.bounds)
            T_node_world, _ = self.scene.graph.get(
                frame_to=node_name, frame_from=self.scene.graph.base_frame
            )
            corners_world = (T_node_world[:3, :3] @ corners_local.T).T + T_node_world[
                :3, 3
            ]
            return corners_world, T_node_world

        def project(
            self, points_world: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            N = len(points_world)
            pts_h = np.hstack([points_world, np.ones((N, 1))])
            pc = (self.V @ pts_h.T).T
            return pc[:, 0], pc[:, 1], pc[:, 2]

    class VisibilityChecker:
        """Strategy for validating visibility against occluders."""

        def is_sufficiently_visible(
            self,
            node_name: str,
            geom: trimesh.Trimesh,
            T_node_world: np.ndarray,
            T_cam_world_local: np.ndarray,
        ) -> float:
            """Return ratio of visible samples in [0, 1]."""
            raise NotImplementedError

    class NoOpVisibilityChecker(VisibilityChecker):
        def is_sufficiently_visible(
            self, node_name, geom, T_node_world, T_cam_world_local
        ) -> float:
            return 1.0

    class RayOcclusionVisibilityChecker(VisibilityChecker):
        """Ray-based visibility via `trimesh` ray engine on the merged scene mesh."""

        def __init__(self, scene_: trimesh.Scene, eps_local: float, samples: int):
            self.scene = scene_
            self.eps = float(eps_local)
            # Build once
            self.ray_engine = self.scene.to_mesh().ray
            self.samples = int(max(8, samples))

        def _samples_world(
            self, geom: trimesh.Trimesh, T_node_world: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            try:
                pts_local = geom.sample(self.samples)
                if pts_local.shape[0] == 0:
                    aabb = _aabb_corners(geom.bounds)
                    centroid = geom.centroid.reshape(1, 3)
                    pts_local = np.vstack([aabb, centroid])
            except Exception:
                aabb = _aabb_corners(geom.bounds)
                centroid = geom.centroid.reshape(1, 3)
                pts_local = np.vstack([aabb, centroid])
            pts_world = (T_node_world[:3, :3] @ pts_local.T).T + T_node_world[:3, 3]
            return pts_world, pts_local

        def is_sufficiently_visible(
            self, node_name, geom, T_node_world, T_cam_world_local
        ) -> float:
            origin = T_cam_world_local[:3, 3]
            pts_world, _ = self._samples_world(geom, T_node_world)
            dirs = pts_world - origin
            dists = np.linalg.norm(dirs, axis=1)
            valid = dists > self.eps
            if not np.any(valid):
                return 0.0
            dirs = dirs[valid] / dists[valid][:, None]
            dists = dists[valid]
            try:
                loc, idx_ray, _ = self.ray_engine.intersects_location(
                    ray_origins=np.repeat(origin[None, :], len(dirs), axis=0),
                    ray_directions=dirs,
                    multiple_hits=False,
                )
            except Exception:
                return 0.0
            if len(idx_ray) == 0:
                return 0.0
            hit_d = np.linalg.norm(loc - origin, axis=1)
            target_d = dists[idx_ray]
            visible = float(np.sum(np.abs(hit_d - target_d) <= 1e-3))
            return visible / float(max(1, len(dirs)))

    class FrustumCuller:
        """Iterate scene nodes, classify AABBs, and validate visibility."""

        def __init__(
            self,
            scene_: trimesh.Scene,
            cfg: FrustumCullConfig,
            V_world_cam: np.ndarray,
            T_cam_world_local: np.ndarray,
        ):
            self.scene = scene_
            self.cfg = cfg
            self.frustum = CameraFrustum(cfg)
            self.proj = NodeProjector(scene_, V_world_cam)
            if cfg.occlusion_check:
                self.visibility = RayOcclusionVisibilityChecker(
                    scene_, cfg.eps, cfg.samples_per_mesh
                )
            else:
                self.visibility = NoOpVisibilityChecker()
            self.T_cam_world = T_cam_world_local

        def run(self) -> tuple[set[str], set[str]]:
            fully: set[str] = set()
            partially: set[str] = set()
            for node_name in self.scene.graph.nodes_geometry:
                _, gkey = self.scene.graph[node_name]
                geom = self.scene.geometry[gkey]
                if not isinstance(geom, trimesh.Trimesh):
                    continue

                corners_world, T_node_world = self.proj.corners_world(node_name, geom)
                x, y, z = self.proj.project(corners_world)
                all_inside, any_inside = self.frustum.classify_corners(x, y, z)

                candidate_full = self.cfg.require_full_visibility and all_inside
                candidate_partial = (
                    not self.cfg.require_full_visibility
                ) and any_inside
                if not (candidate_full or candidate_partial):
                    continue

                vis_ratio = self.visibility.is_sufficiently_visible(
                    node_name=node_name,
                    geom=geom,
                    T_node_world=T_node_world,
                    T_cam_world_local=self.T_cam_world,
                )

                if self.cfg.require_full_visibility:
                    if candidate_full and (vis_ratio >= self.cfg.visibility_threshold):
                        fully.add(node_name)
                else:
                    # For partial, require that at least one sample is visible
                    if candidate_partial and (vis_ratio > 0.0):
                        partially.add(node_name)

            return fully, partially

    # ------------------------------------------------------------------
    # Assemble and execute
    # ------------------------------------------------------------------
    V = camera_view_matrix(T_cam_world)
    cfg = FrustumCullConfig(
        fov_deg_xy=(float(camera.fov[0]), float(camera.fov[1])),
        z_near=float(camera.z_near),
        z_far=float(camera.z_far),
        require_full_visibility=bool(require_full_visibility),
        eps=float(eps),
        occlusion_check=bool(occlusion_check),
        samples_per_mesh=int(samples_per_mesh),
        visibility_threshold=float(visibility_threshold),
    )
    culler = FrustumCuller(scene, cfg, V, T_cam_world)
    return culler.run()


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
    reject_overlapping_cones=False,
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

# After defining frustum utilities, select promising camera poses using a
# greedy non-maximum suppression based on visible-body scores and the
# similarity metric defined earlier. Then visualize the selected set.

# 1) score poses (number of fully visible bodies)
_visibility_cfg = VisibilityComputationConfig(
    occlusion_check=True, samples_per_mesh=64, visibility_threshold=0.8
)
_scores = score_camera_poses_by_visible_bodies(
    scene=scene, camera_poses=generated_camera_poses, visibility_config=_visibility_cfg
)

# 2) greedy selection with similarity suppression and coverage novelty
#    Only accept a camera if it adds at least one body not yet seen by the
#    already selected set, and is not too similar to any selected pose.
_selection_cfg = CameraSelectionConfig(
    pos_threshold_m=0.50, ang_threshold_deg=20.0, max_poses=None
)
_body_sets = compute_visible_body_indices_per_pose(
    scene=scene, camera_poses=generated_camera_poses, config=_visibility_cfg
)

# Optional debug reports for visibility relationships before selection
if debug_visibility_reports:
    print("[DEBUG] ==== Visibility reports BEFORE selection ====")
    print_camera_to_bodies_report(
        camera_poses=generated_camera_poses, body_sets=_body_sets
    )
    print_body_to_cameras_report(
        camera_poses=generated_camera_poses, body_sets=_body_sets
    )
_selected_camera_poses = greedy_select_by_score_similarity_and_novelty(
    camera_poses=generated_camera_poses,
    scores=_scores,
    body_sets=_body_sets,
    cfg=_selection_cfg,
)

# Optional debug reports for visibility relationships after selection
if debug_visibility_reports:
    _selected_body_sets = compute_visible_body_indices_per_pose(
        scene=scene, camera_poses=_selected_camera_poses, config=_visibility_cfg
    )
    print("[DEBUG] ==== Visibility reports AFTER selection ====")
    print_camera_to_bodies_report(
        camera_poses=_selected_camera_poses, body_sets=_selected_body_sets
    )
    print_body_to_cameras_report(
        camera_poses=_selected_camera_poses, body_sets=_selected_body_sets
    )
    # Additionally report which objects are not covered by any selected camera
    print_uncovered_bodies_report(scene=scene, body_sets=_selected_body_sets)

# 3) visualize only the selected camera origins (colors reflect visibility counts)
add_camera_origin_spheres(
    scene,
    _selected_camera_poses,
    radius=0.08,
    occlusion_check=_visibility_cfg.occlusion_check,
    samples_per_mesh=_visibility_cfg.samples_per_mesh,
    visibility_threshold=_visibility_cfg.visibility_threshold,
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
