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

s = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
# Set color (RGBA)
s.visual.vertex_colors = trimesh.visual.color.to_rgba([255, 80, 80, 255])

T = TransformationMatrix.from_xyz_rpy(x=1.0, y=0.5, z=0.8).to_np()
scene.add_geometry(
    s, node_name="debug_sphere_temp", parent_node_name="world", transform=T
)


import numpy as np
import trimesh
from trimesh.scene.cameras import Camera


# --- helpers ---
def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def look_at_transform(
    origin: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None
) -> np.ndarray:
    """
    Build a camera-to-world transform where the camera is at `origin`,
    looking at `target`. The camera looks along -Z, Y is up, X is right.
    """
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])

    f = _safe_normalize(target - origin)  # forward (world)
    z_cam = -f  # camera -Z points along forward

    # Build X from up x Z; if degenerate, pick another up
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


# --- parameters ---
line_color = np.array([255, 32, 32, 255], dtype=np.uint8)
marker_height = 0.15  # meters (scale of the camera frustum)
# Create a reusable camera model for markers (FOV arbitrary but reasonable)
marker_cam = Camera(resolution=(640, 480), fov=(60.0, 45.0))

# --- iterate meshes, draw line and marker ---
for node_name in scene.graph.nodes_geometry:
    # node tuple is (parent_node_name, geometry_key)
    _, gkey = scene.graph[node_name]
    geom = scene.geometry[gkey]
    if not isinstance(geom, trimesh.Trimesh):
        continue

    # Transform from world (base) to this node
    T_node, _ = scene.graph.get(frame_to=node_name, frame_from=scene.graph.base_frame)
    R = T_node[:3, :3]
    t = T_node[:3, 3]

    # Compute area-weighted mean normal in local frame
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

    # Choose centroid: center of mass for volumes, centroid for surfaces
    c_local = geom.center_mass if geom.is_volume else geom.centroid

    # Map to world
    c_world = (R @ c_local) + t
    n_world = _safe_normalize(R @ n_local)
    if np.linalg.norm(n_world) < 1e-12:
        continue

    # 1 meter segment
    p0 = c_world
    p1 = c_world + 1.0 * n_world

    # 1) Add the red normal line in world frame
    path = trimesh.load_path(np.vstack([p0, p1]))
    path.colors = np.tile(line_color, (len(path.entities), 1))
    line_node_name = f"normal_line__{node_name}__{gkey}"
    scene.add_geometry(
        path,
        node_name=line_node_name,
        parent_node_name=scene.graph.base_frame,
        transform=np.eye(4),
    )

    # 2) Add a camera marker at p1, oriented to look toward p0
    T_cam = look_at_transform(origin=p1, target=p0)
    cam_marker = trimesh.creation.camera_marker(marker_cam, marker_height=marker_height)
    cam_node_name = f"camera_marker__{node_name}__{gkey}"
    scene.add_geometry(
        cam_marker,
        node_name=cam_node_name,
        parent_node_name=scene.graph.base_frame,
        transform=T_cam,
    )


import numpy as np

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
