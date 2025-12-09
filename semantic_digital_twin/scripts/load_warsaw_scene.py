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

dir_path = "/home/itsme/work/cram_ws/src/cognitive_robot_abstract_machine/semantic_digital_twin/resources/warsaw_data/objects/"
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

camera_pose2 = TransformationMatrix.from_xyz_rpy(
    x=3, y=0, z=2.5, roll=-np.pi / 2, pitch=np.pi / 4, yaw=np.pi
).to_np()

camera_poses.append(camera_pose2 @ rotate_x @ rotate)

camera_pose3 = TransformationMatrix.from_xyz_rpy(
    x=0, y=-3.5, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=np.pi / 2
).to_np()

camera_poses.append(camera_pose3 @ rotate_x @ rotate)

camera_pose4 = TransformationMatrix.from_xyz_rpy(
    x=0, y=3.5, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=-np.pi / 2
).to_np()

camera_poses.append(camera_pose4 @ rotate_x @ rotate)

output_path = Path("../resources/warsaw_data/scene_images/")

if not output_path.exists():
    output_path.mkdir(parents=True)

scene.camera.fov = [60, 45]  # horizontal, vertical degrees

for j, pose in enumerate(camera_poses):

    scene.graph[scene.camera.name] = pose

    png = scene.save_image(resolution=(1024, 768), visible=True)

    with open(os.path.join(output_path, f"original_render_{j}.png"), "wb") as f:
        f.write(png)

# --- iterate over groups of bodies ---
for i, start in enumerate(range(0, len(bodies), number_of_bodies)):
    group = bodies[start : start + number_of_bodies]

    # reset everything to default look
    reset_scene_visuals()

    # create palette for this group
    palette = make_opencv_palette(len(group))

    # apply colors only to the current group; others keep texture
    for body, color in zip(group, palette):
        body.collision[0].override_mesh_with_color(color)

    # (Re)build ray tracer / scene if needed
    rt = RayTracer(world=world)
    scene = rt.scene

    scene.camera.fov = [60, 45]  # horizontal, vertical degrees

    for j, pose in enumerate(camera_poses):

        scene.graph[scene.camera.name] = pose

        png = scene.save_image(resolution=(1024, 768), visible=True)

        with open(os.path.join(output_path, f"group_{i}_render_{j}.png"), "wb") as f:
            f.write(png)
