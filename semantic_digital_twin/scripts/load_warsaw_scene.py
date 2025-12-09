import os

import numpy as np
import trimesh
from trimesh.collision import CollisionManager

from semantic_digital_twin.adapters.mesh import OBJParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Pipeline,
    TransformGeometry,
    CenterLocalGeometryAndPreserveWorldPose,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

dir_path = "/home/itsme/work/cram_ws/src/cognitive_robot_abstract_machine/semantic_digital_twin/resources/warsaw_data/objects/"
files = [f for f in os.listdir(dir_path) if f.endswith(".obj")]

world = World()
root = Body(name=PrefixedName("root_body"))
with world.modify_world():
    world.add_body(root)
for i, file in enumerate(files):
    obj_world = OBJParser(os.path.join(dir_path, file)).parse()
    with world.modify_world():
        # color = index_to_color[i % 4]
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

rt = RayTracer(world=world)
scene = rt.scene


world_meshes = []  # list of (id, mesh_in_world)

for body in world.bodies_with_enabled_collision:
    transform = body.global_pose.to_np()
    body_id = body.id
    geom_hash = body.collision[0].mesh.identifier_hash
    geom_name = scene.geometry_identifiers[geom_hash]

    mesh = scene.geometry[geom_name].copy()
    mesh.apply_transform(transform)
    world_meshes.append((body_id, mesh))

ids = [id_ for id_, _ in world_meshes]
bounds = {id_: mesh.bounds for id_, mesh in world_meshes}


def boxes_touch(b1, b2, tol=0.0):
    (min1, max1), (min2, max2) = b1, b2
    return np.all(max1 + tol >= min2) and np.all(max2 + tol >= min1)


adj = {id_: set() for id_ in ids}

for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        n1, n2 = ids[i], ids[j]
        if boxes_touch(bounds[n1], bounds[n2], tol=1e-6):
            adj[n1].add(n2)
            adj[n2].add(n1)

# Choose your 4 RGBA colors
color_palette = [
    Color(1, 0, 0),
    Color(0, 1, 0),
    Color(0, 0, 1),
    Color(1, 1, 0),
]

colors_by_id = {}


# Sort nodes by degree (high-degree first = better greedy result)
nodes_sorted = sorted(adj.keys(), key=lambda n: -len(adj[n]))

for node_id in nodes_sorted:
    # colors already used by neighbors
    used = {colors_by_id[nbr] for nbr in adj[node_id] if nbr in colors_by_id}

    # pick the first palette index that's not used
    for color in color_palette:
        if color not in used:
            colors_by_id[node_id] = color
            break
    else:
        # if this ever triggers, you needed more than 4 colors
        raise RuntimeError("Ran out of colors, need more than 4 for this graph?")

for body in world.bodies_with_enabled_collision:
    body_color = colors_by_id[body.id]
    body.collision[0].override_mesh_with_color(body_color)

scene.show()
exit()


camera_pose = TransformationMatrix.from_xyz_rpy(
    x=-3, y=0, z=3, roll=-np.pi / 2, pitch=np.pi / 4, yaw=0
).to_np()
# By default, the camera is looking along the -z axis, so we need to rotate it to look along the x-axis.
rotate = trimesh.transformations.rotation_matrix(
    angle=np.radians(-90.0), direction=[0, 1, 0]
)
rotate_x = trimesh.transformations.rotation_matrix(
    angle=np.radians(180.0), direction=[1, 0, 0]
)

scene.graph[scene.camera.name] = camera_pose @ rotate_x @ rotate

# Adjust field of view (FX/FY)
scene.camera.fov = [60, 45]  # horizontal, vertical degrees

scene.show()

# # Render to PNG bytes
# png = scene.save_image(resolution=(1024, 768), visible=True)
#
# with open("render.png", "wb") as f:
#     f.write(png)
