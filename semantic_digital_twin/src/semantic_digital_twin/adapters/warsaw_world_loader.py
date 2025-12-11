import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import UUID

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
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


@dataclass
class WarsawWorldLoader:
    """
    Load a collection of OBJ files from a directory into a single World,
    """

    input_directory: Path
    """
    Directory containing OBJ files to load.
    """

    world: World = field(init=False)
    """
    Loaded World object.
    """

    _camera_field_of_view: Tuple[float, float] = field(default=(60, 45))
    """
    Camera field of view for rendering.
    """

    original_state: Dict[UUID, Any] = field(init=False, default_factory=dict)
    """
    Original visual states of bodies before highlighting.
    """

    def __post_init__(self):
        # Fetch all OBJ files in input directory
        files = [
            os.path.join(self.input_directory, file)
            for file in os.listdir(self.input_directory)
            if file.endswith(".obj")
        ]
        if len(files) == 0:
            raise ValueError(f"No OBJ files found in {self.input_directory}")

        # Load all OBJ files into a single World
        obj_worlds = [OBJParser(file).parse() for file in files]
        main_world = World()
        root = Body(name=PrefixedName("root_body"))
        with main_world.modify_world():
            main_world.add_body(root)
            for obj_world in obj_worlds:
                main_world.merge_world(obj_world)

        # Fix geometry of objects
        pipeline = Pipeline(
            steps=[
                TransformGeometry(
                    TransformationMatrix.from_xyz_rpy(roll=np.pi / 2, pitch=0, yaw=0)
                ),
                CenterLocalGeometryAndPreserveWorldPose(),
            ]
        )
        self.world = pipeline.apply(main_world)

        # Cache original visual states of bodies
        for body in self.world.bodies_with_enabled_collision:
            self.original_state[body.id] = body.collision[0].mesh.visual.copy()

    # %% Public API
    def export_semantic_annotation_inheritance_structure(
        self, output_directory: Path
    ) -> None:
        """
        Export kinematic structure and semantic annotations to JSON files.

        :param output_directory: Directory to write JSON files to.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        self.world.export_kinematic_structure_tree_to_json(
            output_directory / "kinematic_structure.json",
            include_connections=False,
        )
        InheritanceStructureExporter(
            SemanticAnnotation, output_directory / "semantic_annotations.json"
        ).export()

    def export_scene_to_pngs(self, number_of_bodies: int, output_directory: Path):
        """
        Export rendered images of the scene with highlighted groups of bodies.
        :param number_of_bodies: Number of bodies to highlight in each group.
        :param output_directory: Directory to write images to.
        """
        output_directory.mkdir(parents=True, exist_ok=True)

        self.render_scene_from_predefined_poses(
            output_directory,
            "original_render",
        )

        for i, start in enumerate(
            range(
                0,
                len(bodies := self.world.bodies_with_enabled_collision),
                number_of_bodies,
            )
        ):
            group = bodies[start : start + number_of_bodies]
            self._reset_body_colors()
            self._apply_highlight_to_group(group)
            self.render_scene_from_predefined_poses(
                output_directory,
                f"group_{i}_render",
            )

    def render_scene_from_predefined_poses(
        self, output_path: Path, filename_prefix: str
    ):
        """
        Renders the given world for each camera pose, optionally saves images,
        and returns a list of the images as bytes.
        :param output_path: Directory to save images.
        :param filename_prefix: Prefix for image filenames.
        """
        for index, pose in enumerate(self._predefined_camera_transforms):
            self.render_scene_from_camera_pose(
                pose, os.path.join(output_path, f"{filename_prefix}_{index}.png")
            )

    def render_scene_from_camera_pose(
        self, camera_transform: TransformationMatrix, output_filepath=None
    ) -> bytes:
        """Render world from a single camera pose, return PNG bytes."""
        rt = RayTracer(world=self.world)
        scene = rt.scene
        scene.camera.fov = self._camera_field_of_view
        scene.graph[scene.camera.name] = camera_transform
        png = scene.save_image(resolution=(1024, 768), visible=True)
        if output_filepath:
            with open(output_filepath, "wb") as f:
                f.write(png)
        return png

    # %% Export Helpers

    @cached_property
    def _predefined_camera_transforms(self):
        """Return list of camera poses for rendering."""
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

        return camera_poses

    # %% Body Highlighting
    def _reset_body_colors(self):
        """
        Reset all bodies to their original visual states.
        """
        for body in self.world.bodies_with_enabled_collision:
            body.collision[0].mesh.visual = self.original_state[body.id]

    @staticmethod
    def _apply_highlight_to_group(bodies: List[Body]) -> Dict[UUID, Color]:
        """
        Apply distinct highlight colors to a group of bodies.
        """
        colors = Color.distinct_html_colors(len(bodies))
        for body, color in zip(bodies, colors):
            body_mesh: Mesh = body.collision[0]
            body_mesh.override_mesh_with_color(color)
        return {body.id: color for body, color in zip(bodies, colors)}
