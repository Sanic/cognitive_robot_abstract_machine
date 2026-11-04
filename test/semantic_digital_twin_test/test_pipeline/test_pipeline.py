import os
import re
import unittest
from dataclasses import dataclass

import numpy as np
import trimesh
from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.adapters.fbx import FBXParser
from semantic_digital_twin.adapters.procthor.procthor_pipelines import (
    dresser_from_body_in_world,
    drawer_from_body_in_world,
    door_from_body_in_world,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.pipeline import (
    Step,
    Pipeline,
    BodyFilter,
    CenterLocalGeometryAndPreserveWorldPose,
    BodyFactoryReplace,
    TransformGeometry,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dummy_world = World()
        b1 = Body(name=PrefixedName("body1", "krrood_test"))
        b2 = Body(name=PrefixedName("body2", "krrood_test"))
        c1 = FixedConnection(b1, b2, HomogeneousTransformationMatrix())
        with cls.dummy_world.modify_world():
            cls.dummy_world.add_body(b1)
            cls.dummy_world.add_body(b2)
            cls.dummy_world.add_connection(c1)

        cls.fbx_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "fbx",
            "test_dressers.fbx",
        )

    def test_pipeline_and_step(self):

        @dataclass
        class TestStep(Step):
            body_name: PrefixedName

            def _apply(self, world: World) -> World:
                b1 = Body(name=self.body_name)
                world.add_body(b1)
                return world

        pipeline = Pipeline(
            steps=[TestStep(body_name=PrefixedName("body1", "krrood_test"))]
        )

        dummy_world = World()

        dummy_world = pipeline.apply(dummy_world)

        self.assertEqual(len(dummy_world.bodies), 1)
        self.assertEqual(dummy_world.root.name, PrefixedName("body1", "krrood_test"))

    def test_body_filter(self):

        pipeline = Pipeline(
            steps=[BodyFilter(lambda x: x.name == PrefixedName("body1", "krrood_test"))]
        )

        filtered_world = pipeline.apply(self.dummy_world)
        self.assertEqual(len(filtered_world.bodies), 1)
        self.assertEqual(filtered_world.root.name, PrefixedName("body1", "krrood_test"))

    def test_center_local_geometry_and_preserve_world_pose(self):
        world = FBXParser(self.fbx_path).create_world()

        original_bounding_boxes = [
            body.collision.as_bounding_box_collection_at_origin(
                HomogeneousTransformationMatrix(reference_frame=world.root)
            ).bounding_boxes[0]
            for body in world.bodies_with_collision
        ]

        original_global_poses = [
            body.global_transform.to_np() for body in world.bodies_with_collision
        ]
        for pose in original_global_poses:
            np.testing.assert_almost_equal(pose, np.eye(4))

        pipeline = Pipeline(steps=[CenterLocalGeometryAndPreserveWorldPose()])

        centered_world = pipeline.apply(world)

        centered_global_poses = [
            body.global_transform.to_np()
            for body in centered_world.bodies_with_collision
        ]
        for original, centered in zip(original_global_poses, centered_global_poses):
            assert not np.allclose(original, centered)

        new_bounding_boxes = [
            body.collision.as_bounding_box_collection_at_origin(
                HomogeneousTransformationMatrix(reference_frame=centered_world.root)
            ).bounding_boxes[0]
            for body in centered_world.bodies_with_collision
        ]

        self.assertEqual(original_bounding_boxes, new_bounding_boxes)

    def test_body_replace(self):
        dresser_pattern = re.compile(r"^.*dresser_(?!drawer\b).*$", re.IGNORECASE)
        world = FBXParser(self.fbx_path).create_world()

        self.assertIsNotNone(world.get_body_by_name("dresser_205"))
        self.assertIsNotNone(world.get_body_by_name("dresser_217"))
        self.assertFalse(world.semantic_annotations)

        procthor_replace_pipeline = Pipeline(
            [
                BodyFactoryReplace(
                    body_condition=lambda b: (
                        bool(dresser_pattern.fullmatch(b.name.name))
                        and not (
                            "drawer" in b.name.name.lower()
                            or "door" in b.name.name.lower()
                        )
                    ),
                    annotation_creator=dresser_from_body_in_world,
                )
            ]
        )

        replaced_world = procthor_replace_pipeline.apply(world)

        self.assertIsNotNone(replaced_world.get_body_by_name("dresser_205"))
        self.assertIsNotNone(replaced_world.get_body_by_name("dresser_217"))

        self.assertTrue(replaced_world.semantic_annotations)
        self.assertIsNotNone(
            replaced_world.get_semantic_annotation_by_name("dresser_205")
        )
        self.assertIsNotNone(
            replaced_world.get_semantic_annotation_by_name("dresser_217")
        )

    def test_dresser_from_body(self):
        world = FBXParser(self.fbx_path).create_world()

        self.assertIsNotNone(dresser := world.get_body_by_name("dresser_205"))

        dresser = dresser_from_body_in_world(dresser, world)

        self.assertEqual(dresser.name.name, "dresser_205")

    def test_drawer_from_body(self):
        world = FBXParser(self.fbx_path).create_world()

        self.assertIsNotNone(drawer := world.get_body_by_name("dresser_drawer_205_1"))

        drawer = drawer_from_body_in_world(drawer, world)

        self.assertEqual(drawer.name.name, "dresser_drawer_205_1")

    def test_door_from_body(self):
        world = FBXParser(self.fbx_path).create_world()

        self.assertIsNotNone(door := world.get_body_by_name("dresser_door_217_1"))

        door = door_from_body_in_world(door, world)

        self.assertEqual(door.name.name, "dresser_door_217_1")


if __name__ == "__main__":
    unittest.main()


# %% transforming the geometry of a whole world


def test_transform_geometry_transforms_the_mesh(jeroen_cup_world_fixture):
    """
    The step carries every mesh vertex through the given transformation.
    """
    [cup] = jeroen_cup_world_fixture.bodies
    [shape] = cup.collision
    original_vertices = shape.mesh.vertices.copy()
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(roll=np.pi / 2)

    Pipeline([TransformGeometry(transform)]).apply(jeroen_cup_world_fixture)

    np.testing.assert_allclose(
        shape.mesh.vertices,
        trimesh.transform_points(original_vertices, transform.to_np()),
        atol=1e-9,
    )


def test_transform_geometry_moves_a_primitive_shape():
    """
    A collision shape that is not a mesh is carried by the same transformation.

    A primitive builds its mesh fresh on every access, so moving that mesh moves
    nothing; what stands for its placement is its origin.
    """
    world = World()
    box = Body(name=PrefixedName("box", "krrood_test"))
    box.collision = ShapeCollection(
        shapes=[Box(scale=Scale(1.0, 1.0, 1.0))], reference_frame=box
    )
    with world.modify_world():
        world.add_body(box)

    original_origin = box.collision[0].origin.to_np().copy()
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0, y=2.0, z=3.0)

    Pipeline([TransformGeometry(transform)]).apply(world)

    np.testing.assert_allclose(
        box.collision[0].origin.to_np(),
        transform.to_np() @ original_origin,
        atol=1e-9,
    )
