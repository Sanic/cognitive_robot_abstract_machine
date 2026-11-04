import numpy as np
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Mesh, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_post_init_transformation():
    w = World()
    root = Body(name=PrefixedName("root"))
    b1 = Body(name=PrefixedName("b1"))

    with w.modify_world():
        w.add_connection(
            FixedConnection(
                parent=root,
                child=b1,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, reference_frame=root
                ),
            )
        )

    shape = Sphere(
        radius=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, reference_frame=root),
    )
    shape_collection = ShapeCollection(
        shapes=[shape],
        reference_frame=b1,
    )
    shape_collection.transform_all_shapes_to_own_frame()
    assert shape.origin.reference_frame == b1
    assert shape.origin.position.x == 2.0

    shape = Sphere(
        radius=1,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, reference_frame=root),
    )

    shape_collection = ShapeCollection(reference_frame=b1)
    shape_collection.append(shape)
    shape_collection.transform_all_shapes_to_own_frame()
    assert shape.origin.reference_frame == b1
    assert shape.origin.position.x == 2.0


# %% painting a whole collection


def test_dye_shapes_paints_an_already_built_mesh(tmp_path):
    """
    Dyeing a collection reaches meshes that have already built their trimesh, which keep
    the color they were built with until they are repainted.
    """
    mesh = Mesh.from_trimesh(
        mesh=trimesh.creation.box(extents=(1.0, 1.0, 1.0)), directory=tmp_path
    )
    color = Color.from_rgba(red=0.0, green=1.0, blue=0.0, opacity=1.0)
    # Build and cache the mesh before dyeing, which repaints what is cached.
    mesh.mesh.vertices

    ShapeCollection(shapes=[mesh]).dye_shapes(color)

    np.testing.assert_array_equal(
        np.asarray(mesh.mesh.visual.face_colors),
        np.tile(
            trimesh.visual.color.to_rgba(color.to_rgba()), (len(mesh.mesh.faces), 1)
        ),
    )
