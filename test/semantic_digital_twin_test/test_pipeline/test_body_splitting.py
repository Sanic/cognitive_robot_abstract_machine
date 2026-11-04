import numpy as np
import pytest
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import (
    BodyWithoutShapeError,
    PieceOutsideShapeError,
)
from semantic_digital_twin.part_finding.splitting import BodyCutter, ShapePiece
from semantic_digital_twin.pipeline.body_splitting import SplitBodyIntoPieces
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
)

# %% fixtures


@pytest.fixture
def room() -> World:
    """
    A world whose root body is one box mesh that looks and collides alike.
    """
    world = World.create_with_root_body(root_body_name="room")
    room_body = world.root
    shapes = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=trimesh.creation.box(),
                origin=HomogeneousTransformationMatrix(reference_frame=room_body),
            )
        ],
        reference_frame=room_body,
    )
    with world.modify_world():
        room_body.update_shapes(visual=shapes, collision=shapes)
    return world


@pytest.fixture
def box() -> Body:
    """
    A body, in no world, whose shape is one box mesh that looks and collides alike.
    """
    body = Body(name=PrefixedName("box"))
    shapes = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=trimesh.creation.box(),
                origin=HomogeneousTransformationMatrix(reference_frame=body),
            )
        ],
        reference_frame=body,
    )
    body.visual = shapes
    body.collision = shapes
    return body


def face_count(body: Body) -> int:
    return sum(len(shape.mesh.faces) for shape in body.visual)


# %% splitting


def test_a_piece_gets_the_faces_it_takes(room):
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid]).apply(room)

    assert face_count(lid.body) == len(lid.faces)


def test_the_body_keeps_the_faces_no_piece_takes(room):
    total = face_count(room.root)
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))
    side = ShapePiece(body=Body(name=PrefixedName("side")), faces=np.array([2, 3]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid, side]).apply(room)

    assert face_count(room.root) == total - len(lid.faces) - len(side.faces)


def test_each_piece_is_fixed_to_the_body_it_is_cut_from(room):
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid]).apply(room)

    assert isinstance(lid.body.parent_connection, FixedConnection)
    assert lid.body.parent_connection.parent is room.root


def test_a_piece_without_faces_gets_a_body_without_shape(room):
    empty = ShapePiece(body=Body(name=PrefixedName("empty")), faces=np.array([], int))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[empty]).apply(room)

    assert empty.body in room.bodies
    assert not empty.body.visual


def test_pieces_collide_as_they_look_when_the_body_did(room):
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))

    SplitBodyIntoPieces(body_name=room.root.name, pieces=[lid]).apply(room)

    assert lid.body.collision is lid.body.visual


# %% cutting a body before it is in a world


def test_a_body_cut_before_it_is_in_a_world_joins_it_with_its_pieces(box):
    total = face_count(box)
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))
    cutter = BodyCutter()
    cut = cutter.cut(box, cutter.shape_to_cut(box), [lid])
    world = World()

    with world.modify_world():
        cut.add_to_world(world)

    assert face_count(box) == total - len(lid.faces)
    assert lid.body.parent_connection.parent is box


def test_cutting_a_body_before_it_is_in_a_world_records_no_change_of_its_shapes(box):
    """
    A change of a body's shapes is recorded as the whole body serialized before and
    after it, which for a large mesh costs several times the mesh's memory.
    """
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0, 1]))
    cutter = BodyCutter()
    cut = cutter.cut(box, cutter.shape_to_cut(box), [lid])
    world = World()

    with world.modify_world():
        cut.add_to_world(world)

    assert not any(
        isinstance(modification, AttributeUpdateModification)
        for block in world.get_world_model_manager().model_modification_blocks
        for modification in block.modifications
    )


# %% misuse


def test_a_piece_taking_a_face_the_shape_lacks_raises(room):
    beyond = ShapePiece(
        body=Body(name=PrefixedName("beyond")),
        faces=np.array([face_count(room.root)]),
    )

    with pytest.raises(PieceOutsideShapeError):
        SplitBodyIntoPieces(body_name=room.root.name, pieces=[beyond]).apply(room)


def test_cutting_a_body_without_shape_raises():
    world = World.create_with_root_body(root_body_name="room")
    lid = ShapePiece(body=Body(name=PrefixedName("lid")), faces=np.array([0]))

    with pytest.raises(BodyWithoutShapeError):
        SplitBodyIntoPieces(body_name=world.root.name, pieces=[lid]).apply(world)
