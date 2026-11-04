"""
Tests for extracting stable per-body meshes from a world state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from semantic_digital_twin.adapters.world_mesh_exporter import (
    GeometrySource,
    WorldMeshExporter,
    WorldMeshExtractor,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% world fixture


def world_with_compound_body_and_handle() -> World:
    """
    Build a world with visual, collision, compound, and empty bodies.
    """
    world = World.create_with_root_body("root")
    cabinet = Body(
        name=PrefixedName("cabinet"),
        collision=ShapeCollection(
            [
                Box(scale=Scale(1.0, 1.0, 1.0)),
                Box(
                    scale=Scale(0.5, 0.5, 0.5),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
                ),
            ]
        ),
    )
    handle = Body(
        name=PrefixedName("handle"),
        collision=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.2))]),
        visual=ShapeCollection([Box(scale=Scale(0.4, 0.2, 0.2))]),
    )
    empty_frame = Body(name=PrefixedName("empty_frame"))

    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=cabinet,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.0
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=cabinet,
                child=handle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=1.0
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=empty_frame,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=3.0
                ),
            )
        )
        handle_annotation = Handle(
            name=PrefixedName("cabinet_handle_annotation"), root=handle
        )
        drawer_annotation = Drawer(
            name=PrefixedName("cabinet_drawer_annotation"),
            root=cabinet,
            handles=[handle_annotation],
        )
        world.add_semantic_annotation_recursively(drawer_annotation)
    return world


# %% extraction


def test_extracts_one_mesh_per_body_without_descendant_geometry() -> None:
    """
    A body's mesh should contain its own shapes but not its child's shapes.
    """
    world = world_with_compound_body_and_handle()

    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world
    )
    meshes_by_name = {body_mesh.name: body_mesh for body_mesh in snapshot.body_meshes}

    assert len(meshes_by_name["cabinet"].local_mesh.faces) == 24
    assert len(meshes_by_name["handle"].local_mesh.faces) == 12
    assert meshes_by_name["empty_frame"].local_mesh is None


def test_records_hierarchy_and_world_transform() -> None:
    """
    Each extracted body should retain its parent and current world transform.
    """
    world = world_with_compound_body_and_handle()

    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world
    )
    meshes_by_name = {body_mesh.name: body_mesh for body_mesh in snapshot.body_meshes}

    assert (
        meshes_by_name["handle"].parent_identifier
        == meshes_by_name["cabinet"].identifier
    )
    np.testing.assert_allclose(
        meshes_by_name["handle"].world_transform[:3, 3], [2.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        meshes_by_name["handle"].world_mesh.bounds,
        [[1.9, 0.9, -0.1], [2.1, 1.1, 0.1]],
    )


def test_records_direct_and_context_semantic_annotations() -> None:
    """
    Bodies should distinguish their own annotations from containing ones.
    """
    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world_with_compound_body_and_handle()
    )
    bodies_by_name = {body.name: body for body in snapshot.body_meshes}
    annotations_by_type = {
        annotation.type_name: annotation for annotation in snapshot.semantic_annotations
    }

    handle_annotation = annotations_by_type["Handle"]
    drawer_annotation = annotations_by_type["Drawer"]
    cabinet = bodies_by_name["cabinet"]
    handle = bodies_by_name["handle"]

    assert handle_annotation.root_body_identifier == handle.identifier
    assert handle_annotation.body_identifiers == (handle.identifier,)
    assert drawer_annotation.root_body_identifier == cabinet.identifier
    assert drawer_annotation.body_identifiers == (
        cabinet.identifier,
        handle.identifier,
    )
    assert cabinet.direct_semantic_annotation_identifiers == (
        drawer_annotation.identifier,
    )
    assert cabinet.context_semantic_annotation_identifiers == ()
    assert handle.direct_semantic_annotation_identifiers == (
        handle_annotation.identifier,
    )
    assert handle.context_semantic_annotation_identifiers == (
        drawer_annotation.identifier,
    )


def test_visual_geometry_falls_back_to_collision() -> None:
    """
    Visual geometry should be preferred and collision used only when absent.
    """
    world = world_with_compound_body_and_handle()

    snapshot = WorldMeshExtractor(
        geometry_source=GeometrySource.VISUAL_WITH_COLLISION_FALLBACK
    ).extract(world)
    meshes_by_name = {body_mesh.name: body_mesh for body_mesh in snapshot.body_meshes}

    assert meshes_by_name["cabinet"].used_geometry_source is GeometrySource.COLLISION
    assert meshes_by_name["handle"].used_geometry_source is GeometrySource.VISUAL
    np.testing.assert_allclose(
        meshes_by_name["handle"].local_mesh.extents, [0.4, 0.2, 0.2]
    )
    assert meshes_by_name["empty_frame"].used_geometry_source is None


# %% serialization


def test_exports_reloadable_glb_with_one_node_per_body(tmp_path: Path) -> None:
    """
    The GLB should retain body nodes, hierarchy, transforms, and local meshes.
    """
    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world_with_compound_body_and_handle()
    )

    paths = WorldMeshExporter().export(snapshot, tmp_path)
    scene = trimesh.load(paths.scene, force="scene")

    assert {"root", "cabinet", "handle", "empty_frame"} <= set(scene.graph.nodes)
    handle_transform, handle_geometry = scene.graph.get("handle")
    _, cabinet_geometry = scene.graph.get("cabinet")
    assert scene.graph.transforms.parents["handle"] == "cabinet"
    np.testing.assert_allclose(handle_transform[:3, 3], [2.0, 1.0, 0.0])
    np.testing.assert_allclose(
        scene.graph.get("empty_frame")[0][:3, 3], [0.0, 0.0, 3.0]
    )
    assert len(scene.geometry[cabinet_geometry].faces) == 24
    assert len(scene.geometry[handle_geometry].faces) == 12


def test_exports_manifest_linking_body_names_nodes_and_meshes(
    tmp_path: Path,
) -> None:
    """
    The manifest should explicitly connect labels to GLB nodes and geometry.
    """
    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world_with_compound_body_and_handle()
    )

    paths = WorldMeshExporter().export(snapshot, tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    bodies_by_name = {body["name"]: body for body in manifest["bodies"]}

    annotations_by_type = {
        annotation["type_name"]: annotation
        for annotation in manifest["semantic_annotations"]
    }

    assert manifest["schema_version"] == 2
    assert annotations_by_type["Drawer"]["name"] == "cabinet_drawer_annotation"
    assert (
        annotations_by_type["Drawer"]["root_body_id"] == bodies_by_name["cabinet"]["id"]
    )
    assert annotations_by_type["Drawer"]["body_ids"] == [
        bodies_by_name["cabinet"]["id"],
        bodies_by_name["handle"]["id"],
    ]
    assert bodies_by_name["cabinet"]["direct_semantic_annotation_ids"] == [
        annotations_by_type["Drawer"]["id"]
    ]
    assert bodies_by_name["handle"]["direct_semantic_annotation_ids"] == [
        annotations_by_type["Handle"]["id"]
    ]
    assert bodies_by_name["handle"]["context_semantic_annotation_ids"] == [
        annotations_by_type["Drawer"]["id"]
    ]
    assert bodies_by_name["handle"]["parent_id"] == bodies_by_name["cabinet"]["id"]
    assert bodies_by_name["handle"]["node_name"] == "handle"
    assert (
        bodies_by_name["handle"]["geometry_name"]
        == f"body_{bodies_by_name['handle']['id']}"
    )
    assert bodies_by_name["handle"]["geometry_source"] == "collision"
    assert bodies_by_name["handle"]["shape_types"] == ["Box"]
    assert bodies_by_name["empty_frame"]["geometry_name"] is None
    assert bodies_by_name["empty_frame"]["vertex_count"] == 0
    assert bodies_by_name["empty_frame"]["face_count"] == 0


def test_exports_world_aligned_mesh_for_each_nonempty_body(tmp_path: Path) -> None:
    """
    Each body's standalone PLY should use the common world coordinate frame.
    """
    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world_with_compound_body_and_handle()
    )

    paths = WorldMeshExporter().export(snapshot, tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    bodies_by_name = {body["name"]: body for body in manifest["bodies"]}
    handle_mesh_path = tmp_path / bodies_by_name["handle"]["mesh_file"]
    handle_mesh = trimesh.load(handle_mesh_path, force="mesh")

    assert handle_mesh_path.parent == paths.bodies_directory
    np.testing.assert_allclose(
        handle_mesh.bounds,
        [[1.9, 0.9, -0.1], [2.1, 1.1, 0.1]],
    )
    assert bodies_by_name["empty_frame"]["mesh_file"] is None


def test_exports_combined_npz_with_body_labels(tmp_path: Path) -> None:
    """
    The combined arrays should label every face and vertex by source body ID.
    """
    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world_with_compound_body_and_handle()
    )
    snapshot_by_name = {body.name: body for body in snapshot.body_meshes}

    paths = WorldMeshExporter().export(snapshot, tmp_path)
    with np.load(paths.labeled_mesh, allow_pickle=False) as labeled_mesh:
        cabinet_id = snapshot_by_name["cabinet"].identifier
        handle_id = snapshot_by_name["handle"].identifier
        assert len(labeled_mesh["faces"]) == 36
        assert np.count_nonzero(labeled_mesh["face_body_ids"] == cabinet_id) == 24
        assert np.count_nonzero(labeled_mesh["face_body_ids"] == handle_id) == 12
        assert set(labeled_mesh["vertex_body_ids"]) == {cabinet_id, handle_id}
        assert list(labeled_mesh["body_names"]) == [
            body.name for body in snapshot.body_meshes
        ]
        np.testing.assert_allclose(
            labeled_mesh["vertices"][labeled_mesh["vertex_body_ids"] == handle_id].min(
                axis=0
            ),
            [1.9, 0.9, -0.1],
        )


def test_exports_npz_semantic_annotation_relations(tmp_path: Path) -> None:
    """
    The NPZ should preserve direct and contextual body-annotation relations.
    """
    snapshot = WorldMeshExtractor(geometry_source=GeometrySource.COLLISION).extract(
        world_with_compound_body_and_handle()
    )
    bodies_by_name = {body.name: body for body in snapshot.body_meshes}
    annotations_by_type = {
        annotation.type_name: annotation for annotation in snapshot.semantic_annotations
    }

    paths = WorldMeshExporter().export(snapshot, tmp_path)
    with np.load(paths.labeled_mesh, allow_pickle=False) as labeled_mesh:
        assert list(labeled_mesh["annotation_type_names"]) == ["Handle", "Drawer"]
        assert list(labeled_mesh["annotation_names"]) == [
            "cabinet_handle_annotation",
            "cabinet_drawer_annotation",
        ]
        relations = set(
            zip(
                labeled_mesh["body_annotation_body_ids"].tolist(),
                labeled_mesh["body_annotation_ids"].tolist(),
                labeled_mesh["body_annotation_relations"].tolist(),
            )
        )
        assert relations == {
            (
                bodies_by_name["cabinet"].identifier,
                annotations_by_type["Drawer"].identifier,
                "direct",
            ),
            (
                bodies_by_name["handle"].identifier,
                annotations_by_type["Handle"].identifier,
                "direct",
            ),
            (
                bodies_by_name["handle"].identifier,
                annotations_by_type["Drawer"].identifier,
                "context",
            ),
        }
