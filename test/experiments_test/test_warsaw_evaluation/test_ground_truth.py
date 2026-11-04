"""
Reading a modelled world out as the ground truth a run is compared against.

What matters here is that nothing the world says is lost: an entity carrying no class is
still an entity a reconstruction could have found, and a relation is only worth
comparing if it names both of its ends and the field it was mounted through.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from semantic_digital_twin.adapters.world_mesh_exporter import GeometrySource
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Drawer,
    Floor,
    Handle,
    Mug,
    Room,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import MountKind
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.warsaw.evaluation.ground_truth import (
    ClassCorrection,
    GroundTruthCorrections,
    GroundTruthEdge,
    GroundTruthGraph,
    GroundTruthNode,
    RelationCorrection,
    entity_name_of,
    world_from_provider,
)
from experiments.warsaw.exceptions import (
    CorrectedEntityNotInGraphError,
    GroundTruthAlreadyCorrectedError,
    RelationHasNoEntityError,
    WorldProviderNotFoundError,
)

# %% a small modelled world


def modelled_kitchen() -> World:
    """
    Build a cabinet holding a drawer, that drawer holding a handle, a mug standing in
    the cabinet, and one body the world names nothing.
    """
    world = World.create_with_root_body("root")
    cabinet_body = Body(
        name=PrefixedName("cabinet"),
        collision=ShapeCollection([Box(scale=Scale(1.0, 1.0, 1.0))]),
    )
    drawer_body = Body(
        name=PrefixedName("drawer"),
        collision=ShapeCollection([Box(scale=Scale(0.8, 0.8, 0.2))]),
    )
    handle_body = Body(
        name=PrefixedName("handle"),
        collision=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.2))]),
    )
    mug_body = Body(
        name=PrefixedName("mug"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
    )
    unnamed_body = Body(name=PrefixedName("unnamed_block"))
    with world.modify_world():
        for parent, child, offset in (
            (world.root, cabinet_body, HomogeneousTransformationMatrix()),
            (
                cabinet_body,
                drawer_body,
                HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0),
            ),
            (
                drawer_body,
                handle_body,
                HomogeneousTransformationMatrix.from_xyz_rpy(y=3.0),
            ),
            (world.root, mug_body, HomogeneousTransformationMatrix()),
            (world.root, unnamed_body, HomogeneousTransformationMatrix()),
        ):
            world.add_connection(
                FixedConnection(
                    parent=parent,
                    child=child,
                    parent_T_connection_expression=offset,
                )
            )
        handle = Handle(name=PrefixedName("drawer_handle"), root=handle_body)
        drawer = Drawer(
            name=PrefixedName("top_drawer"), root=drawer_body, handles=[handle]
        )
        cabinet = Cabinet(
            name=PrefixedName("tall_cabinet"), root=cabinet_body, drawers=[drawer]
        )
        world.add_semantic_annotation_recursively(cabinet)
        mug = Mug(name=PrefixedName("blue_mug"), root=mug_body)
        world.add_semantic_annotation_recursively(mug)
    with world.modify_world():
        cabinet.add_object(mug)
    return world


def graph_of_modelled_kitchen() -> GroundTruthGraph:
    """
    Read the small modelled world with its collision shapes measured.
    """
    return GroundTruthGraph.from_world(
        modelled_kitchen(),
        scene="modelled_kitchen",
        geometry_source=GeometrySource.COLLISION,
    )


# %% the entities a world offers to be found


def test_every_entity_is_a_node_whether_or_not_it_carries_a_class():
    """
    A body the world names nothing is still a body a reconstruction could have found, so
    leaving it out would count a correct detection as a false positive.
    """
    graph = graph_of_modelled_kitchen()

    classes_by_name = {node.name: node.semantic_classes for node in graph.nodes}

    assert classes_by_name == {
        "root": [],
        "cabinet": ["Cabinet"],
        "drawer": ["Drawer"],
        "handle": ["Handle"],
        "mug": ["Mug"],
        "unnamed_block": [],
    }


def test_a_node_carries_the_geometry_a_correspondence_is_measured_from():
    """
    Matching a prediction to ground truth needs where an entity is and how big it is.
    """
    graph = graph_of_modelled_kitchen()
    handle = next(node for node in graph.nodes if node.name == "handle")

    assert handle.faces == 12
    np.testing.assert_allclose(np.array(handle.world_transform)[:3, 3], [2.0, 3.0, 0.0])
    np.testing.assert_allclose(handle.bounds, [[1.9, 2.9, -0.1], [2.1, 3.1, 0.1]])


def test_a_node_without_geometry_is_reported_as_having_none():
    """
    An entity with no shapes is distinguished from one whose shapes were not measured.
    """
    graph = graph_of_modelled_kitchen()
    unnamed = next(node for node in graph.nodes if node.name == "unnamed_block")

    assert unnamed.faces == 0
    assert unnamed.bounds is None


def test_a_node_names_the_entity_it_hangs_from():
    """
    The kinematic tree is what tells a partial scan which entities belong together.
    """
    graph = graph_of_modelled_kitchen()
    parents = {node.name: node.parent for node in graph.nodes}

    assert parents["handle"] == "drawer"
    assert parents["drawer"] == "cabinet"
    assert parents["cabinet"] == "root"
    assert parents["root"] is None


# %% the relations a world holds


def test_a_mounted_part_becomes_an_edge_between_the_entities_carrying_it():
    """
    Both ends are named by their entity, because that is what a run's graph names too.
    """
    graph = graph_of_modelled_kitchen()

    parts = [edge for edge in graph.edges if edge.relation == MountKind.PART.value]

    assert sorted(parts, key=lambda edge: edge.part) == [
        GroundTruthEdge(
            whole="cabinet", part="drawer", relation="part", field_name="drawers"
        ),
        GroundTruthEdge(
            whole="drawer", part="handle", relation="part", field_name="handles"
        ),
    ]


def test_an_occupant_becomes_a_containment_edge_rather_than_a_part_edge():
    """
    A mug in a cabinet is not a structural part of it, and scoring it as one would
    credit the wrong relation.
    """
    graph = graph_of_modelled_kitchen()

    assert [
        edge for edge in graph.edges if edge.relation == MountKind.CONTAINS.value
    ] == [
        GroundTruthEdge(
            whole="cabinet", part="mug", relation="contains", field_name="objects"
        )
    ]


def test_a_relation_reaching_nothing_the_world_carries_is_refused():
    """
    A room is not rooted at any body or region, so nothing carries it and no edge can
    name it.

    Naming both ends is what makes an edge comparable, so this has to be raised rather
    than quietly dropped.
    """
    room = Room(
        name=PrefixedName("kitchen"),
        floor=Floor(
            name=PrefixedName("kitchen_floor"), root=Body(name=PrefixedName("floor"))
        ),
    )

    with pytest.raises(RelationHasNoEntityError):
        entity_name_of(room)


def test_an_annotation_is_named_by_the_entity_it_is_rooted_at():
    """
    Inferred worlds call every handle 'Handle', so an annotation's own name identifies
    nothing and the entity underneath it has to.
    """
    world = modelled_kitchen()
    [handle] = world.get_semantic_annotations_by_type(Handle)

    assert isinstance(handle, HasRootKinematicStructureEntity)
    assert entity_name_of(handle) == entity_name_of(handle.root)


# %% what is written out


def test_the_graph_reads_back_as_what_was_written():
    """
    The snapshot is read long after the world it came from is gone.
    """
    graph = graph_of_modelled_kitchen()

    assert GroundTruthGraph.from_json(graph.to_json()) == graph


def test_the_graph_records_what_it_was_read_from():
    """
    A result is only reproducible if it says which world and which shapes made it.
    """
    graph = graph_of_modelled_kitchen()

    assert graph.scene == "modelled_kitchen"
    assert graph.geometry_source == GeometrySource.COLLISION.value
    assert graph.frame == "root"


def test_the_graph_is_written_in_a_stable_order():
    """
    Inferring a world's classes materializes joint bodies in a varying order, so two
    exports of one world would differ everywhere unless the snapshot fixes the order.
    """
    graph = graph_of_modelled_kitchen()

    assert [node.name for node in graph.nodes] == sorted(
        node.name for node in graph.nodes
    )
    assert graph.edges == sorted(
        graph.edges,
        key=lambda edge: (edge.whole, edge.part, edge.relation, edge.field_name),
    )


# %% ground truth a person supplied


def graph_with_an_unnamed_cabinet() -> GroundTruthGraph:
    """
    Build a graph whose cabinet the world left unnamed, as the IAI apartment does.
    """
    return GroundTruthGraph(
        scene="apartment.urdf",
        frame="root",
        geometry_source="visual",
        nodes=[
            GroundTruthNode(
                name="cabinet12",
                source_id="a",
                parent="side_A",
                semantic_classes=[],
                faces=92,
                world_transform=[[1.0]],
                bounds=None,
            ),
            GroundTruthNode(
                name="cabinet12_door_top_left",
                source_id="b",
                parent="cabinet12",
                semantic_classes=[],
                faces=12,
                world_transform=[[1.0]],
                bounds=None,
            ),
            GroundTruthNode(
                name="handle_cab1",
                source_id="c",
                parent="cabinet1",
                semantic_classes=["Handle"],
                faces=30,
                world_transform=[[1.0]],
                bounds=None,
            ),
        ],
    )


def corrections_for_the_cabinet() -> GroundTruthCorrections:
    """
    The classes and relation a person supplies for a cabinet modelled without one.
    """
    return GroundTruthCorrections(
        scene="apartment.urdf",
        classes=[
            ClassCorrection(
                entity="cabinet12",
                semantic_class="Cabinet",
                reason="Modelled without handles, which the door rules require.",
            ),
            ClassCorrection(
                entity="cabinet12_door_top_left",
                semantic_class="Door",
                reason="Swings on a revolute joint but carries no handle.",
            ),
        ],
        relations=[
            RelationCorrection(
                edge=GroundTruthEdge(
                    whole="cabinet12",
                    part="cabinet12_door_top_left",
                    relation="part",
                    field_name="doors",
                ),
                reason="The door the cabinet holds, which follows from its classes.",
            )
        ],
    )


def test_a_corrected_entity_is_evaluated_as_the_class_a_person_gave_it():
    """
    The point of the overlay: the comparison uses what the person decided.
    """
    corrected = corrections_for_the_cabinet().applied_to(
        graph_with_an_unnamed_cabinet()
    )

    cabinet = next(node for node in corrected.nodes if node.name == "cabinet12")

    assert cabinet.classes == ["Cabinet"]


def test_a_correction_does_not_overwrite_what_the_world_said():
    """
    Ground truth a person supplied has to stay distinguishable from ground truth the
    world carried, or an audit of the rules cannot tell the two apart afterwards.
    """
    corrected = corrections_for_the_cabinet().applied_to(
        graph_with_an_unnamed_cabinet()
    )

    cabinet = next(node for node in corrected.nodes if node.name == "cabinet12")

    assert cabinet.semantic_classes == []
    assert cabinet.correction.semantic_class == "Cabinet"
    assert cabinet.correction.reason == (
        "Modelled without handles, which the door rules require."
    )


def test_an_uncorrected_entity_is_evaluated_as_the_world_named_it():
    """
    An overlay says nothing about the entities it does not mention.
    """
    corrected = corrections_for_the_cabinet().applied_to(
        graph_with_an_unnamed_cabinet()
    )

    handle = next(node for node in corrected.nodes if node.name == "handle_cab1")

    assert handle.classes == ["Handle"]
    assert handle.correction is None


def test_a_corrected_relation_joins_the_edges_the_world_holds():
    """
    A cabinet whose doors are named but not attached would cost a run precision on
    hierarchy edges it got right.
    """
    corrected = corrections_for_the_cabinet().applied_to(
        graph_with_an_unnamed_cabinet()
    )

    assert corrected.edges == [
        GroundTruthEdge(
            whole="cabinet12",
            part="cabinet12_door_top_left",
            relation="part",
            field_name="doors",
        )
    ]


def test_the_corrected_graph_records_every_correction_applied_to_it():
    """
    A result has to say what was supplied by hand, not only what it concluded.
    """
    corrections = corrections_for_the_cabinet()

    corrected = corrections.applied_to(graph_with_an_unnamed_cabinet())

    assert corrected.corrections == corrections


def test_correcting_an_entity_the_graph_does_not_hold_is_refused():
    """
    A name matching nothing is a typo or a stale overlay, and applying it quietly would
    leave the ground truth wrong in the way the overlay was written to put right.
    """
    corrections = GroundTruthCorrections(
        scene="apartment.urdf",
        classes=[
            ClassCorrection(
                entity="cabinet99", semantic_class="Cabinet", reason="A typo."
            )
        ],
    )

    with pytest.raises(CorrectedEntityNotInGraphError):
        corrections.applied_to(graph_with_an_unnamed_cabinet())


def test_an_overlay_is_read_back_as_what_was_written(tmp_path):
    """
    The overlay is checked in beside the code and read by the export command.
    """
    corrections = corrections_for_the_cabinet()
    written = tmp_path / "corrections.json"
    written.write_text(json.dumps(corrections.to_json(), indent=2))

    assert GroundTruthCorrections.read(written) == corrections


def test_a_corrected_graph_reads_back_as_what_was_written():
    """
    The snapshot is read long after the world and the overlay are gone.
    """
    corrected = corrections_for_the_cabinet().applied_to(
        graph_with_an_unnamed_cabinet()
    )

    assert GroundTruthGraph.from_json(corrected.to_json()) == corrected


def test_correcting_a_graph_that_is_already_corrected_is_refused():
    """
    A graph records the one overlay it was given.

    Applying a second drops the first from that record while quietly undoing it in the
    nodes, so the graph would claim corrections it no longer carries.
    """
    corrected = corrections_for_the_cabinet().applied_to(
        graph_with_an_unnamed_cabinet()
    )
    more = GroundTruthCorrections(
        scene="apartment.urdf",
        classes=[
            ClassCorrection(
                entity="handle_cab1",
                semantic_class="Handle",
                reason="A second thought about a different entity.",
            )
        ],
    )

    with pytest.raises(GroundTruthAlreadyCorrectedError):
        more.applied_to(corrected)


# %% worlds that build themselves


def test_a_world_provider_is_built_by_name():
    """
    A modelled world written as Python names its own classes as it builds, so nothing
    has to be inferred afterwards.
    """
    world = world_from_provider(
        "semantic_digital_twin.predefined_maps.kitchen_environment:KitchenEnvironment"
    )

    assert len(world.bodies) > 0
    assert {type(one).__name__ for one in world.semantic_annotations} >= {
        "Cabinet",
        "Drawer",
        "Handle",
    }


def test_a_name_that_is_not_module_and_class_is_refused():
    """
    Half a reference reaches no module, and importing it would fail somewhere less
    obvious than here.
    """
    with pytest.raises(WorldProviderNotFoundError):
        world_from_provider("semantic_digital_twin.predefined_maps")


def test_a_module_holding_no_such_provider_is_refused():
    """
    A renamed or misspelled provider names a module that really exists, so the failure
    has to say which half was wrong.
    """
    with pytest.raises(WorldProviderNotFoundError) as raised:
        world_from_provider(
            "semantic_digital_twin.predefined_maps.kitchen_environment:NoSuchThing"
        )

    assert "NoSuchThing" in str(raised.value)
