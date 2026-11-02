"""
Reading the annotation taxonomy back out of the classes the interpreter holds.

What a model is shown about the ontology is built here, so what matters is that the
order it puts bases in is one Python can actually combine, and that a class's reported
bases are the ones it really derives from.
"""

from __future__ import annotations

from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
    HasDoors,
    HasDrawers,
    HasRootBody,
    IsStorageSpace,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.part_whole import (
    field_holding,
    fields_accepting,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    CounterTop,
    Dishwasher,
    Drawer,
    Furniture,
    Handle,
    KitchenIsland,
    Mug,
    Sink,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    MountKind,
    annotation_classes,
    compose_class,
    declared_annotation_classes,
    in_base_order,
    mounted_relations_of,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

# %% ordering the bases a proposal names


def test_a_base_named_before_its_own_subclass_is_put_after_it():
    """
    Python refuses a class naming a base in front of something deriving from that base.
    """
    assert issubclass(IsStorageSpace, HasRootBody)

    assert in_base_order([HasRootBody, IsStorageSpace]) == (
        IsStorageSpace,
        HasRootBody,
    )


def test_bases_already_in_a_usable_order_are_left_as_they_are():
    """
    Reordering what is already right would change a class's method resolution for
    nothing.
    """
    assert in_base_order([IsStorageSpace, HasRootBody]) == (
        IsStorageSpace,
        HasRootBody,
    )


def test_a_base_named_twice_is_kept_once():
    """
    Python refuses a class naming the same base twice.
    """
    assert in_base_order([HasDoors, HasDrawers, HasDoors]) == (HasDoors, HasDrawers)


def test_the_order_it_returns_can_be_combined_into_a_class():
    """
    The point of the ordering: the classes go together afterwards.
    """
    composed = compose_class(
        "StorageWithRootBody", Furniture, in_base_order([HasRootBody, IsStorageSpace])
    )

    assert issubclass(composed, IsStorageSpace)
    assert issubclass(composed, HasRootBody)


# %% which classes there are


def test_every_annotation_class_is_reported_under_its_own_name():
    """
    A name read back from a file or a model is looked up in exactly this mapping.
    """
    classes = annotation_classes(SemanticAnnotation)

    assert classes["Furniture"] is Furniture
    assert classes["HasDrawers"] is HasDrawers
    assert all(name == found.__name__ for name, found in classes.items())


# %% what an annotation holds


def annotated_cabinet_world() -> World:
    """
    Build a cabinet holding one drawer, that drawer holding one handle, and a mug
    standing on the cabinet.
    """
    world = World.create_with_root_body("root")
    cabinet_body = Body(name=PrefixedName("cabinet"))
    drawer_body = Body(name=PrefixedName("drawer"))
    handle_body = Body(name=PrefixedName("handle"))
    mug_body = Body(name=PrefixedName("mug"))
    with world.modify_world():
        for parent, child in (
            (world.root, cabinet_body),
            (cabinet_body, drawer_body),
            (drawer_body, handle_body),
            (world.root, mug_body),
        ):
            world.add_connection(FixedConnection(parent=parent, child=child))
        handle = Handle(name=PrefixedName("drawer_handle"), root=handle_body)
        drawer = Drawer(
            name=PrefixedName("top_drawer"), root=drawer_body, handle=handle
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


def test_a_mounted_part_is_reported_through_the_field_holding_it():
    """
    What a world actually holds is read through the same fields a mount routes by.
    """
    world = annotated_cabinet_world()
    [cabinet] = world.get_semantic_annotations_by_type(Cabinet)

    parts = [
        relation
        for relation in mounted_relations_of(cabinet)
        if relation.kind is MountKind.PART
    ]

    assert [(relation.field_name, type(relation.target)) for relation in parts] == [
        ("drawers", Drawer)
    ]


def test_a_field_holding_one_part_is_reported_like_a_field_holding_many():
    """
    A caller reads both through one list rather than by knowing which field is which.
    """
    world = annotated_cabinet_world()
    [drawer] = world.get_semantic_annotations_by_type(Drawer)
    [handle] = world.get_semantic_annotations_by_type(Handle)

    [relation] = mounted_relations_of(drawer)

    assert relation.kind is MountKind.PART
    assert relation.field_name == "handle"
    assert relation.target is handle


def test_an_occupant_is_reported_as_contained_rather_than_as_a_part():
    """
    A mug standing in a cabinet is not a structural part of it.
    """
    world = annotated_cabinet_world()
    [cabinet] = world.get_semantic_annotations_by_type(Cabinet)
    [mug] = world.get_semantic_annotations_by_type(Mug)

    contained = [
        relation
        for relation in mounted_relations_of(cabinet)
        if relation.kind is MountKind.CONTAINS
    ]

    assert [(relation.field_name, relation.target) for relation in contained] == [
        ("objects", mug)
    ]


def test_an_annotation_holding_nothing_reports_no_relations():
    """
    An empty field says the world does not hold that relation, not that it might.
    """
    world = annotated_cabinet_world()
    [handle] = world.get_semantic_annotations_by_type(Handle)

    assert mounted_relations_of(handle) == []


# %% which field a mount goes through


def test_the_field_a_whole_holds_a_part_in_is_reported():
    """
    A mount carried out without naming a field can be told afterwards which it used.
    """
    assert field_holding(Cabinet, Drawer).field_name == "drawers"
    assert field_holding(Drawer, Handle).field_name == "handle"


def test_a_part_the_whole_cannot_hold_has_no_field():
    """
    Nothing to report is not the same as reporting the wrong field.
    """
    assert field_holding(Handle, Cabinet) is None


def test_every_field_that_accepts_a_part_is_reported():
    assert [field.field_name for field in fields_accepting(Cabinet, Drawer)] == [
        "drawers"
    ]


def test_no_field_is_reported_for_a_part_the_whole_cannot_hold():
    assert fields_accepting(Handle, Cabinet) == []


def test_the_field_reported_is_the_one_a_mount_actually_routes_to():
    """
    Asserted against the mount itself rather than a second copy of the answer, so the
    two cannot drift apart.
    """
    world = annotated_cabinet_world()
    [cabinet] = world.get_semantic_annotations_by_type(Cabinet)
    [drawer] = world.get_semantic_annotations_by_type(Drawer)

    [mounted] = [
        relation
        for relation in mounted_relations_of(cabinet)
        if relation.target is drawer
    ]

    assert mounted.field_name == field_holding(Cabinet, Drawer).field_name


# %% furniture built from other furniture


def test_a_kitchen_island_holds_the_cabinets_it_is_built_from():
    """
    A kitchen is a run of carcasses under one worktop.

    Without a field for them the carcasses can only be mounted as something they are
    not, or not at all.
    """
    assert field_holding(KitchenIsland, Cabinet).field_name == "cabinets"
    assert field_holding(KitchenIsland, Dishwasher).field_name == "cabinets"


def test_a_dishwasher_is_a_cabinet_so_that_it_can_stand_in_a_run():
    """
    Bounding the units by the case a dishwasher and a cabinet share does not survive
    being stored: a stored class inherits from one parent only, and a cabinet's is its
    furniture rather than its case, so a stored cabinet is not a stored case.

    A dishwasher is therefore a cabinet, as a fridge already is.
    """
    assert issubclass(Dishwasher, Cabinet)
    assert issubclass(Dishwasher, HasCaseAsRootBody)
    assert field_holding(KitchenIsland, Dishwasher).field_name == "cabinets"


def test_a_drawer_built_into_an_island_is_held_as_a_drawer_rather_than_a_unit():
    """
    A scan that never resolved the carcass around a drawer still says something true by
    reporting the drawer, so an island holds one directly.

    It is not a unit: a drawer is not a cabinet, which is also what keeps the two fields
    from both matching it and making the mount the ambiguity ``add`` refuses.
    """
    assert field_holding(KitchenIsland, Drawer).field_name == "drawers"
    assert not issubclass(Drawer, Cabinet)


def test_an_island_holds_its_counter_top_as_a_thing_rather_than_a_surface():
    """
    A supporting surface is the bare region something can be put on.

    The worktop is what a scan sees and what carries the sink, so it is held as itself.
    """
    assert field_holding(KitchenIsland, CounterTop).field_name == "counter_top"
    assert field_holding(CounterTop, Sink).field_name == "sink"


def test_a_kitchen_island_can_be_built_and_mounted_into():
    """
    The relations are only real if a world will carry them out.
    """
    world = World.create_with_root_body("root")
    island_body = Body(name=PrefixedName("island"))
    cabinet_body = Body(name=PrefixedName("a_cabinet"))
    top_body = Body(name=PrefixedName("a_counter_top"))
    with world.modify_world():
        for child in (island_body, cabinet_body, top_body):
            world.add_connection(FixedConnection(parent=world.root, child=child))
        island = KitchenIsland(name=PrefixedName("the_island"), root=island_body)
        cabinet = Cabinet(name=PrefixedName("a_unit"), root=cabinet_body)
        counter_top = CounterTop(name=PrefixedName("the_top"), root=top_body)
        for annotation in (island, cabinet, counter_top):
            world.add_semantic_annotation_recursively(annotation)
    with world.modify_world():
        island.add(cabinet)
        island.add(counter_top)

    held = {
        (relation.field_name, type(relation.target).__name__)
        for relation in mounted_relations_of(island)
    }
    assert held == {("cabinets", "Cabinet"), ("counter_top", "CounterTop")}


# %% the ontology's own classes against everything below its root


def test_a_composed_class_is_below_the_root_without_the_ontology_declaring_it():
    """
    Composing a class registers it below the annotation root for the rest of the
    process.

    Anything asking which classes the ontology *has* would otherwise be told yes about a
    class this process invented a moment ago.
    """
    composed = compose_class("AClassNoModuleDeclares", SemanticAnnotation)

    assert composed.__name__ in annotation_classes(SemanticAnnotation)
    assert composed.__name__ not in declared_annotation_classes(SemanticAnnotation)


def test_the_ontology_declares_fewer_classes_than_derive_from_its_root():
    """
    The declared classes are what the ontology is; the classes below its root are those
    plus whatever else this process has imported or composed.

    Reading the second for the first is what makes an answer depend on import order.
    """
    declared = declared_annotation_classes(SemanticAnnotation)
    below_the_root = annotation_classes(SemanticAnnotation)

    assert "KitchenIsland" in declared
    assert set(declared) <= set(below_the_root)
