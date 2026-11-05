"""
Carrying the decided mounts out: every one is a structural part, held in the field it
was measured for.
"""

from __future__ import annotations

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.warsaw.pipeline.steps.annotate import MountAnnotations
from experiments.warsaw.scene_split import Pairing

# %% a world of annotations nothing is mounted into yet


def drawers_and_handles() -> tuple[World, dict]:
    """
    :return: A world holding two drawers and two handles, each annotated and none
        mounted, and the annotations by body name.
    """
    world = World.create_with_root_body("root")
    names = ("drawer_1", "drawer_2", "handle_1", "handle_2")
    bodies = {name: Body(name=PrefixedName(name)) for name in names}
    with world.modify_world():
        for body in bodies.values():
            world.add_connection(FixedConnection(parent=world.root, child=body))
        annotations = {
            name: (Drawer if name.startswith("drawer") else Handle)(
                name=PrefixedName(f"the_{name}"), root=bodies[name]
            )
            for name in names
        }
        for annotation in annotations.values():
            world.add_semantic_annotation(annotation)
    return world, annotations


# %% carrying a mount out


def test_a_structural_part_is_mounted_into_the_field_it_was_measured_for(tmp_path):
    world, annotations = drawers_and_handles()

    MountAnnotations(directory=tmp_path).mount(
        world,
        annotations,
        [Pairing(whole="drawer_1", part="handle_1", field_name="handles")],
    )

    assert annotations["drawer_1"].handles == [annotations["handle_1"]]


def test_a_part_goes_where_its_class_fits_whatever_field_the_pairing_named(tmp_path):
    """
    The pairing named its field from what the labels were read as; the bodies may then
    be classified otherwise, and the part belongs in the field its class fits.
    """
    world, annotations = drawers_and_handles()

    mounted = MountAnnotations(directory=tmp_path).mount(
        world,
        annotations,
        [Pairing(whole="drawer_1", part="handle_1", field_name="knobs")],
    )

    assert mounted.refused == []
    assert annotations["drawer_1"].handles == [annotations["handle_1"]]


def test_a_whole_that_will_not_hold_the_part_is_recorded_as_refusing_it(tmp_path):
    """
    The class a body was given may simply not admit the part another step said it holds,
    and that is an answer about this pairing rather than a run that should not have got
    this far.
    """
    world, annotations = drawers_and_handles()
    pairing = Pairing(whole="drawer_1", part="drawer_2", field_name="")

    mounted = MountAnnotations(directory=tmp_path).mount(world, annotations, [pairing])

    assert [refused.pairing for refused in mounted.refused] == [pairing]


def test_a_whole_classified_as_something_that_holds_no_parts_is_refused(tmp_path):
    """
    The pairing was made from what a label was read as, and the body may then be named
    as a class that holds nothing, such as a handle; that costs the one mount, not the
    run.
    """
    world, annotations = drawers_and_handles()
    held = Pairing(whole="handle_1", part="handle_2", field_name="handles")
    kept = Pairing(whole="drawer_1", part="handle_1", field_name="handles")

    mounted = MountAnnotations(directory=tmp_path).mount(
        world, annotations, [held, kept]
    )

    assert [refused.pairing for refused in mounted.refused] == [held]
    assert annotations["drawer_1"].handles == [annotations["handle_1"]]
