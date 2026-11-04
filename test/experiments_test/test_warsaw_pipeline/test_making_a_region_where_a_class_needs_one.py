"""
Giving a region-rooted class the region it takes, built from the body it was answered
about -- where the run was told to.

A reconstruction gives every object a body, so a class rooted on a region instead has
nowhere to stand and is refused. That refusal is not free: a window in a wall is the
commonest part-whole relation an interior has, and every one of them was being dropped
at the last step. A region the size and pose of the measured body is what bridges the
two, and it costs no geometry to try, because the cut is the mount's business and
happens only where the ontology says the relation removes it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.warsaw.pipeline.provenance import settings_to_json
from experiments.warsaw.pipeline.records import BodyAnswer, Classifications
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.scene_split import Pairing
from experiments.warsaw.pipeline.steps.annotate import (
    ROOT,
    AnnotationFromBody,
    MountAnnotations,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Room,
    Wall,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body, Region

# %% a wall with a window standing in it

WINDOW = "window"
"""
The body a region is derived from in these tests.
"""

WALL = "wall"
"""
The body that holds it.
"""

APERTURES = "apertures"
"""
The field a wall holds an aperture in, and the one the ontology says removes the part's
volume from the whole.
"""

WINDOW_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 1.5, 1.0)
"""
Where the window stands, away from the origin, which is the only place a region put at
the body's pose can be told apart from one left at the world's.
"""


@pytest.fixture
def wall_and_window() -> World:
    """
    :return: A world holding a wall and, standing away from the origin, a window.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        Wall.create_with_new_body_in_world(
            name=WALL, scale=Scale(0.2, 4.0, 2.5), world=world
        )
        Wall.create_with_new_body_in_world(
            name=WINDOW,
            scale=Scale(0.2, 1.0, 1.0),
            world=world,
            world_root_T_self=WINDOW_POSE,
        )
    return world


def answered(**classes: str) -> Classifications:
    """
    :param classes: What each body, by name, was answered to be.
    :return: Those answers, as the step reads them.
    """
    return Classifications(
        scene="",
        model="",
        bodies=[
            BodyAnswer(name=name, class_name=class_name)
            for name, class_name in classes.items()
        ],
    )


def told_to(tmp_path: Path, *, making_regions: bool) -> MountAnnotations:
    """
    :param tmp_path: A directory for the run.
    :param making_regions: What the run was told about classes needing a region.
    :return: The step, reading that back out of the run's own files.
    """
    run = Run(directory=tmp_path)
    run.write_json(
        RunFile.PROVENANCE,
        {
            "settings": settings_to_json(
                PipelineSettings(
                    skip_classes_a_body_cannot_make=True,
                    make_a_region_where_a_class_needs_one=making_regions,
                )
            )
        },
    )
    return MountAnnotations(directory=tmp_path)


# %% what the run can supply


def test_a_region_rooted_class_is_short_of_its_root_when_no_region_may_be_made():
    """
    The refusal the three HM3D rooms were losing their windows to.
    """
    assert AnnotationFromBody().cannot_supply(Aperture) == ["root"]


def test_a_region_rooted_class_is_short_of_nothing_when_a_region_may_be_made():
    """
    A region built from the body is the root such a class takes, so nothing is left
    missing and the class is made rather than skipped.
    """
    assert AnnotationFromBody(may_make_a_region=True).cannot_supply(Aperture) == []


def test_making_a_region_rescues_nothing_a_region_is_not():
    """
    A room is its floor and is rooted on neither a body nor a region, so permission to
    build one leaves it exactly as short as it was. Only a region-rooted class's root is
    bridged, and only the root.
    """
    assert AnnotationFromBody(may_make_a_region=True).cannot_supply(Room) == [
        "floor",
        ROOT,
    ]


# %% the step


def test_the_scans_are_left_as_they_were():
    """
    Making regions is not the default, so a scan answers as it always has and its
    numbers stay comparable with the ones already reported.
    """
    assert PipelineSettings().make_a_region_where_a_class_needs_one is False


def test_a_window_is_left_alone_when_the_run_was_not_told_to_make_regions(
    wall_and_window, tmp_path: Path
):
    """
    The behaviour the three rooms already measured, unchanged.
    """
    annotated = told_to(tmp_path, making_regions=False).annotate(
        wall_and_window, answered(**{WINDOW: Aperture.__name__})
    )
    assert annotated == {}


def test_a_window_is_annotated_over_a_region_when_the_run_was_told_to(
    wall_and_window, tmp_path: Path
):
    """
    The body is measured, the region is what the class takes, and the annotation is made
    rather than counted as a loss.
    """
    annotated = told_to(tmp_path, making_regions=True).annotate(
        wall_and_window, answered(**{WINDOW: Aperture.__name__})
    )
    assert [type(one).__name__ for one in annotated.values()] == [Aperture.__name__]
    assert isinstance(annotated[WINDOW].root, Region)


def test_the_region_stands_where_the_body_it_was_built_from_stands(
    wall_and_window, tmp_path: Path
):
    """
    A region left at the world's origin would describe whatever stands there instead,
    and a mount that cuts would cut the wrong hole.
    """
    annotated = told_to(tmp_path, making_regions=True).annotate(
        wall_and_window, answered(**{WINDOW: Aperture.__name__})
    )
    wall_and_window.update_forward_kinematics()
    assert np.allclose(
        annotated[WINDOW].root.global_transform.to_np(), WINDOW_POSE.to_np()
    )


def test_a_body_rooted_class_is_still_annotated_over_its_own_body(
    wall_and_window, tmp_path: Path
):
    """
    The bridge is reached only by a class that needs it; everything else is made from
    the body exactly as before.
    """
    annotated = told_to(tmp_path, making_regions=True).annotate(
        wall_and_window, answered(**{WALL: Wall.__name__})
    )
    assert isinstance(annotated[WALL].root, Body)


# %% the relation the refusal was costing


def test_a_window_is_mounted_into_its_wall_and_cuts_it(wall_and_window, tmp_path: Path):
    """
    The whole point of building the region: the measurement already finds a window in a
    wall, and this is what carries that relation into the world instead of dropping it.

    The cut follows, because the ontology says an aperture removes its volume from what
    it is an aperture of, and the wall's single box becomes the pieces left around the
    hole.
    """
    step = told_to(tmp_path, making_regions=True)
    annotated = step.annotate(
        wall_and_window,
        answered(**{WALL: Wall.__name__, WINDOW: Aperture.__name__}),
    )
    before = len(annotated[WALL].root.collision)

    mounted = step.mount(
        wall_and_window,
        annotated,
        [Pairing(whole=WALL, part=WINDOW, field_name=APERTURES)],
    )

    assert mounted.refused == []
    assert mounted.carried_out == 1
    assert annotated[WINDOW] in annotated[WALL].apertures
    assert len(annotated[WALL].root.collision) > before


def test_the_same_mount_is_refused_when_no_region_was_made(
    wall_and_window, tmp_path: Path
):
    """
    The loss this setting exists to undo: with no region to be an aperture over, the
    window has no annotation and the pairing has nothing to mount.
    """
    step = told_to(tmp_path, making_regions=False)
    annotated = step.annotate(
        wall_and_window,
        answered(**{WALL: Wall.__name__, WINDOW: Aperture.__name__}),
    )

    mounted = step.mount(
        wall_and_window,
        annotated,
        [Pairing(whole=WALL, part=WINDOW, field_name=APERTURES)],
    )

    assert mounted.carried_out == 0
    assert [one.reason for one in mounted.refused] == ["one end has no annotation"]
