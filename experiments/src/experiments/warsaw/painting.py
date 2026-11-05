"""
Painting a world's bodies so that one can be told from another.

A world published in the colours it was scanned or modelled in is one colour: a grey
room in which nothing can be picked out. Painting it by what each body *is*, or simply
by which body it is, is what makes a viewer worth opening.

Only the world held in memory is painted. Worlds reach a viewer read afresh from a
database or built afresh from a file, and are never written back, so nothing here
reaches what was recorded.
"""

from __future__ import annotations

import colorsys
from enum import StrEnum

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color

from experiments.warsaw.exceptions import TooManyToPaintError
from typing_extensions import Dict, List

# %% colours that can be told apart

GOLDEN_ANGLE = 0.381966
"""
How far around the colour wheel to step for each next colour.

The golden angle as a fraction of a turn. Stepping by it never revisits a hue and keeps
consecutive ones far apart, so a palette of any size comes out spread rather than
bunched, without knowing beforehand how many colours are wanted.
"""

SATURATION = 0.65
"""
How strong the colours are.

Full saturation glares against a viewer's grey ground.
"""

BRIGHTNESS = 0.95
"""
How light the colours are, kept high so dark shapes stay legible.
"""

UNNAMED_GREY = Color.from_rgba(red=0.55, green=0.55, blue=0.55)
"""
What a body carrying no class is painted, so it is plainly one that was never named.
"""

UNNAMED = "(no class)"
"""
Stands in for the class of a body nothing named.
"""


def spread_colours(count: int) -> List[Color]:
    """
    Make colours that are easy to tell apart.

    :param count: How many are wanted.
    :return: That many, spread around the colour wheel.
    """
    return [
        Color(
            *colorsys.hsv_to_rgb((index * GOLDEN_ANGLE) % 1.0, SATURATION, BRIGHTNESS)
        )
        for index in range(count)
    ]


# %% colours a model is told the names of


class PaintColor(StrEnum):
    """
    A colour a body is painted in a picture a model is shown, and named in the question.

    Every name is a plain colour word that names no object, so a model reading that
    something is painted "tomato" or "olive" cannot take the paint for the thing. The
    members are ordered so that the first few are the furthest apart.
    """

    RED = "#e6261f"
    """
    Red.
    """

    BLUE = "#2a63d6"
    """
    Blue.
    """

    GREEN = "#2fa84f"
    """
    Green.
    """

    YELLOW = "#f7d000"
    """
    Yellow.
    """

    PURPLE = "#8e3cc8"
    """
    Purple.
    """

    CYAN = "#14c8dc"
    """
    Cyan.
    """

    MAGENTA = "#e000c8"
    """
    Magenta.
    """

    BROWN = "#8c5014"
    """
    Brown.
    """

    PINK = "#ff8cb4"
    """
    Pink.
    """

    NAVY = "#001e6e"
    """
    Navy.
    """

    @classmethod
    def distinct(cls, count: int) -> List[PaintColor]:
        """
        :param count: How many objects are to be painted apart.
        :return: That many colours, each a different one.
        :raises TooManyToPaintError: If there are more objects than colours.
        """
        if count > len(cls):
            raise TooManyToPaintError(wanted=count, available=len(cls))
        return list(cls)[:count]

    @property
    def color(self) -> Color:
        """
        :return: The colour, to paint with.
        """
        return Color.from_hex(self.value)

    @property
    def word(self) -> str:
        """
        :return: The colour's name, as a model is told it.
        """
        return self._name_.lower()


# %% what the bodies are painted by


class Coloring(StrEnum):
    """
    What a world's bodies are painted by.
    """

    BY_CLASS = "by_class"
    """
    One colour per class, so what each body is can be read off at a glance.
    """

    BY_BODY = "by_body"
    """
    One colour per body, so two of the same class standing side by side can still be
    told apart.
    """

    UNCHANGED = "unchanged"
    """
    Left as the world carries them.
    """


def class_of_every_body(world: World) -> Dict[str, str]:
    """
    Say what class each body carries.

    :param world: The world to read.
    :return: The class of every body rooted at an annotation, by the body's name.
    """
    return {
        str(annotation.root.name): type(annotation).__name__
        for annotation in world.semantic_annotations
        if isinstance(annotation, HasRootKinematicStructureEntity)
    }


def paint(world: World, coloring: Coloring) -> Dict[str, Color]:
    """
    Paint a world's bodies so they can be told apart.

    :param world: The world to paint, changed in place.
    :param coloring: What to paint them by.
    :return: What each name was painted, for saying afterwards, empty where nothing was
        painted.
    """
    if coloring is Coloring.UNCHANGED:
        return {}
    named = class_of_every_body(world)

    def key_of(body) -> str:
        """
        :param body: The body being painted.
        :return: What decides its colour.
        """
        if coloring is Coloring.BY_CLASS:
            return named.get(str(body.name), UNNAMED)
        return str(body.name)

    keys = sorted({key_of(body) for body in world.bodies})
    painted = dict(zip(keys, spread_colours(len(keys))))
    painted[UNNAMED] = UNNAMED_GREY
    for body in world.bodies:
        for shapes in (body.visual, body.collision):
            shapes.dye_shapes(painted[key_of(body)])
    return painted
