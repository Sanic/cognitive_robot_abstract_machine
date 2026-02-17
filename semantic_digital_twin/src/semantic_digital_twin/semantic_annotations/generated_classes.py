from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasApertures, HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cloth, Decor, Furniture
from semantic_digital_twin.robots.abstract_robot import Sensor
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

@dataclass(eq=False)
class Ceiling(HasRootBody):
    pass


@dataclass(eq=False)
class Television(Sensor):
    pass


@dataclass(eq=False)
class Blanket(Cloth):
    pass


@dataclass(eq=False)
class Window(HasApertures):
    pass


@dataclass(eq=False)
class Chandelier(Decor):
    pass


@dataclass(eq=False)
class Pad(HasRootBody):
    pass


@dataclass(eq=False)
class Nightstand(Furniture):
    pass


@dataclass(eq=False)
class TableLamp(Decor):
    pass


@dataclass(eq=False)
class PlushToy(HasRootBody):
    pass


@dataclass(eq=False)
class Pillow(SemanticAnnotation):
    pass


@dataclass(eq=False)
class WindowFrame(SemanticAnnotation):
    pass
