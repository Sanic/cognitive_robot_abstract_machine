from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    SemanticEnvironmentAnnotation,
)


@dataclass(eq=False)
class Countertop(HasRootBody):
    pass


@dataclass(eq=False)
class Stovetop(HasRootBody):
    pass


@dataclass(eq=False)
class Side(HasRootBody):
    pass


@dataclass(eq=False)
class Back(SemanticEnvironmentAnnotation):
    pass


@dataclass(eq=False)
class Things(SemanticEnvironmentAnnotation):
    pass


@dataclass(eq=False)
class Board(HasRootBody):
    pass


@dataclass(eq=False)
class Tap(SemanticAnnotation):
    pass
