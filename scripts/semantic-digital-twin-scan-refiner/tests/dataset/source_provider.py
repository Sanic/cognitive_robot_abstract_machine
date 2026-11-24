"""
A trusted Python world whose semantic annotation must survive source refinement.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_source import WorldSource
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)


# %% Semantic source fixture
@dataclass(kw_only=True)
class FurnitureWorld(WorldSource):
    """
    Create furniture geometry with a semantic observation attached to the cabinet.
    """

    def create_world(self) -> World:
        world = URDFParser.from_file(
            file_path=str(Path(__file__).with_name("source_world.urdf"))
        ).create_world()
        with world.modify_world():
            world.add_semantic_annotation(
                NaturalLanguageDescription(
                    root=world.get_body_by_name(
                        PrefixedName(prefix="test", name="box")
                    ),
                    description="Fixture cabinet",
                )
            )
        return world
