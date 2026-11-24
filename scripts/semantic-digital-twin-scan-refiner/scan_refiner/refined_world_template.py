"""
Reload an exported refined world from its own source bundle.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path
from scan_refiner.source_application import (
    ReviewedWorldSource,
    SourceApplication,
    SourceArtifact,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_source import WorldSource
from semantic_digital_twin.adapters.urdf import URDFParser


# %% Exported world provider
@dataclass(kw_only=True)
class RefinedWorld(WorldSource):
    """
    Create a live world from the validated source packaged beside this provider.
    """

    def create_world(self) -> World:
        directory = Path(__file__).parent
        if (directory / SourceArtifact.URDF).is_file():
            return URDFParser.from_file(
                file_path=str(directory / SourceArtifact.URDF),
                path_resolver=SourceApplication.bundle_resolver(directory=directory),
            ).create_world()
        return ReviewedWorldSource(directory=directory).create_world()
