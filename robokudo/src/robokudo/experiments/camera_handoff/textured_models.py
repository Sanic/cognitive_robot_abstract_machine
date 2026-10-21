# %% imports

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from math import pi
from pathlib import Path

import numpy as np

from krrood.patterns.path_enumeration import PathEnumeration
from robokudo.experiments.camera_handoff.exceptions import InvalidExperiment
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Mesh, Scale

# %% repository assets


class ModelDirectory(PathEnumeration):
    """
    Repository locations of the textured episode assets.
    """

    EPISODE = "segmind/resources/fame_episodes/alessandro_sliding_bueno/models"


class ModelAsset(StrEnum):
    """
    Textured household objects available in the recorded episode.
    """

    CRACKER_BOX = "obj_000001"
    CUP = "obj_000004"

    @cached_property
    def preview_filename(self) -> str:
        """
        Filename of the camera preview recorded for this object.
        """
        return f"{self.value}.png"


@dataclass
class TexturedModel:
    """
    A textured episode mesh with an explicit metric conversion.
    """

    asset: ModelAsset = ModelAsset.CRACKER_BOX
    """
    Object selected from the episode's mesh collection.
    """

    directory: Path = ModelDirectory.EPISODE
    """
    Directory containing the PLY mesh and its referenced texture.
    """

    meters_per_unit: float = 0.001
    """
    Metric conversion matching the existing episode player.
    """

    yaw: float = pi / 2
    """
    Model rotation around world z before tabletop placement.
    """

    def load(self) -> Mesh:
        """
        Load the metric model while preserving the PLY's UV coordinates.
        """
        filename = self.directory / f"{self.asset}.ply"
        if (
            not filename.is_file()
            or not (self.directory / self.asset.preview_filename).is_file()
        ):
            raise InvalidExperiment(f"Model or texture is missing beside {filename}.")
        shape = Mesh(
            filename=str(filename.resolve()),
            scale=Scale(
                self.meters_per_unit, self.meters_per_unit, self.meters_per_unit
            ),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(yaw=self.yaw),
        )
        if shape.mesh.visual.kind != "texture":
            raise InvalidExperiment(f"Model {filename} did not retain its texture.")
        return shape

    def measure_bounds(self, shape: Mesh) -> np.ndarray:
        """
        Return model bounds after its placement rotation, in meters.
        """
        vertices = shape.mesh.vertices @ shape.origin.to_np()[:3, :3].T
        return np.array([vertices.min(axis=0), vertices.max(axis=0)])
