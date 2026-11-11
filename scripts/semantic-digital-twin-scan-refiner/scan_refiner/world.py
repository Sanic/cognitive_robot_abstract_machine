"""
Freeze labelled visual and collision surfaces from an actual SemDT world.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import TYPE_CHECKING

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.records import ArrayField, GeometryRole

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% Geometry snapshots
@dataclass
class BodyMesh:
    """
    One body surface expressed in the world's root frame.
    """

    body_name: str
    """Qualified body identity retained from the source world."""
    role: GeometryRole
    """
    Whether the surface describes appearance or collision.
    """

    mesh: trimesh.Trimesh
    """Frozen triangles including shape origins, scales and world kinematics."""


@dataclass
class WorldSnapshot:
    """
    Independent geometry evidence with no dependency on a live world instance.
    """

    meshes: list[BodyMesh]
    """Visual and collision surfaces; empty roles remain absent."""

    @classmethod
    def from_world(cls, world: World) -> WorldSnapshot:
        """
        Capture a validated world without substituting collision for visual geometry.
        """
        world.validate()
        surfaces = []
        for body in world.bodies:
            for role, collection in (
                (GeometryRole.VISUAL, body.visual),
                (GeometryRole.COLLISION, body.collision),
            ):
                if not collection:
                    continue
                mesh = collection.combined_mesh.copy()
                mesh.apply_transform(
                    world.compute_forward_kinematics_np(world.root, body)
                )
                surfaces.append(BodyMesh(str(body.name), role, mesh))
        return cls(surfaces)

    @classmethod
    def from_provider(cls, identifier: str) -> WorldSnapshot:
        """
        Capture a trusted provider class exposing a no-argument create_world method.
        """
        module_name, separator, class_name = identifier.partition(":")
        if not separator or not module_name or not class_name:
            raise InvalidEvidence("Provider must be module.path:WorldClass.")
        provider_class = vars(import_module(module_name))[class_name]
        return cls.from_world(provider_class().create_world())

    def select(self, role: GeometryRole) -> list[BodyMesh]:
        """
        Return exactly the requested representation, without geometry fallback.
        """
        meshes = [surface for surface in self.meshes if surface.role == role]
        if not meshes:
            raise InvalidEvidence(f"World contains no {role} geometry.")
        return meshes

    def samples(self, role: GeometryRole, count: int = 8000) -> np.ndarray:
        """
        Sample the combined surface reproducibly with area-weighted triangles.
        """
        mesh = trimesh.util.concatenate([surface.mesh for surface in self.select(role)])
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
        return points

    def save(self, path: Path) -> None:
        """
        Store numeric mesh arrays without pickle or source asset references.
        """
        if not self.meshes:
            raise InvalidEvidence("World snapshot must contain geometry.")
        np.savez_compressed(
            path,
            **{
                ArrayField.VERTICES: np.concatenate(
                    [item.mesh.vertices for item in self.meshes]
                ),
                ArrayField.FACES: np.concatenate(
                    [item.mesh.faces for item in self.meshes]
                ),
                ArrayField.BODY_NAMES: np.array(
                    [item.body_name for item in self.meshes]
                ),
                ArrayField.GEOMETRY_ROLES: np.array(
                    [str(item.role) for item in self.meshes]
                ),
                ArrayField.VERTEX_OFFSETS: np.cumsum(
                    [0] + [len(item.mesh.vertices) for item in self.meshes]
                ),
                ArrayField.FACE_OFFSETS: np.cumsum(
                    [0] + [len(item.mesh.faces) for item in self.meshes]
                ),
            },
        )

    @classmethod
    def read(cls, path: Path) -> WorldSnapshot:
        """
        Restore exact frozen meshes without resolving packages or invoking a provider.
        """
        meshes = []
        with np.load(path, allow_pickle=False) as arrays:
            vertices = arrays[ArrayField.VERTICES]
            faces = arrays[ArrayField.FACES]
            roles = arrays[ArrayField.GEOMETRY_ROLES]
            vertex_offsets = arrays[ArrayField.VERTEX_OFFSETS]
            face_offsets = arrays[ArrayField.FACE_OFFSETS]
            for index, body_name in enumerate(arrays[ArrayField.BODY_NAMES]):
                mesh = trimesh.Trimesh(
                    vertices=vertices[
                        vertex_offsets[index] : vertex_offsets[index + 1]
                    ],
                    faces=faces[face_offsets[index] : face_offsets[index + 1]],
                    process=False,
                )
                meshes.append(
                    BodyMesh(
                        str(body_name),
                        GeometryRole(roles[index]),
                        mesh,
                    )
                )
        return cls(meshes)
