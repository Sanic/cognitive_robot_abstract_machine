"""
Freeze labelled visual and collision surfaces from an actual SemDT world.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import json

import numpy as np
import trimesh
from typing_extensions import TYPE_CHECKING
from krrood.adapters.json_serializer import DataclassJSONSerializer, to_json

from scan_refiner.assemblies import AssemblyMembership
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.records import ArrayField, GeometryRole

if TYPE_CHECKING:
    from scan_refiner.dimensions import BodyDimensions
    from scan_refiner.refinement import BodyTranslation
    from semantic_digital_twin.world import World


# %% Geometry snapshots
@dataclass(kw_only=True)
class KinematicNode:
    """
    A world entity and its parent in the captured kinematic tree.
    """

    name: str
    """
    Qualified identity shared with any corresponding mesh.
    """

    parent_name: str | None
    """
    Qualified identity of the parent, or None for the world root.
    """


@dataclass(kw_only=True)
class BodyMesh:
    """
    One body surface expressed in the world's root frame.
    """

    body_name: str
    """
    Qualified body identity retained from the source world.
    """

    role: GeometryRole
    """
    Whether the surface describes appearance or collision.
    """

    mesh: trimesh.Trimesh
    """
    Frozen triangles including shape origins, scales and world kinematics.
    """


@dataclass(kw_only=True)
class WorldSnapshot:
    """
    Independent geometry evidence with no dependency on a live world instance.
    """

    meshes: list[BodyMesh]
    """
    Visual and collision surfaces; empty roles remain absent.
    """

    assemblies: list[AssemblyMembership] = field(default_factory=list, kw_only=True)
    """
    Semantic groups captured with the geometry baseline.
    """

    kinematic_nodes: list[KinematicNode] = field(default_factory=list, kw_only=True)
    """
    Topology retained even for entities with no geometry.
    """

    @classmethod
    def from_world(cls, *, world: World) -> WorldSnapshot:
        """
        Capture a validated world without substituting collision for visual geometry.
        """
        world.validate()
        root = world.root
        surfaces = []
        for body in world.bodies:
            for role, collection in (
                (GeometryRole.VISUAL, body.visual),
                (GeometryRole.COLLISION, body.collision),
            ):
                if not collection:
                    continue
                mesh = collection.combined_mesh.copy()
                mesh.apply_transform(world.compute_forward_kinematics_np(root, body))
                surfaces.append(
                    BodyMesh(body_name=str(body.name), role=role, mesh=mesh)
                )
        return cls(
            meshes=surfaces,
            assemblies=AssemblyMembership.discover(
                world=world,
                geometric_bodies={surface.body_name for surface in surfaces},
            ),
            kinematic_nodes=(
                [
                    KinematicNode(name=str(root.name), parent_name=None),
                    *(
                        KinematicNode(
                            name=str(connection.child.name),
                            parent_name=str(connection.parent.name),
                        )
                        for connection in sorted(
                            world.connections, key=lambda edge: str(edge.child.name)
                        )
                    ),
                ]
                if root is not None
                else []
            ),
        )

    @classmethod
    def from_provider(cls, *, identifier: str) -> WorldSnapshot:
        """
        Capture a trusted provider class exposing a no-argument create_world method.
        """
        module_name, separator, class_name = identifier.partition(":")
        if not separator or not module_name or not class_name:
            raise InvalidEvidence(reason="Provider must be module.path:WorldClass.")
        provider_class = vars(import_module(module_name))[class_name]
        return cls.from_world(world=provider_class().create_world())

    def select(self, *, role: GeometryRole) -> list[BodyMesh]:
        """
        Return exactly the requested representation, without geometry fallback.
        """
        meshes = [surface for surface in self.meshes if surface.role == role]
        if not meshes:
            raise InvalidEvidence(reason=f"World contains no {role} geometry.")
        return meshes

    def body_mesh(self, *, body_name: str, role: GeometryRole) -> BodyMesh:
        """
        Combine all parts of one body representation without dropping compound geometry.
        """
        surfaces = [
            surface.mesh
            for surface in self.select(role=role)
            if surface.body_name == body_name
        ]
        if not surfaces:
            raise InvalidEvidence(
                reason="The selected body has no geometry in this role."
            )
        return BodyMesh(
            body_name=body_name, role=role, mesh=trimesh.util.concatenate(surfaces)
        )

    def apply_dimensions(self, *, overrides: list[BodyDimensions]) -> None:
        """
        Apply each body's common dimension stretch to every present geometry role.
        """
        dimensions = {item.body_name: item for item in overrides}
        for surface in self.meshes:
            if surface.body_name in dimensions:
                override = dimensions[surface.body_name]
                surface.mesh.vertices = (
                    surface.mesh.vertices @ override.linear.T + override.offset
                )

    def apply_translations(self, *, overrides: list[BodyTranslation]) -> None:
        """
        Displace named bodies in this snapshot while preserving every geometry role.
        """
        translations = {item.body_name: item.translation for item in overrides}
        for surface in self.meshes:
            if surface.body_name in translations:
                surface.mesh.apply_translation(translations[surface.body_name])

    def samples(self, *, role: GeometryRole, count: int = 8000) -> np.ndarray:
        """
        Sample the combined surface reproducibly with area-weighted triangles.
        """
        mesh = trimesh.util.concatenate(
            [surface.mesh for surface in self.select(role=role)]
        )
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
        return points

    def save(self, *, path: Path) -> None:
        """
        Store numeric mesh arrays without pickle or source asset references.
        """
        if not self.meshes:
            raise InvalidEvidence(reason="World snapshot must contain geometry.")
        metadata = {
            name: np.array(json.dumps(to_json(items)))
            for name, items in (
                (ArrayField.ASSEMBLIES, self.assemblies),
                (ArrayField.KINEMATIC_NODES, self.kinematic_nodes),
            )
            if items
        }
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
            **metadata,
        )

    @classmethod
    def read(cls, *, path: Path) -> WorldSnapshot:
        """
        Restore exact frozen meshes without resolving packages or invoking a provider.
        """
        meshes = []
        with np.load(path, allow_pickle=False) as arrays:
            assemblies = (
                [
                    DataclassJSONSerializer.from_json(
                        data=item, clazz=AssemblyMembership
                    )
                    for item in json.loads(arrays[ArrayField.ASSEMBLIES].item())
                ]
                if ArrayField.ASSEMBLIES in arrays
                else []
            )
            kinematic_nodes = (
                [
                    DataclassJSONSerializer.from_json(data=item, clazz=KinematicNode)
                    for item in json.loads(arrays[ArrayField.KINEMATIC_NODES].item())
                ]
                if ArrayField.KINEMATIC_NODES in arrays
                else []
            )
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
                        body_name=str(body_name),
                        role=GeometryRole(roles[index]),
                        mesh=mesh,
                    )
                )
        return cls(
            meshes=meshes, assemblies=assemblies, kinematic_nodes=kinematic_nodes
        )
