from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh
from typing_extensions import Any

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.part_finding.splitting import BodyCutter, ShapePiece
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% scene files


class ScanNetPPSceneFile(StrEnum):
    """
    The files of one ScanNet++ scene, as paths relative to the scene's directory.
    """

    MESH = "scans/mesh_aligned_0.05.ply"
    """
    The scanned room as one vertex-coloured triangle mesh, in meters.
    """

    SEGMENTS = "scans/segments.json"
    """
    The segment each mesh vertex belongs to.
    """

    SEGMENT_ANNOTATIONS = "scans/segments_anno.json"
    """
    The annotated objects, each a group of segments.
    """


# %% annotations


@dataclass
class OrientedBoundingBox:
    """
    A box around an annotated object, rotated to fit the object.
    """

    centroid: np.ndarray
    """
    The center of the box, in the scene's frame.
    """

    normalized_axes: np.ndarray
    """
    The box's three unit axes in the scene's frame, one axis per row.
    """

    axes_lengths: np.ndarray
    """
    The box's side length along each of :attr:`normalized_axes`.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OrientedBoundingBox:
        """
        :param data: The ``obb`` entry of a segment group in ``segments_anno.json``.
        :return: The box the entry describes.
        """
        return cls(
            centroid=np.asarray(data["centroid"]),
            normalized_axes=np.asarray(data["normalizedAxes"]).reshape(3, 3),
            axes_lengths=np.asarray(data["axesLengths"]),
        )


@dataclass
class ScanNetPPObject:
    """
    An object annotated in a ScanNet++ scene.
    """

    object_id: int
    """
    The object's identifier, unique within its scene.
    """

    label: str
    """
    The annotator's free-text name for the object, e.g. ``"kitchen cabinet"``.
    """

    segment_ids: np.ndarray
    """
    The mesh segments the object covers.
    """

    oriented_bounding_box: OrientedBoundingBox
    """
    The box around the object.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ScanNetPPObject:
        """
        :param data: One entry of ``segGroups`` in ``segments_anno.json``.
        :return: The object the entry describes.
        """
        return cls(
            object_id=data["objectId"],
            label=data["label"],
            segment_ids=np.asarray(data["segments"]),
            oriented_bounding_box=OrientedBoundingBox.from_json(data["obb"]),
        )


# %% scene


@dataclass
class ScanNetPPScene:
    """
    A ScanNet++ scene as recorded: the scanned mesh and the objects annotated on it.
    """

    scene_id: str
    """
    The scene's identifier, e.g. ``"f9f95681fd"``.
    """

    mesh: trimesh.Trimesh
    """
    The scanned room, with its vertices in the order the segments refer to.
    """

    mesh_path: Path
    """
    The file :attr:`mesh` was read from.
    """

    vertex_segment_ids: np.ndarray
    """
    The segment of each vertex of :attr:`mesh`.
    """

    objects: list[ScanNetPPObject]
    """
    The annotated objects.
    """

    def vertex_indices(self, scannetpp_object: ScanNetPPObject) -> np.ndarray:
        """
        :param scannetpp_object: An object annotated in this scene.
        :return: The indices of the vertices of :attr:`mesh` that belong to the object.
        """
        return np.flatnonzero(
            np.isin(self.vertex_segment_ids, scannetpp_object.segment_ids)
        )

    def create_world(self) -> World:
        """
        Build a world with one body per annotated object, fixed to the root body,
        ``scan``, which keeps the surface no object covers.

        Each object's body shows the faces of :attr:`mesh` whose three vertices all
        belong to the object, and every body is shown in a color of its own. Faces
        spanning several objects stay with ``scan``. Every object's body carries a
        :class:`~semantic_digital_twin.semantic_annotations.natural_language.NaturalLanguageDescription`
        of its label.

        :return: The built world.
        """
        face_owners = self._face_owners()
        pieces = [
            ShapePiece(
                body=Body(
                    name=PrefixedName(
                        f"{scannetpp_object.label}_{scannetpp_object.object_id}"
                    )
                ),
                faces=np.flatnonzero(face_owners == index),
            )
            for index, scannetpp_object in enumerate(self.objects)
        ]
        world = self.create_world_cut_into(pieces)
        colors = Color.distinct_colors(len(pieces) + 1)
        with world.modify_world():
            # Setting the color is enough: the pieces' meshes are not built yet, and
            # are colored when they are; dyeing them would load every piece now.
            for shape in world.root.visual:
                shape.color = colors[-1]
            for piece, scannetpp_object, color in zip(pieces, self.objects, colors):
                for shape in piece.body.visual:
                    shape.color = color
                world.add_semantic_annotation(
                    NaturalLanguageDescription(
                        root=piece.body, description=scannetpp_object.label
                    )
                )
        return world

    def create_scan_world(self) -> World:
        """
        :return: A world whose only body, ``scan``, shows the whole of :attr:`mesh`, with
            its faces in the order the annotations refer to.
        """
        scan = self._scan_body()
        world = World()
        with world.modify_world():
            world.add_body(scan)
        return world

    def create_world_cut_into(self, pieces: list[ShapePiece]) -> World:
        """
        :param pieces: Faces of :attr:`mesh`, in the order the annotations refer to, each
            with the body they are cut into.
        :return: A world whose root body, ``scan``, shows the faces of :attr:`mesh` no
            piece takes, with the body of each piece fixed to it.
        """
        scan = self._scan_body()
        cutter = BodyCutter()
        cut = cutter.cut(scan, cutter.shape_to_cut(scan), pieces)
        world = World()
        with world.modify_world():
            cut.add_to_world(world)
        return world

    def _scan_body(self) -> Body:
        """
        :return: The body ``scan``, in no world yet, showing the whole of :attr:`mesh`.
            It is added with its shape: changing the shapes of a body in a world records
            the whole body serialized before and after the change.
        """
        scan = Body(name=PrefixedName("scan"))
        surface = Mesh(
            filename=str(self.mesh_path),
            origin=HomogeneousTransformationMatrix(reference_frame=scan),
        )
        # Reuse the mesh read without processing instead of reading the file again.
        surface.mesh = self.mesh
        scan.visual = ShapeCollection([surface], reference_frame=scan)
        return scan

    def _face_owners(self) -> np.ndarray:
        """
        :return: For each face of :attr:`mesh`, the index into :attr:`objects` of the
            object owning all three of its vertices, or ``len(objects)`` where no single
            object does.
        """
        unannotated = len(self.objects)
        vertex_owners = np.full(len(self.mesh.vertices), unannotated)
        for index, scannetpp_object in enumerate(self.objects):
            vertex_owners[self.vertex_indices(scannetpp_object)] = index
        face_vertex_owners = vertex_owners[self.mesh.faces]
        has_single_owner = (face_vertex_owners == face_vertex_owners[:, :1]).all(axis=1)
        return np.where(has_single_owner, face_vertex_owners[:, 0], unannotated)
