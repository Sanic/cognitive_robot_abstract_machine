from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import Any, Type

from semantic_digital_twin.adapters.scannetpp_dataset.schema import ScanNetPPScene
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.part_finding.splitting import ShapePiece
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

# %% annotation files


class Articulate3DAnnotationFile(StrEnum):
    """
    The annotation files Articulate3D has for every scan.
    """

    PARTS = "parts.json"
    """
    The parts of the scan's mesh.
    """

    ARTICULATIONS = "artic.json"
    """
    How the movable parts move.
    """

    def path(self, directory: Path, scene_id: str) -> Path:
        """
        :param directory: The Articulate3D dataset directory.
        :param scene_id: The ScanNet++ scene the annotations belong to.
        :return: The path of this file for the scene.
        """
        return directory / f"{scene_id}_{self.value}"

    def scene_ids(self, directory: Path) -> list[str]:
        """
        :param directory: The Articulate3D dataset directory.
        :return: The scenes the directory holds this file for, in name order.
        """
        suffix = f"_{self.value}"
        return sorted(
            path.name.removesuffix(suffix) for path in directory.glob(f"*{suffix}")
        )


# %% parts


@dataclass
class Articulate3DPart:
    """
    A part of an annotated object, e.g. a cabinet door, as a set of faces of the scan's
    mesh.
    """

    part_id: int
    """
    The part's identifier, unique within its scene.
    """

    label: str
    """
    The part's name followed by its position in the part hierarchy, e.g.
    ``"door.1.1.2_1"`` for the first door of the cabinet labelled ``"cabinet.1.1"``.
    """

    face_indices: np.ndarray
    """
    The faces of the ScanNet++ mesh the part covers.
    """

    @property
    def part_name(self) -> str:
        """
        The part's name without its hierarchy position, e.g. ``"door"``.
        """
        return self.label.split(".", 1)[0]

    @property
    def hierarchy_path(self) -> str | None:
        """
        The part's position in the part hierarchy, e.g. ``"1.1.2_1"``, or ``None`` for a
        part annotated without one.
        """
        if "." not in self.label:
            return None
        return self.label.split(".", 1)[1]

    @property
    def parent_hierarchy_path(self) -> str | None:
        """
        The hierarchy position of the part this part belongs to, e.g. ``"1.1"`` for
        ``"door.1.1.2_1"``, or ``None`` for a part at the top of the hierarchy.
        """
        if self.hierarchy_path is None or self.hierarchy_path.count(".") < 2:
            return None
        return self.hierarchy_path.rsplit(".", 1)[0]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Articulate3DPart:
        """
        :param data: One entry of ``data.annotations`` in ``parts.json``.
        :return: The part the entry describes.
        """
        return cls(
            part_id=data["partId"],
            label=data["label"],
            face_indices=np.asarray(data["triIndices"], dtype=int),
        )

    @classmethod
    def all_in_file(cls, path: Path) -> list[Articulate3DPart]:
        """
        :param path: A scene's ``parts.json``.
        :return: Every part the file annotates, in the file's order.
        """
        parts = json.loads(Path(path).read_text())
        return [
            cls.from_json(annotation) for annotation in parts["data"]["annotations"]
        ]


# %% articulations


class Articulate3DMotionType(StrEnum):
    """
    How an articulated part moves relative to its base.
    """

    ROTATION = "rotation"
    """
    The part turns about an axis, like a door on its hinges.
    """

    TRANSLATION = "translation"
    """
    The part slides along an axis, like a drawer.
    """

    @property
    def connection_type(self) -> Type[ActiveConnection1DOF]:
        """
        The connection that moves a part this way.
        """
        if self is Articulate3DMotionType.ROTATION:
            return RevoluteConnection
        return PrismaticConnection


@dataclass
class Articulate3DArticulation:
    """
    How one part moves relative to its base part.

    Positions are measured from the pose the part was scanned in, in radians for a
    rotation and in meters for a translation.
    """

    part_id: int
    """
    The part that moves.
    """

    motion_type: Articulate3DMotionType
    """
    How the part moves.
    """

    axis: np.ndarray
    """
    The unit axis the part turns about or slides along, in the scan's frame.
    """

    origin: np.ndarray
    """
    A point on the axis, in the scan's frame.
    """

    lower_limit: float
    """
    The lowest position the part reaches.
    """

    upper_limit: float
    """
    The highest position the part reaches.
    """

    base_part_ids: list[int]
    """
    The parts the part moves relative to, in the annotation's order.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Articulate3DArticulation:
        """
        :param data: One entry of ``data.articulations`` in ``artic.json``.
        :return: The articulation the entry describes.
        """
        return cls(
            part_id=data["pid"],
            motion_type=Articulate3DMotionType(data["type"]),
            axis=np.asarray(data["axis"], dtype=float),
            origin=np.asarray(data["origin"], dtype=float),
            lower_limit=data["rangeMin"],
            upper_limit=data["rangeMax"],
            base_part_ids=list(data["base"]),
        )


# %% scene


@dataclass
class Articulate3DScene:
    """
    A ScanNet++ scene together with its Articulate3D part and articulation annotations.
    """

    scannetpp_scene: ScanNetPPScene
    """
    The scanned scene the annotations refer to.
    """

    parts: list[Articulate3DPart]
    """
    The annotated parts.
    """

    articulations: list[Articulate3DArticulation]
    """
    How the movable parts move, as annotated; a part may have several.
    """

    def create_world(self) -> World:
        """
        Build a world with one body per part, cut off the root body, ``scan``, which
        keeps the faces no part covers; every body is shown in a color of its own.

        Each articulated part is connected to its base part by a
        :class:`~semantic_digital_twin.world_description.connections.RevoluteConnection`
        or :class:`~semantic_digital_twin.world_description.connections.PrismaticConnection`
        whose position is zero in the scanned pose, or to the root body, ``scan``, when
        its base is no part. Of several listed bases, the base is the one that is the
        part's parent in the part hierarchy, or the first listed when not exactly one
        is. Every other part is fixed to its parent in the part hierarchy, or to
        ``scan`` at the top of the hierarchy. Every part's body carries a
        :class:`~semantic_digital_twin.semantic_annotations.natural_language.NaturalLanguageDescription`
        of its :attr:`~Articulate3DPart.part_name`.

        .. note::
            The annotations sometimes give a part several articulations; only the last
            one moves it. Articulations of parts without annotation are left out.

        :return: The built world.
        """
        bodies = {
            part.part_id: Body(name=PrefixedName(part.label)) for part in self.parts
        }
        world = self.scannetpp_scene.create_world_cut_into(
            [
                ShapePiece(body=bodies[part.part_id], faces=part.face_indices)
                for part in self.parts
            ]
        )
        articulations = self._articulation_per_part()
        base_part_ids = self._base_part_ids(articulations)
        parts_by_hierarchy_path = self._parts_by_hierarchy_path()
        colors = Color.distinct_colors(len(self.parts) + 1)
        with world.modify_world():
            # Setting the color is enough: the pieces' meshes are not built yet, and
            # are colored when they are; dyeing them would load every piece now.
            for shape in world.root.visual:
                shape.color = colors[-1]
            for part, color in zip(self.parts, colors):
                for shape in bodies[part.part_id].visual:
                    shape.color = color
                world.add_semantic_annotation(
                    NaturalLanguageDescription(
                        root=bodies[part.part_id], description=part.part_name
                    )
                )
            for part in self.parts:
                child = bodies[part.part_id]
                if part.part_id in articulations:
                    world.remove_connection(child.parent_connection)
                    self._add_articulated_connection(
                        world,
                        articulations[part.part_id],
                        bodies.get(base_part_ids[part.part_id], world.root),
                        child,
                    )
                    continue
                parent_part = parts_by_hierarchy_path.get(part.parent_hierarchy_path)
                if parent_part is None:
                    continue
                world.remove_connection(child.parent_connection)
                world.add_connection(
                    FixedConnection(parent=bodies[parent_part.part_id], child=child)
                )
        return world

    def _articulation_per_part(self) -> dict[int, Articulate3DArticulation]:
        """
        :return: The last articulation of every annotated part, by part.
        """
        part_ids = {part.part_id for part in self.parts}
        articulations = {}
        for articulation in self.articulations:
            if articulation.part_id not in part_ids:
                logger.warning(
                    f"Leaving out the articulation of part {articulation.part_id} in "
                    f"scene {self.scannetpp_scene.scene_id}, which is not annotated."
                )
                continue
            if articulation.part_id in articulations:
                logger.warning(
                    f"Part {articulation.part_id} in scene "
                    f"{self.scannetpp_scene.scene_id} has several articulations; only "
                    f"the last one moves it."
                )
            articulations[articulation.part_id] = articulation
        return articulations

    def _parts_by_hierarchy_path(self) -> dict[str, Articulate3DPart]:
        """
        :return: Each hierarchy position's first part.
        """
        parts = {}
        for part in self.parts:
            if part.hierarchy_path is not None:
                parts.setdefault(part.hierarchy_path, part)
        return parts

    def _base_part_ids(
        self, articulations: dict[int, Articulate3DArticulation]
    ) -> dict[int, int]:
        """
        :param articulations: The articulation that moves each articulated part.
        :return: For every articulated part, the base it moves relative to: of the bases
            its articulation lists, the one that is the part's parent in the part
            hierarchy, or the first listed when not exactly one is.
        """
        parts = {part.part_id: part for part in self.parts}
        base_part_ids = {}
        for articulation in articulations.values():
            parent_hierarchy_path = parts[articulation.part_id].parent_hierarchy_path
            hierarchy_parents = [
                base_part_id
                for base_part_id in articulation.base_part_ids
                if parent_hierarchy_path is not None
                and base_part_id in parts
                and parts[base_part_id].hierarchy_path == parent_hierarchy_path
            ]
            if len(hierarchy_parents) == 1:
                base_part_ids[articulation.part_id] = hierarchy_parents[0]
                continue
            if len(articulation.base_part_ids) > 1:
                logger.warning(
                    f"Part {articulation.part_id} in scene "
                    f"{self.scannetpp_scene.scene_id} lists the bases "
                    f"{articulation.base_part_ids}, none of them its single parent in "
                    f"the part hierarchy; it moves relative to the first."
                )
            base_part_ids[articulation.part_id] = articulation.base_part_ids[0]
        return base_part_ids

    @staticmethod
    def _add_articulated_connection(
        world: World,
        articulation: Articulate3DArticulation,
        parent: Body,
        child: Body,
    ) -> None:
        """
        Connect a part to the body it moves relative to, so that it moves as the
        articulation describes.

        Every body's frame coincides with the scan's frame in the scanned pose, so the
        connection sits at the articulation's origin and the child is offset back from
        it.

        :param world: The world being modified.
        :param articulation: How the part moves.
        :param parent: The body the part moves relative to.
        :param child: The body of the moving part.
        """
        connection = articulation.motion_type.connection_type.create_with_dofs(
            world,
            parent,
            child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                *articulation.origin, reference_frame=parent
            ),
            connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                *-articulation.origin, child_frame=child
            ),
            axis=Vector3(*articulation.axis, reference_frame=parent),
            dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=articulation.lower_limit,
                upper_position=articulation.upper_limit,
            ),
        )
        world.add_connection(connection)
