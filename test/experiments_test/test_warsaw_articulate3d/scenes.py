"""
An Articulate3D scene small enough to check by hand, and the protocol it is judged by.

The scan is a row of squares, two triangles each, every triangle written with vertices
of its own, so two parts only meet along an edge once the vertices are welded. On it sit
a cabinet with a door and a drawer, each with a handle; a switch whose base and toggle
are annotated under names that cover different kinds of things; a hinge annotated with
no faces; and a part the protocol ignores. The last triangle is annotated by nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh
from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DAnnotationFile,
    Articulate3DArticulation,
    Articulate3DMotionType,
    Articulate3DPart,
    Articulate3DScene,
)
from semantic_digital_twin.adapters.scannetpp_dataset.schema import (
    ScanNetPPScene,
    ScanNetPPSceneFile,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)

from experiments.warsaw.articulate3d.rules import MeshSize
from experiments.warsaw.articulate3d.protocol import (
    EvaluationProtocol,
    PartClass,
    PartNameMeaning,
    PartNameRuling,
)

# %% the scan


class SceneName(StrEnum):
    """
    The scenes written here.
    """

    KITCHEN_CORNER = "kitchen_corner"
    """
    The cabinet, the switch and what the protocol ignores.
    """


def row_of_squares(squares: int) -> trimesh.Trimesh:
    """
    :param squares: How many unit squares to lay in a row along x.
    :return: Their triangles in order along the row, each with vertices of its own.
    """
    corners = []
    for left in range(squares):
        right = left + 1
        corners.append([[left, 0, 0], [right, 0, 0], [left, 1, 0]])
        corners.append([[right, 0, 0], [right, 1, 0], [left, 1, 0]])
    vertices = np.asarray(corners, dtype=float).reshape(-1, 3)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.arange(len(vertices)).reshape(-1, 3),
        process=False,
    )


# %% the parts


def part(part_id: int, label: str, faces: list[int]) -> Articulate3DPart:
    """
    :param part_id: The part's number.
    :param label: Its name and position in the part hierarchy.
    :param faces: The triangles it covers.
    :return: The part.
    """
    return Articulate3DPart(
        part_id=part_id, label=label, face_indices=np.asarray(faces, dtype=int)
    )


def kitchen_corner_parts() -> list[Articulate3DPart]:
    """
    :return: The scene's parts, as Articulate3D annotates them.
    """
    return [
        part(1, "cabinet.1.1", [0, 1, 2, 3]),
        part(2, "door.1.1.2_1", [4, 5]),
        part(3, "handle.1.1.2_1.3_1", [6]),
        part(4, "drawer.1.1.2_2", [7, 8]),
        part(5, "handle.1.1.2_2.3_1", [9]),
        part(6, "switch base.2.1", [10, 11]),
        part(7, "control.2.1.2_1", [12]),
        part(8, "hinge.1.1.2_1.3_2", []),
        part(9, "unknown.3.1", [13]),
    ]


def kitchen_corner() -> Articulate3DScene:
    """
    :return: The scene, with eight squares of scan under it.
    """
    mesh = row_of_squares(8)
    return Articulate3DScene(
        scannetpp_scene=ScanNetPPScene(
            scene_id=SceneName.KITCHEN_CORNER,
            mesh=mesh,
            mesh_path=Path(f"{SceneName.KITCHEN_CORNER}.ply"),
            vertex_segment_ids=np.zeros(len(mesh.vertices), dtype=int),
            objects=[],
        ),
        parts=kitchen_corner_parts(),
        articulations=[],
    )


@dataclass
class WrittenDataset:
    """
    An Articulate3D directory and a ScanNet++ data directory, written the way the two
    datasets write a scene.
    """

    articulate3d_directory: Path
    """
    Where the annotation files are written.
    """

    scannetpp_directory: Path
    """
    Where each scene's mesh is written.
    """

    def write_scene(
        self,
        scene_id: str,
        parts: list[Articulate3DPart],
        mesh: trimesh.Trimesh | None,
        articulations: list[Articulate3DArticulation] | None = None,
        recorded_mesh_size: MeshSize | None = None,
    ) -> None:
        """
        :param scene_id: The scene.
        :param parts: Its parts.
        :param mesh: Its mesh, written with no ScanNet++ objects on it, or nothing to
            write no mesh.
        :param articulations: How its parts move; none by default.
        :param recorded_mesh_size: The mesh size the annotation records; that of the
            mesh by default.
        """
        if recorded_mesh_size is None:
            recorded_mesh_size = MeshSize(
                vertices=len(mesh.vertices), faces=len(mesh.faces)
            )
        Articulate3DAnnotationFile.PARTS.path(
            self.articulate3d_directory, scene_id
        ).write_text(
            json.dumps(
                {
                    "data": {
                        "stats": {
                            "totalVertices": recorded_mesh_size.vertices,
                            "totalFaces": recorded_mesh_size.faces,
                        },
                        "annotations": [
                            {
                                "partId": one.part_id,
                                "label": one.label,
                                "triIndices": one.face_indices.tolist(),
                            }
                            for one in parts
                        ],
                    }
                }
            )
        )
        Articulate3DAnnotationFile.ARTICULATIONS.path(
            self.articulate3d_directory, scene_id
        ).write_text(
            json.dumps(
                {
                    "data": {
                        "articulations": [
                            {
                                "pid": one.part_id,
                                "type": one.motion_type.value,
                                "axis": one.axis.tolist(),
                                "origin": one.origin.tolist(),
                                "rangeMin": one.lower_limit,
                                "rangeMax": one.upper_limit,
                                "base": one.base_part_ids,
                            }
                            for one in articulations or []
                        ]
                    }
                }
            )
        )
        if mesh is None:
            return
        scene_directory = self.scannetpp_directory / scene_id
        (scene_directory / ScanNetPPSceneFile.MESH).parent.mkdir(parents=True)
        mesh.export(scene_directory / ScanNetPPSceneFile.MESH)
        (scene_directory / ScanNetPPSceneFile.SEGMENTS).write_text(
            json.dumps({"segIndices": [0] * len(mesh.vertices)})
        )
        (scene_directory / ScanNetPPSceneFile.SEGMENT_ANNOTATIONS).write_text(
            json.dumps({"segGroups": []})
        )


def articulation(part_id: int, base_part_ids: list[int]) -> Articulate3DArticulation:
    """
    :param part_id: The part that moves.
    :param base_part_ids: The parts it moves relative to.
    :return: A rotation of the part about the z axis.
    """
    return Articulate3DArticulation(
        part_id=part_id,
        motion_type=Articulate3DMotionType.ROTATION,
        axis=np.asarray([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        lower_limit=0.0,
        upper_limit=1.0,
        base_part_ids=base_part_ids,
    )


# %% the protocol


def kitchen_corner_protocol() -> EvaluationProtocol:
    """
    :return: A protocol ruling on every name the scene uses, with the switch base
        classified on sight.
    """
    return EvaluationProtocol(
        part_names=[
            PartNameMeaning(
                part_name="cabinet",
                ruling=PartNameRuling.CLASS,
                semantic_class="Cabinet",
            ),
            PartNameMeaning(
                part_name="door", ruling=PartNameRuling.CLASS, semantic_class="Door"
            ),
            PartNameMeaning(
                part_name="drawer", ruling=PartNameRuling.CLASS, semantic_class="Drawer"
            ),
            PartNameMeaning(
                part_name="handle", ruling=PartNameRuling.CLASS, semantic_class="Handle"
            ),
            PartNameMeaning(part_name="switch base", ruling=PartNameRuling.AMBIGUOUS),
            PartNameMeaning(part_name="control", ruling=PartNameRuling.AMBIGUOUS),
            PartNameMeaning(part_name="hinge", ruling=PartNameRuling.AMBIGUOUS),
            PartNameMeaning(part_name="unknown", ruling=PartNameRuling.IGNORED),
        ],
        part_classes=[
            PartClass(
                scene_id=SceneName.KITCHEN_CORNER,
                part_id=6,
                semantic_class="ElectricalDevice",
            )
        ],
    )


# %% a whole that could hold one part through two fields


@dataclass
class PartTwoFieldsAccept:
    """
    A part a whole could hold through either of two fields.
    """


@dataclass
class WholeWithTwoFieldsForOnePart:
    """
    A whole whose two fields both accept the same kind of part.
    """

    front: list[PartTwoFieldsAccept] = field(
        default_factory=list, metadata=IsPartWholeRelationship().as_dict()
    )
    """
    One field accepting the part.
    """

    back: list[PartTwoFieldsAccept] = field(
        default_factory=list, metadata=IsPartWholeRelationship().as_dict()
    )
    """
    Another field accepting the same part.
    """
