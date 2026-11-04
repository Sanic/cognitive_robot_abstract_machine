"""
A scanned scene, and the objects its labels mark out in it.

A scene arrives as one mesh whose faces carry one integer property per class, naming
which object of that class each face belongs to. The same face can carry several, since
a drawer front is labelled both as the drawer and as the cabinet holding it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from typing_extensions import Dict, Iterable, Iterator, List, Optional

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.world_loader.appearance import AppearanceMesh
from experiments.warsaw.exceptions import (
    AmbiguousWarsawSceneError,
    WarsawLabelsMissingError,
    WarsawSceneNotFoundError,
)

# %% where a scan keeps its labels


class PlyPayload(StrEnum):
    """
    Where a PLY file keeps the labels a scan wrote onto its faces.

    The path into the payload is the file format's, not ours, so it is named here rather
    than spelled at the one place that walks it.
    """

    RAW = "_ply_raw"
    """
    The header and data of the file, as the reader kept them.
    """

    FACE = "face"
    """
    The element the labels are written per.
    """

    DATA = "data"
    """
    The rows themselves, one per face.
    """

    GEOMETRY = "vertex_indices"
    """
    The face property holding a face's geometry rather than one of its labels.
    """


class ScanLabel(StrEnum):
    """
    The labels of a scan the loader itself reads, rather than passing on to be asked about.
    """

    FLOOR = "floor"
    """
    The floor, whose plane says which way up the scan stands.
    """


# %% the objects a scan labels


def segment_label(segments: Iterable[LabelSegment], maximum_length: int = 120) -> str:
    """
    Name the segments a render highlights, so its filename says what is colored in it.

    :param segments: The segments highlighted in the render.
    :param maximum_length: How many characters of names the filename can hold.
    :return: Their names joined by dashes, cut short of that length.
    """
    names = [str(segment.name).replace(" ", "_") for segment in segments]
    if len("-".join(names)) <= maximum_length:
        return "-".join(names)

    kept: List[str] = []
    length = 0
    for name in names:
        if length + len(name) + 1 > maximum_length:
            break
        kept.append(name)
        length += len(name) + 1
    return "-".join(kept + [f"and_{len(names) - len(kept)}_more"])


@dataclass
class LabelSegment:
    """
    One object a Warsaw scene labels: the faces one of its classes marks as one
    instance.

    A face can belong to segments of several classes at once, since a scene labels, for
    example, a drawer's front both as ``drawer`` and as the ``cabinet`` holding it.
    """

    class_name: str
    """
    The class that labels this object.
    """

    instance: int
    """
    Which object of that class this is.
    """

    face_indices: np.ndarray
    """
    Which of the scene mesh's faces this object is made of.
    """

    @property
    def name(self) -> PrefixedName:
        """
        :return: The name identifying this object among the scene's objects.
        """
        return PrefixedName(f"{self.class_name}_{self.instance}")

    def __len__(self) -> int:
        """
        :return: How many of the scene's faces this object is made of.
        """
        return len(self.face_indices)


# %% the frame a scan is written in


def source_rolled_upright() -> HomogeneousTransformationMatrix:
    """
    Turn the frame a scan file is written in roughly into the world's.

    A scan measures height roughly down its own y axis, so a floor is written at a
    greater y than the ceiling above it. This rolls that axis onto the world's upward z.
    How far the result still leans depends on the scan, since a reconstruction has no
    sense of gravity; :func:`stood_on_its_floor` takes out what is left.

    :return: The transform from a scan file's coordinates to the world's, before
        levelling.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(roll=-np.pi / 2)


def stood_on_its_floor(
    mesh: trimesh.Trimesh,
    floor_faces: np.ndarray,
    world_T_source: HomogeneousTransformationMatrix,
) -> HomogeneousTransformationMatrix:
    """
    Level a scene turned into the world's frame, so that its floor lies flat.

    The turn left after :func:`source_rolled_upright` is the smallest rotation taking the
    floor's plane onto the horizontal, so the scene keeps the heading the roll gave it.
    Upward is the side of the floor most of the scene stands on.

    :param mesh: The scene, in the coordinates of its file.
    :param floor_faces: Which of its faces are floor.
    :param world_T_source: The turn into the world's frame to level.
    :return: That turn followed by the levelling.
    """
    turned = world_T_source.to_np()
    vertices = trimesh.transform_points(mesh.vertices, turned)
    centre, normal = trimesh.points.plane_fit(
        vertices[np.unique(mesh.faces[floor_faces])]
    )
    if np.median((vertices - centre) @ normal) < 0:
        normal = -normal
    upward = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, upward)
    if np.linalg.norm(axis) < np.finfo(np.float64).eps:
        # A floor already flat, or flat upside down, turns about any horizontal axis.
        axis = np.array([1.0, 0.0, 0.0])
    levelling = trimesh.transformations.rotation_matrix(
        angle=float(np.arccos(np.clip(normal @ upward, -1.0, 1.0))), direction=axis
    )
    return HomogeneousTransformationMatrix(data=levelling @ turned)


class SourceFrame(StrEnum):
    """
    The way up a scene's file is written, of the ways a scene can be written.
    """

    SCANNED = "scanned"
    """
    A scan, measuring height roughly down its own y and levelled by its floor where it
    labels one.
    """

    UPRIGHT = "upright"
    """
    Already the world's way up, with z pointing up and nothing to turn.
    """

    @property
    def world_T_source(self) -> HomogeneousTransformationMatrix:
        """
        :return: The transform from a file written this way up into the world's frame,
            before a scan is levelled by its floor.
        """
        if self is SourceFrame.SCANNED:
            return source_rolled_upright()
        return HomogeneousTransformationMatrix()


SCENE_FRAME_FILE = "scene_frame.json"
"""
What a scene directory says which way up its mesh is written in.

A scan says nothing, because every scene predates the question and rolling one is what
the pipeline has always done. A scene written from somewhere else says so here rather
than being written upside down to suit that default: a frame nothing states is the one
mistake this has already cost a week.
"""


@dataclass
class SceneFrame(JsonRecord):
    """
    Which way up the mesh beside this record is written.
    """

    source: SourceFrame = SourceFrame.SCANNED
    """
    The frame the file is in.
    """

    @classmethod
    def beside(cls, directory: Path) -> SceneFrame:
        """
        :param directory: The scene directory to look in.
        :return: What it says, or a scan when it says nothing.
        """
        written = Path(directory) / SCENE_FRAME_FILE
        if not written.is_file():
            return cls()
        return cls.from_json(json.loads(written.read_text()))

    def write_beside(self, directory: Path) -> Path:
        """
        :param directory: The scene directory to write into.
        :return: The file written.
        """
        written = Path(directory) / SCENE_FRAME_FILE
        written.write_text(json.dumps(self.to_json(), indent=2))
        return written


# %% the scan itself


@dataclass
class WarsawScene:
    """
    A Warsaw scene: one mesh whose faces carry, per class, the instance they belong to.

    The scene is stored as a single mesh with one integer face property per class.
    Instance :attr:`unsegmented` marks the faces a class does not cover.
    """

    mesh_path: Path
    """
    The file the scene was read from.
    """

    mesh: trimesh.Trimesh
    """
    The scene's geometry, carrying the colors it was scanned in.
    """

    face_labels: Dict[str, np.ndarray]
    """
    Per class, the instance each face belongs to.
    """

    unsegmented: int = 0
    """
    The instance marking every face a class does not cover.
    """

    world_T_source: HomogeneousTransformationMatrix = field(
        default_factory=lambda: SourceFrame.SCANNED.world_T_source
    )
    """
    Turns the scene from the frame it is written in into the world's, standing a scan on
    its floor.
    """

    appearance: Optional[AppearanceMesh] = None
    """
    What the scene looks like, over the same faces, where its source says more about that
    than a welded and labelled mesh can carry. None for a scan, which says nothing.
    """

    @classmethod
    def from_directory(
        cls, directory: Path, scene_mesh_pattern: str = "*.ply"
    ) -> WarsawScene:
        """
        Read the scene a directory holds.

        :param directory: The directory holding the scene's mesh.
        :param scene_mesh_pattern: How that mesh is named.
        :raises WarsawSceneNotFoundError: If the directory holds no mesh.
        :raises AmbiguousWarsawSceneError: If it holds more than one.
        """
        directory = Path(directory)
        scene_meshes = sorted(directory.glob(scene_mesh_pattern))
        if not scene_meshes:
            raise WarsawSceneNotFoundError(
                directory=directory, scene_mesh_pattern=scene_mesh_pattern
            )
        if len(scene_meshes) > 1:
            raise AmbiguousWarsawSceneError(
                directory=directory, scene_meshes=scene_meshes
            )
        return cls.from_file(
            scene_meshes[0],
            SceneFrame.beside(directory).source,
            appearance=AppearanceMesh.beside(directory),
        )

    @classmethod
    def from_file(
        cls,
        scene_mesh_path: Path,
        frame: SourceFrame = SourceFrame.SCANNED,
        appearance: Optional[AppearanceMesh] = None,
    ) -> WarsawScene:
        """
        Read the scene one mesh file holds.

        :param scene_mesh_path: The mesh to read.
        :param frame: Which way up that file is written.
        :param appearance: What the scene looks like, where its source says more about
            that than this mesh can carry.
        :raises WarsawLabelsMissingError: If the mesh carries no per-face class labels.
        """
        scene_mesh_path = Path(scene_mesh_path)
        # Processing welds vertices and drops degenerate faces, which renumbers the
        # faces and would leave every label pointing at another face than the one it
        # was written for.
        mesh = trimesh.load(scene_mesh_path, process=False)
        face_labels = cls._read_face_labels(mesh, scene_mesh_path)
        # The payload is the file as it was written, which on a scan is most of what the
        # mesh weighs. The labels are the only thing read out of it, and they are read
        # here, so the mesh carries it no further and nothing copying the mesh copies it.
        mesh.metadata.pop(PlyPayload.RAW.value)
        return cls(
            mesh_path=scene_mesh_path,
            mesh=mesh,
            face_labels=face_labels,
            world_T_source=cls._world_T_source(mesh, face_labels, frame),
            appearance=appearance,
        )

    @classmethod
    def _world_T_source(
        cls,
        mesh: trimesh.Trimesh,
        face_labels: Dict[str, np.ndarray],
        frame: SourceFrame,
    ) -> HomogeneousTransformationMatrix:
        """
        :param mesh: The scene, in the coordinates of its file.
        :param face_labels: Per class, the instance each face belongs to.
        :param frame: Which way up the file is written.
        :return: The turn into the world's frame, a scan stood on its floor where it
            labels one and a scan labelling none only rolled.
        """
        if frame is not SourceFrame.SCANNED or ScanLabel.FLOOR not in face_labels:
            return frame.world_T_source
        return stood_on_its_floor(
            mesh,
            face_labels[ScanLabel.FLOOR] != cls.unsegmented,
            frame.world_T_source,
        )

    @staticmethod
    def _read_face_labels(
        mesh: trimesh.Trimesh,
        scene_mesh_path: Path,
        geometry_property: str = PlyPayload.GEOMETRY,
    ) -> Dict[str, np.ndarray]:
        """
        Read the instance each face belongs to, per class.

        :param mesh: The mesh the scene was read from.
        :param scene_mesh_path: The file it was read from, for the error message.
        :param geometry_property: The face property holding a face's geometry rather
            than one of its labels.
        :return: Per class, the instance each face belongs to.
        :raises WarsawLabelsMissingError: If the mesh carries no labels.
        """
        raw = mesh.metadata.get(PlyPayload.RAW.value) or {}
        face = raw.get(PlyPayload.FACE.value) or {}
        faces = face.get(PlyPayload.DATA.value)
        named = faces.dtype.names if faces is not None else None
        class_names = [name for name in (named or ()) if name != geometry_property]
        if not class_names:
            raise WarsawLabelsMissingError(scene_mesh=scene_mesh_path)
        return {name: np.asarray(faces[name]) for name in class_names}

    @property
    def class_names(self) -> List[str]:
        """
        :return: The classes the scene is labelled by, in the order it declares them.
        """
        return list(self.face_labels)

    def segments(self) -> Iterator[LabelSegment]:
        """
        :return: Every object the scene labels, in class order.
        """
        for class_name, instances in self.face_labels.items():
            for instance in np.unique(instances):
                if instance == self.unsegmented:
                    continue
                yield LabelSegment(
                    class_name=class_name,
                    instance=int(instance),
                    face_indices=np.flatnonzero(instances == instance),
                )
