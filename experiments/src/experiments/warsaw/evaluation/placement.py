"""
Both worlds as objects that can be paired by where they are.

Class and size are all two worlds in unrelated frames agree on, and in a kitchen they
are nearly evidence-free: every drawer of a run of drawers is the same class and the
same size, so which one was paired with which is close to arbitrary. Where an object
sits is what tells them apart, and it becomes available the moment landmarks relate the
two frames.

Bringing them together is one job with two halves. A run's bodies are cut back out of
the scene they were segmented from and moved by the fitted transform, which also carries
the scale, so what comes out is in the modelled world's frame and in metres however the
scan was scaled. The modelled world's objects are already there.
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray
from typing_extensions import Dict, Iterator, List, Optional, Set, Tuple

from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.evaluation.ground_truth import GroundTruthGraph
from experiments.warsaw.evaluation.size import ObjectSize
from experiments.warsaw.pipeline.run import Run, RunFile

from dataclasses import dataclass

# %% an object of either world, in the frame they share


@dataclass(frozen=True)
class PlacedObject(JsonRecord):
    """
    One object of either world, in the modelled world's frame and in metres.
    """

    name: str
    """
    What identifies it within its own world.
    """

    classes: List[str]
    """
    The classes it stands as.
    """

    centre: Optional[List[float]]
    """
    The middle of the box around it, or nothing where it has no geometry.
    """

    size: Optional[ObjectSize] = None
    """
    How big it is, or nothing where it has none to measure.
    """

    bounds: Optional[List[List[float]]] = None
    """
    What it spans, as low and high corners, or nothing where it spans nothing.
    """

    @classmethod
    def of(
        cls, name: str, classes: List[str], mesh: Optional[trimesh.Trimesh]
    ) -> PlacedObject:
        """
        Place and measure one object from its geometry.

        The middle of its box rather than the middle of its surface, because a scan
        covers the front of a cabinet far more densely than its sides and a centroid
        would follow the sampling rather than the object.

        :param name: What identifies it within its own world.
        :param classes: The classes it stands as.
        :param mesh: Its geometry, already in the modelled world's frame.
        :return: The object, placed where it has geometry to place it by.
        """
        if mesh is None or len(mesh.faces) == 0:
            return cls(name=name, classes=classes, centre=None)
        return cls(
            name=name,
            classes=classes,
            centre=[float(one) for one in mesh.bounds.mean(axis=0)],
            size=ObjectSize.of(mesh),
            bounds=[[float(one) for one in corner] for corner in mesh.bounds],
        )


# %% what a run built


def bodies_of_a_run(run: Run) -> Iterator[Tuple[str, trimesh.Trimesh]]:
    """
    Cut each body a run built back out of the scene it was cut from.

    A run records which faces of the scene each of its bodies is made of rather than the
    faces themselves, which on a scan are most of what the file weighs.

    :param run: The finished run to read.
    :return: Each body in turn, by the name the run gave it.
    """
    [scene_mesh] = sorted(run.scene_directory.glob("*.ply"))
    scene = trimesh.load(scene_mesh, process=False)
    faces_of = np.load(run.path(RunFile.SPLIT_FACES))
    for name in faces_of.files:
        yield name, scene.submesh([faces_of[name]], append=True, repair=False)


def placed_run_bodies(
    run: Run, scene_file_to_ground_truth: NDArray[np.float64]
) -> List[PlacedObject]:
    """
    Place every body a run built in the modelled world's frame.

    :param run: The finished run to place.
    :param scene_file_to_ground_truth: The fitted transform, scale included, as the
        landmarks picked on the scene file describe it.
    :return: Its bodies, placed and measured in metres.
    """
    classes_of = {
        node.name: node.classes
        for node in run.read_record(RunFile.EVALUATION_GRAPH, EvaluationGraph).nodes
    }
    placed = []
    for name, body in bodies_of_a_run(run):
        body.apply_transform(scene_file_to_ground_truth)
        placed.append(PlacedObject.of(name, classes_of.get(name, []), body))
    return placed


# %% what was modelled


def placed_modelled_objects(ground_truth: GroundTruthGraph) -> List[PlacedObject]:
    """
    Place every object of a modelled world, which is already in its own frame.

    :param ground_truth: The modelled world's graph.
    :return: Its objects, placed where the export recorded what they span.
    """
    reaching = _what_each_whole_spans(ground_truth)
    return [
        PlacedObject(
            name=node.name,
            classes=node.classes,
            centre=_middle_of(node.bounds or reaching.get(node.name)),
            size=node.size,
            bounds=node.bounds or reaching.get(node.name),
        )
        for node in ground_truth.nodes
    ]


def _what_each_whole_spans(
    ground_truth: GroundTruthGraph,
) -> Dict[str, List[List[float]]]:
    """
    Work out an extent for the wholes that carry no geometry of their own.

    A modelled world groups its furniture under nodes with no faces and no bounds -- the
    apartment holds the island's cabinets under ``side_B``. Nothing can be paired with
    an object that has no position, so every relation naming one as the whole would be
    thrown away, and the extent of a grouping is the extent of what it groups.

    :param ground_truth: The modelled world's graph.
    :return: Per whole that has no bounds, what everything inside it spans together.
    """
    own = {node.name: node.bounds for node in ground_truth.nodes}
    inside: Dict[str, List[str]] = {}
    for edge in ground_truth.edges:
        inside.setdefault(edge.whole, []).append(edge.part)

    spans: Dict[str, List[List[float]]] = {}

    def reaching(name: str, seen: Set[str]) -> Optional[List[List[float]]]:
        """
        :param name: The object to measure.
        :param seen: What is already being measured, so a world holding a part in a
            circle is walked once rather than forever.
        :return: What it and everything inside it span, or nothing where none of it has
            geometry.
        """
        if own.get(name):
            return own[name]
        if name in seen:
            return None
        corners = [
            found
            for part in inside.get(name, [])
            if (found := reaching(part, seen | {name})) is not None
        ]
        if not corners:
            return None
        low = np.min([one[0] for one in corners], axis=0)
        high = np.max([one[1] for one in corners], axis=0)
        return [[float(one) for one in low], [float(one) for one in high]]

    for whole in inside:
        if own.get(whole):
            continue
        found = reaching(whole, set())
        if found is not None:
            spans[whole] = found
    return spans


def _middle_of(bounds: Optional[List[List[float]]]) -> Optional[List[float]]:
    """
    :param bounds: What an object spans, as low and high corners, or nothing.
    :return: The middle of that box, or nothing where the object spans nothing.
    """
    if bounds is None:
        return None
    return [float(one) for one in np.asarray(bounds, dtype=np.float64).mean(axis=0)]


# %% both worlds at once


def placed_by_name(placed: List[PlacedObject]) -> Dict[str, PlacedObject]:
    """
    :param placed: Objects of one world.
    :return: The same, by the name that identifies each within it.
    """
    return {one.name: one for one in placed}
