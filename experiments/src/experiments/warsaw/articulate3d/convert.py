"""
Write an Articulate3D scene as a scene the Warsaw pipeline already reads.

The pipeline reads one mesh whose faces carry, per label, the instance each face belongs
to. An Articulate3D part becomes one instance of the label its name is written as, and
nothing else of the annotation reaches the mesh:

- a part's position in the part hierarchy is left out, since it is the answer the
  pipeline is judged on;
- instance numbers are drawn at random per label, so their order says nothing about the
  hierarchy either;
- labels are the annotators' words or, to judge what the pictures alone tell a model,
  neutral names that say nothing about what a part is.

Every face of the scan is kept, in the order the annotation numbers them, so a body a run
cuts out of the scene names the same faces the ground truth does. The parts are written
as the protocol's annotation rules correct them, and a scene the rules leave out is not
written.

Run ``python -m experiments.warsaw.articulate3d.convert --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np
from semantic_digital_twin.adapters.articulate3d_dataset.loader import (
    Articulate3DDatasetLoader,
)
from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DPart,
    Articulate3DScene,
)
from semantic_digital_twin.adapters.scannetpp_dataset.loader import (
    ScanNetPPDatasetLoader,
)

from experiments.warsaw.articulate3d.exceptions import (
    ConvertedLabelsCollideError,
    SceneExcludedError,
    SceneNotConvertedError,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol, ProtocolFile
from experiments.warsaw.articulate3d.rules import AnnotatedScene
from experiments.warsaw.articulate3d.splits import ValidationLock
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.habitat.convert import label_property
from experiments.warsaw.world_loader.scene import SceneFrame, SourceFrame

# %% what a converted scene is written as


class ConvertedSceneFile(StrEnum):
    """
    The files a converted scene's directory holds beside its frame.
    """

    MESH = "mesh_all_classes.ply"
    """
    The scan, its faces carrying the labels.
    """

    RECORD = "articulate3d_scene.json"
    """
    Where the scene came from and which segment each part was written as.
    """


class LabelDisclosure(StrEnum):
    """
    What the labels of a converted scene say about the parts they mark.
    """

    WORDS = "words"
    """
    The names annotators gave the parts.
    """

    NEUTRAL = "neutral"
    """
    Names that only tell the parts of one annotated name from those of another.
    """


class LabelOverlap(StrEnum):
    """
    Which faces a part's label covers.
    """

    DISJOINT = "disjoint"
    """
    The faces the part is annotated with.
    """

    SUBTREE = "subtree"
    """
    The faces of the part and of every part below it in the part hierarchy, so a cabinet
    is labelled over its doors as well, the way a scan's labels overlap.
    """


# %% what was written


@dataclass(frozen=True)
class ConvertedPart(JsonRecord):
    """
    One Articulate3D part, as the dataset has it and as the pipeline will.
    """

    part_id: int
    """
    The part, as Articulate3D numbers the parts of a scene.
    """

    label: str
    """
    Its full annotated label, position in the part hierarchy included.
    """

    segment: str
    """
    What every step of a run calls it.
    """

    faces: int
    """
    How many faces its label was written on.
    """


@dataclass(frozen=True)
class ConvertedScene(JsonRecord):
    """
    Where a converted scene came from, which nothing in its mesh says.
    """

    scene_id: str
    """
    The ScanNet++ scene, as both datasets name it.
    """

    disclosure: LabelDisclosure
    """
    What its labels say about the parts they mark.
    """

    overlap: LabelOverlap
    """
    Which faces each part's label covers.
    """

    seed: int
    """
    What the random instance numbers and neutral names were drawn from.
    """

    parts: list[ConvertedPart] = field(default_factory=list)
    """
    Every part written onto the mesh.
    """

    faceless: list[int] = field(default_factory=list)
    """
    The parts annotated with no faces, which nothing can find in a mesh.
    """

    @classmethod
    def beside(cls, directory: Path) -> ConvertedScene:
        """
        :param directory: A converted scene's directory.
        :return: What it says about where it came from.
        :raises SceneNotConvertedError: If the directory was not converted from
            Articulate3D.
        """
        path = Path(directory) / ConvertedSceneFile.RECORD
        if not path.is_file():
            raise SceneNotConvertedError(scene_directory=Path(directory))
        return cls.from_json(json.loads(path.read_text()))

    def write_beside(self, directory: Path) -> None:
        """
        :param directory: The converted scene's directory.
        """
        (Path(directory) / ConvertedSceneFile.RECORD).write_text(
            json.dumps(self.to_json(), indent=2)
        )


# %% converting a scene


@dataclass(frozen=True)
class SceneConversion:
    """
    How an Articulate3D scene is written for the pipeline.
    """

    disclosure: LabelDisclosure = LabelDisclosure.WORDS
    """
    What the labels say about the parts they mark.
    """

    overlap: LabelOverlap = LabelOverlap.DISJOINT
    """
    Which faces each part's label covers.
    """

    seed: int = 0
    """
    What the random instance numbers and neutral names are drawn from.
    """

    lock: ValidationLock = field(default_factory=ValidationLock)
    """
    What keeps the validation scenes closed until the code is frozen.
    """

    def write(self, scene: Articulate3DScene, output: Path) -> ConvertedScene:
        """
        Write a scene into a directory the pipeline can be pointed at.

        :param scene: The scene and its annotations.
        :param output: The directory to write it into.
        :return: What was written.
        :raises ValidationSceneLockedError: If it is a validation scene and the checkout
            is not frozen.
        """
        self.lock.check(scene.scannetpp_scene.scene_id)
        random = np.random.default_rng(self.seed)
        drawn = [part for part in scene.parts if len(part.face_indices)]
        properties = self._properties(drawn, random)
        instances = self._instances(drawn, properties, random)
        mesh = scene.scannetpp_scene.mesh.copy()
        # Welding gives two parts an edge to meet along, which is the evidence the
        # pipeline relates them by. It renumbers vertices, never faces.
        mesh.merge_vertices()
        labels: dict[str, np.ndarray] = {
            name: np.zeros(len(mesh.faces), dtype=np.uint32)
            for name in sorted(set(properties.values()))
        }
        written: dict[int, int] = {}
        for part in sorted(drawn, key=self._depth):
            faces = self._faces_labelled(part, drawn)
            labels[properties[part.part_id]][faces] = instances[part.part_id]
            written[part.part_id] = len(faces)
        for name, values in labels.items():
            mesh.face_attributes[name] = values

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        mesh.export(output / ConvertedSceneFile.MESH)
        SceneFrame(source=SourceFrame.UPRIGHT).write_beside(output)
        converted = ConvertedScene(
            scene_id=scene.scannetpp_scene.scene_id,
            disclosure=self.disclosure,
            overlap=self.overlap,
            seed=self.seed,
            parts=[
                ConvertedPart(
                    part_id=part.part_id,
                    label=part.label,
                    segment=f"{properties[part.part_id]}_{instances[part.part_id]}",
                    faces=written[part.part_id],
                )
                for part in drawn
            ],
            faceless=[
                part.part_id for part in scene.parts if not len(part.face_indices)
            ],
        )
        converted.write_beside(output)
        return converted

    def _properties(
        self, parts: list[Articulate3DPart], random: np.random.Generator
    ) -> dict[int, str]:
        """
        :param parts: The parts to write.
        :param random: What neutral names are drawn from.
        :return: The face property each part's label is written as, by part.
        :raises ConvertedLabelsCollideError: If two part names would be written as one
            property.
        """
        part_names = sorted({part.part_name for part in parts})
        if self.disclosure is LabelDisclosure.NEUTRAL:
            order = random.permutation(len(part_names))
            width = len(str(len(part_names)))
            named = {
                part_name: f"label_{position + 1:0{width}d}"
                for part_name, position in zip(part_names, order)
            }
        else:
            named = {part_name: label_property(part_name) for part_name in part_names}
        for property_name in set(named.values()):
            sharing = [one for one in part_names if named[one] == property_name]
            if len(sharing) > 1:
                raise ConvertedLabelsCollideError(
                    part_names=sharing, property_name=property_name
                )
        return {part.part_id: named[part.part_name] for part in parts}

    @staticmethod
    def _instances(
        parts: list[Articulate3DPart],
        properties: dict[int, str],
        random: np.random.Generator,
    ) -> dict[int, int]:
        """
        :param parts: The parts to write.
        :param properties: The face property each part is written as.
        :param random: What the instance numbers are drawn from.
        :return: A distinct instance number per part of each property, from one up, by
            part.
        """
        instances = {}
        for property_name in sorted(set(properties.values())):
            sharing = [
                part for part in parts if properties[part.part_id] == property_name
            ]
            for part, number in zip(sharing, random.permutation(len(sharing)) + 1):
                instances[part.part_id] = int(number)
        return instances

    @staticmethod
    def _depth(part: Articulate3DPart) -> int:
        """
        :param part: A part.
        :return: How far down the part hierarchy it sits, so that where two parts of one
            name claim a face, the one further down is written last and keeps it.
        """
        if part.hierarchy_path is None:
            return 0
        return part.hierarchy_path.count(".")

    def _faces_labelled(
        self, part: Articulate3DPart, parts: list[Articulate3DPart]
    ) -> np.ndarray:
        """
        :param part: The part to label.
        :param parts: Every part written.
        :return: The faces its label covers.
        """
        if self.overlap is LabelOverlap.DISJOINT or part.hierarchy_path is None:
            return part.face_indices
        below = f"{part.hierarchy_path}."
        return np.unique(
            np.concatenate(
                [part.face_indices]
                + [
                    other.face_indices
                    for other in parts
                    if other.hierarchy_path is not None
                    and other.hierarchy_path.startswith(below)
                ]
            )
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for converting a scene.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="The directory holding Articulate3D's annotation files",
    )
    parser.add_argument(
        "--scannetpp",
        type=Path,
        required=True,
        help="ScanNet++'s data directory, holding the meshes the annotations were made on",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ProtocolFile.ARTICULATE3D,
        help="The frozen protocol, whose annotation rules the scene is written by; the "
        "one kept with this package by default",
    )
    parser.add_argument("--scene", required=True, help="The scene to convert")
    parser.add_argument("--output", type=Path, required=True, help="Where to write it")
    parser.add_argument(
        "--disclosure",
        type=LabelDisclosure,
        default=LabelDisclosure.WORDS,
        choices=list(LabelDisclosure),
        help="What the labels say about the parts",
    )
    parser.add_argument(
        "--overlap",
        type=LabelOverlap,
        default=LabelOverlap.DISJOINT,
        choices=list(LabelOverlap),
        help="Which faces each part's label covers",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="What the random numbering is drawn from"
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Write an Articulate3D scene as a scene the pipeline reads.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the scene is written.
    :raises SceneExcludedError: If the protocol's annotation rules leave the scene out.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    ruled = EvaluationProtocol.read(parsed.protocol).annotation_rules.apply(
        AnnotatedScene.read(parsed.scene, parsed.dataset, parsed.scannetpp)
    )
    if ruled.excluded:
        raise SceneExcludedError(
            scene_id=parsed.scene,
            findings=[finding.finding for finding in ruled.exclusions],
        )
    scene = replace(
        Articulate3DDatasetLoader(
            directory=parsed.dataset,
            scannetpp_dataset_loader=ScanNetPPDatasetLoader(directory=parsed.scannetpp),
        ).load(parsed.scene),
        parts=ruled.scene.parts,
    )
    converted = SceneConversion(
        disclosure=parsed.disclosure, overlap=parsed.overlap, seed=parsed.seed
    ).write(scene, parsed.output)
    logging.getLogger(__name__).info(
        "wrote %s parts of %s to %s, %s without faces left out",
        len(converted.parts),
        converted.scene_id,
        parsed.output,
        len(converted.faceless),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
