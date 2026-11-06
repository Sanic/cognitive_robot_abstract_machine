"""
The ground truth of one Articulate3D scene, as the protocol makes it.

Every object comes from Articulate3D's own annotation: its parts, the faces each covers
and the part hierarchy. What a part *is* comes from the protocol, which a person wrote
before any run was judged, and which field holds a part comes from the ontology the run
was given. Nothing comes from inference rules, and nothing from what a run produced.

Run ``python -m experiments.warsaw.articulate3d.ground_truth --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DAnnotationFile,
    Articulate3DPart,
)
from semantic_digital_twin.semantic_annotations.part_whole import fields_accepting
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.articulate3d.exceptions import (
    AmbiguousPartFieldError,
    AmbiguousWholeError,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol, ProtocolFile
from experiments.warsaw.articulate3d.rules import RuleFindings
from experiments.warsaw.bases import JsonRecord

# %% what the ontology makes of a relation


class Expressibility(StrEnum):
    """
    Whether the ontology can hold one relation of the part hierarchy.
    """

    EXPRESSIBLE = "expressible"
    """
    The whole's class has a field that holds the part's class.
    """

    CARRIED = "carried"
    """
    No field of the whole's class holds the part's class, but the part's class holds the
    whole's class through a field whose parts carry their whole: mounted that way round,
    the whole hangs from the part as annotated.
    """

    INEXPRESSIBLE = "inexpressible"
    """
    No field of either class holds the other, so no run can mount the relation.
    """

    UNDETERMINED = "undetermined"
    """
    The class of an end is not judged, so whether a field could hold it is not known.
    """

    @property
    def reachable(self) -> bool:
        """
        :return: Whether a run can hang the part from the whole as annotated, through a
            field of either class.
        """
        return self in (Expressibility.EXPRESSIBLE, Expressibility.CARRIED)


# %% the ground truth of one scene


@dataclass(frozen=True)
class GroundTruthPart(JsonRecord):
    """
    One annotated part the evaluation covers.
    """

    part_id: int
    """
    The part, as Articulate3D numbers the parts of a scene.
    """

    part_name: str
    """
    The name its annotator gave it.
    """

    semantic_class: str | None
    """
    The class it is judged against, or nothing where its class is not judged.
    """

    faces: list[int]
    """
    The faces of the scan it covers.
    """


@dataclass(frozen=True)
class GroundTruthRelation(JsonRecord):
    """
    One relation of the part hierarchy: a part and the whole it belongs to.
    """

    whole: int
    """
    The part holding the other.
    """

    part: int
    """
    The part held.
    """

    expressibility: Expressibility
    """
    Whether the ontology can hold the relation.
    """

    field_name: str | None = None
    """
    The field the whole holds the part through, where the ontology can hold it.
    """

    @classmethod
    def between(
        cls,
        whole: int,
        part: int,
        whole_class: str | None,
        part_class: str | None,
        ontology: dict[str, type],
        chosen_field: str | None = None,
    ) -> GroundTruthRelation:
        """
        :param whole: The part holding the other.
        :param part: The part held.
        :param whole_class: The class the whole is judged against, if any.
        :param part_class: The class the part is judged against, if any.
        :param ontology: The ontology's classes by name.
        :param chosen_field: The field a person chose, where the ontology has several.
        :return: The relation, with the field the ontology holds it through; for a
            relation it holds the other way round, the field of the part's class.
        :raises AmbiguousPartFieldError: If several fields could hold it and none was
            chosen.
        """
        if whole_class is None or part_class is None:
            return cls(
                whole=whole, part=part, expressibility=Expressibility.UNDETERMINED
            )
        if chosen_field is not None:
            return cls(
                whole=whole,
                part=part,
                expressibility=Expressibility.EXPRESSIBLE,
                field_name=chosen_field,
            )
        expressibility = Expressibility.EXPRESSIBLE
        fields = fields_accepting(ontology[whole_class], ontology[part_class])
        if not fields:
            expressibility = Expressibility.CARRIED
            fields = [
                one
                for one in fields_accepting(ontology[part_class], ontology[whole_class])
                if one.part_carries_whole
            ]
        if not fields:
            return cls(
                whole=whole, part=part, expressibility=Expressibility.INEXPRESSIBLE
            )
        if len(fields) > 1:
            raise AmbiguousPartFieldError(
                whole_class=whole_class,
                part_class=part_class,
                fields=[one.field_name for one in fields],
            )
        return cls(
            whole=whole,
            part=part,
            expressibility=expressibility,
            field_name=fields[0].field_name,
        )


@dataclass(frozen=True)
class SceneGroundTruth(JsonRecord):
    """
    Everything one scene is judged against.
    """

    scene_id: str
    """
    The ScanNet++ scene, as both datasets name it.
    """

    parts: list[GroundTruthPart] = field(default_factory=list)
    """
    Every part the evaluation covers.
    """

    relations: list[GroundTruthRelation] = field(default_factory=list)
    """
    Every relation of the part hierarchy between two of those parts.
    """

    @classmethod
    def read(cls, path: Path) -> SceneGroundTruth:
        """
        :param path: A ground truth file.
        :return: What it holds.
        """
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path: Path) -> None:
        """
        :param path: Where to write the ground truth.
        """
        Path(path).write_text(json.dumps(self.to_json()))


# %% building it


@dataclass(frozen=True)
class GroundTruthBuilder:
    """
    Turns Articulate3D's annotation of a scene into ground truth, by a frozen protocol.
    """

    protocol: EvaluationProtocol
    """
    What a person decided about the dataset's annotations.
    """

    ontology: dict[str, type]
    """
    The ontology's classes by name, which the protocol's classes must belong to.
    """

    def __post_init__(self) -> None:
        self.protocol.check_classes(set(self.ontology))

    def build(
        self, scene_id: str, annotated: list[Articulate3DPart]
    ) -> SceneGroundTruth:
        """
        Make the ground truth of one scene.

        A part without faces is left out, since no run can find it in a mesh, and so is
        a part whose name the protocol ignores. A relation is kept only where both of
        its ends are.

        :param scene_id: The scene.
        :param annotated: Its annotated parts.
        :return: The scene's ground truth.
        :raises AmbiguousWholeError: If a part lies below a place in the part hierarchy
            several covered parts share.
        """
        covered = [
            part
            for part in annotated
            if len(part.face_indices) and self.protocol.covers(part.part_name)
        ]
        parts = [
            GroundTruthPart(
                part_id=part.part_id,
                part_name=part.part_name,
                semantic_class=self.protocol.class_of(
                    scene_id, part.part_id, part.part_name
                ),
                faces=[int(face) for face in part.face_indices],
            )
            for part in covered
        ]
        class_of = {part.part_id: part.semantic_class for part in parts}
        at_position: dict[str, list[Articulate3DPart]] = {}
        for part in covered:
            if part.hierarchy_path is not None:
                at_position.setdefault(part.hierarchy_path, []).append(part)
        relations = []
        for part in covered:
            wholes = at_position.get(part.parent_hierarchy_path, [])
            if len(wholes) > 1:
                raise AmbiguousWholeError(
                    scene_id=scene_id,
                    part_label=part.label,
                    whole_labels=[whole.label for whole in wholes],
                )
            if wholes:
                whole_class = class_of[wholes[0].part_id]
                part_class = class_of[part.part_id]
                relations.append(
                    GroundTruthRelation.between(
                        whole=wholes[0].part_id,
                        part=part.part_id,
                        whole_class=whole_class,
                        part_class=part_class,
                        ontology=self.ontology,
                        chosen_field=self.protocol.field_choice(
                            whole_class, part_class
                        ),
                    )
                )
        return SceneGroundTruth(scene_id=scene_id, parts=parts, relations=relations)


# %% command-line entry point


class GroundTruthFile(StrEnum):
    """
    How a scene's ground truth file is named.
    """

    SUFFIX = "_ground_truth.json"
    """
    What follows the scene id.
    """

    RULE_FINDINGS = "rule_findings.json"
    """
    The file beside the scenes' ground truth that lists every annotation rule that
    applied to them.
    """

    def path(self, directory: Path, scene_id: str) -> Path:
        """
        :param directory: The directory holding ground truth files.
        :param scene_id: The scene.
        :return: Where the scene's ground truth is written.
        """
        return Path(directory) / f"{scene_id}{self.value}"


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for building ground truth.
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
        help="The frozen protocol; the one kept with this package by default",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the ground truth"
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        help="A file of scene ids, one per line; every annotated scene by default",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Build and write the ground truth of a set of scenes.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once every scene's ground truth is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    load_annotation_modules()
    protocol = EvaluationProtocol.read(parsed.protocol)
    builder = GroundTruthBuilder(
        protocol=protocol, ontology=annotation_classes(SemanticAnnotation)
    )
    scene_ids = (
        parsed.scenes.read_text().split()
        if parsed.scenes
        else Articulate3DAnnotationFile.PARTS.scene_ids(parsed.dataset)
    )
    parsed.output.mkdir(parents=True, exist_ok=True)
    ruled_scenes = protocol.annotation_rules.apply_to_scenes(
        scene_ids, parsed.dataset, parsed.scannetpp
    )
    for ruled in ruled_scenes:
        if ruled.excluded:
            logger.info("%s: left out", ruled.scene.scene_id)
            continue
        built = builder.build(ruled.scene.scene_id, ruled.scene.parts)
        built.write(GroundTruthFile.SUFFIX.path(parsed.output, ruled.scene.scene_id))
        logger.info(
            "%s: %s parts, %s relations, %s corrections",
            ruled.scene.scene_id,
            len(built.parts),
            len(built.relations),
            len(ruled.corrections),
        )
    RuleFindings.of(ruled_scenes).write(parsed.output / GroundTruthFile.RULE_FINDINGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
