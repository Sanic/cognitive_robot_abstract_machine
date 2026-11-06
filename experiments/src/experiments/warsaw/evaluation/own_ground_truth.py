"""
Write the ground truth of a scene of our own.

A scene's labelled mesh says which objects there are and which faces each covers. What it
does not say is what each object is and which whole it belongs to, so a person says that:

- ``draft`` writes a file listing every object as undecided, beside what each one meets;
- a person decides each of them in that file, before looking at any run of the scene;
- ``build`` turns the decided file into the ground truth a run is scored against, and
  refuses a file that leaves anything open.

Run ``python -m experiments.warsaw.evaluation.own_ground_truth --help`` for its options.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Sequence

from experiments.warsaw.articulate3d.convert import ConvertedScene
from experiments.warsaw.articulate3d.exceptions import ClassNotInOntologyError
from experiments.warsaw.articulate3d.ground_truth import (
    GroundTruthFile,
    GroundTruthPart,
    GroundTruthRelation,
    SceneGroundTruth,
)
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.exceptions import (
    DraftAlreadyWrittenError,
    PartHierarchyCycleError,
    UndecidedGroundTruthError,
    UnknownWholeError,
)
from experiments.warsaw.segment_relations import segment_evidence
from experiments.warsaw.world_loader.loader import WarsawWorldLoader

# %% what is written in place of a class or a whole


class Decision(StrEnum):
    """
    What a person writes where the answer is not the name of a class or of an object.
    """

    UNDECIDED = "undecided"
    """
    Nobody has said yet, which a ground truth cannot be built from.
    """

    NOT_JUDGED = "not judged"
    """
    The object's class is not judged, as where a label covers different kinds of things.
    """

    AS_ITS_LABEL = "as its label"
    """
    The object is what its label was decided to mean.
    """

    NO_WHOLE = "no whole"
    """
    The object belongs to no other object of the scene.
    """


# %% the file a person fills in


@dataclass
class LabelDecision(JsonRecord):
    """
    What one label of the scene means.
    """

    label: str
    """
    The label, as the mesh writes it.
    """

    semantic_class: str = Decision.UNDECIDED.value
    """
    The class of the ontology it means, or a :class:`Decision`.
    """


@dataclass
class DraftObject(JsonRecord):
    """
    One object of the scene, what was measured of it and what a person decided about it.
    """

    name: str
    """
    The object, by the name a run gives its body.
    """

    label: str
    """
    The label it carries.
    """

    faces: int
    """
    How many faces it covers.
    """

    height: float
    """
    How high its middle sits above the lowest point of the scene.
    """

    meets: list[str] = field(default_factory=list)
    """
    The objects it shares faces with or touches, the most touched first.
    """

    semantic_class: str = Decision.AS_ITS_LABEL.value
    """
    The class of the ontology it is, or a :class:`Decision`.
    """

    whole: str = Decision.UNDECIDED.value
    """
    The object it belongs to, or a :class:`Decision`.
    """

    field_name: str | None = None
    """
    The field its whole holds it through, needed only where the ontology has several.
    """


@dataclass
class GroundTruthDraft(JsonRecord):
    """
    Every object of a scene, and what a person decided about it.

    An object removed from the file is left out of the ground truth, and whatever a run
    makes of its faces is not judged.
    """

    scene_id: str
    """
    The scene, as a run of it is scored under.
    """

    labels: list[LabelDecision] = field(default_factory=list)
    """
    What each label of the scene means.
    """

    objects: list[DraftObject] = field(default_factory=list)
    """
    Every object the ground truth covers.
    """

    @classmethod
    def of_scene(cls, directory: Path) -> GroundTruthDraft:
        """
        :param directory: The directory holding a scene's labelled mesh.
        :return: Every object of the scene, with nothing decided about it.
        """
        loader = WarsawWorldLoader(input_directory=directory)
        measured = segment_evidence(loader)
        meetings: dict[str, list[tuple[int, str]]] = {}
        for pair in measured.pairs:
            contact = pair.touching_edges + pair.shared_faces
            if contact:
                meetings.setdefault(pair.one, []).append((contact, pair.other))
                meetings.setdefault(pair.other, []).append((contact, pair.one))
        descriptors = measured.descriptors
        return cls(
            scene_id=ConvertedScene.scene_id_of(directory),
            labels=[
                LabelDecision(label)
                for label in sorted({one.class_name for one in descriptors.values()})
            ],
            objects=[
                DraftObject(
                    name=name,
                    label=descriptors[name].class_name,
                    faces=descriptors[name].faces,
                    height=round(descriptors[name].height, 2),
                    meets=[
                        other
                        for _, other in sorted(
                            meetings.get(name, []),
                            key=lambda one: (-one[0], one[1]),
                        )
                    ],
                )
                for name in sorted(descriptors)
            ],
        )

    @classmethod
    def read(cls, path: Path) -> GroundTruthDraft:
        """
        :param path: A draft file.
        :return: What it holds.
        """
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path: Path) -> None:
        """
        :param path: Where to write the draft, in a form a person can edit.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    # %% turning it into the ground truth

    def ground_truth(
        self, faces_of: dict[str, list[int]], ontology: dict[str, type]
    ) -> SceneGroundTruth:
        """
        :param faces_of: The faces of each object of the scene, by name.
        :param ontology: The ontology's classes by name.
        :return: The scene's ground truth.
        :raises UndecidedGroundTruthError: If anything is still undecided.
        :raises ClassNotInOntologyError: If a class is not one of the ontology.
        :raises UnknownWholeError: If a whole is not an object of the draft.
        :raises PartHierarchyCycleError: If objects belong to one another in a circle.
        :raises AmbiguousPartFieldError: If several fields could hold a part and the draft
            names none.
        """
        self.check_decided()
        self.check_wholes()
        classes = {one.name: self.class_of(one) for one in self.objects}
        for class_name in classes.values():
            if class_name is not None and class_name not in ontology:
                raise ClassNotInOntologyError(class_name=class_name)
        ids = {one.name: index for index, one in enumerate(self.objects, start=1)}
        return SceneGroundTruth(
            scene_id=self.scene_id,
            parts=[
                GroundTruthPart(
                    part_id=ids[one.name],
                    part_name=one.name,
                    semantic_class=classes[one.name],
                    faces=[int(face) for face in faces_of[one.name]],
                )
                for one in self.objects
            ],
            relations=[
                GroundTruthRelation.between(
                    whole=ids[one.whole],
                    part=ids[one.name],
                    whole_class=classes[one.whole],
                    part_class=classes[one.name],
                    ontology=ontology,
                    chosen_field=one.field_name,
                )
                for one in self.objects
                if one.whole != Decision.NO_WHOLE
            ],
        )

    def check_decided(self) -> None:
        """
        :raises UndecidedGroundTruthError: If a person has still something to decide.
        """
        undecided = [
            f"the class of label {one.label}"
            for one in self.labels
            if one.semantic_class == Decision.UNDECIDED
        ] + [
            f"the whole of {one.name}"
            for one in self.objects
            if one.whole == Decision.UNDECIDED
        ]
        if undecided:
            raise UndecidedGroundTruthError(undecided=undecided)

    def check_wholes(self) -> None:
        """
        :raises UnknownWholeError: If a whole is not an object of the draft.
        :raises PartHierarchyCycleError: If objects belong to one another in a circle.
        """
        wholes = {
            one.name: one.whole
            for one in self.objects
            if one.whole != Decision.NO_WHOLE
        }
        listed = {one.name for one in self.objects}
        for part, whole in wholes.items():
            if whole not in listed:
                raise UnknownWholeError(part=part, whole=whole)
        for start in wholes:
            walked = [start]
            while walked[-1] in wholes:
                following = wholes[walked[-1]]
                if following in walked:
                    raise PartHierarchyCycleError(objects=walked)
                walked.append(following)

    def class_of(self, one: DraftObject) -> str | None:
        """
        :param one: An object of the draft.
        :return: The class it is judged against, or nothing where its class is not judged.
        """
        by_label = {label.label: label.semantic_class for label in self.labels}
        decided = (
            by_label[one.label]
            if one.semantic_class == Decision.AS_ITS_LABEL
            else one.semantic_class
        )
        return None if decided == Decision.NOT_JUDGED else decided


# %% command-line entry point


class Command(StrEnum):
    """
    What the command can be asked to do.
    """

    DRAFT = "draft"
    """
    Write the file a person fills in.
    """

    BUILD = "build"
    """
    Turn the filled-in file into the ground truth.
    """


def argument_parser() -> argparse.ArgumentParser:
    """
    :return: The command-line interface for writing a scene's ground truth.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser(Command.DRAFT, help="Write the file to fill in")
    draft.add_argument("scene", type=Path, help="The scene's directory")
    draft.add_argument("--output", type=Path, required=True, help="The file to write")
    build = commands.add_parser(Command.BUILD, help="Build the ground truth from it")
    build.add_argument("scene", type=Path, help="The scene's directory")
    build.add_argument("draft", type=Path, help="The file a person filled in")
    build.add_argument(
        "--output",
        type=Path,
        required=True,
        help="The directory holding every scene's ground truth",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Write a scene's draft, or build its ground truth from one.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the file is written.
    :raises DraftAlreadyWrittenError: If a draft would be written over an existing file.
    """
    parsed = argument_parser().parse_args(arguments)
    if parsed.command == Command.DRAFT:
        if parsed.output.exists():
            raise DraftAlreadyWrittenError(path=parsed.output)
        GroundTruthDraft.of_scene(parsed.scene).write(parsed.output)
        print(parsed.output)
        return 0
    load_annotation_modules()
    draft = GroundTruthDraft.read(parsed.draft)
    loader = WarsawWorldLoader(input_directory=parsed.scene)
    built = draft.ground_truth(
        {str(one.name): one.face_indices.tolist() for one in loader.label_segments},
        annotation_classes(SemanticAnnotation),
    )
    parsed.output.mkdir(parents=True, exist_ok=True)
    path = GroundTruthFile.SUFFIX.path(parsed.output, draft.scene_id)
    built.write(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
