"""
The rules that turn Articulate3D's annotation into what the evaluation reads.

Some scenes cannot be evaluated at all: their mesh is missing or is not the one that was
annotated, a number the annotation records about the mesh is wrong, or the articulations
contradict the part hierarchy. A few parts are attached to the wrong whole. Each such
case is decided by a rule written here, never by hand per run, and every rule that
applies to a scene says so, which is what the paper reports.

Exclusions are general: they look at every scene and need no list of scenes. A
correction may name the one part it corrects.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

from plyfile import PlyData
from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DAnnotationFile,
    Articulate3DArticulation,
    Articulate3DPart,
)
from semantic_digital_twin.adapters.scannetpp_dataset.schema import (
    ScanNetPPSceneFile,
)
from typing_extensions import Any

from experiments.warsaw.bases import JsonRecord

# %% a scene as the rules see it


@dataclass(frozen=True)
class MeshSize(JsonRecord):
    """
    How many vertices and faces a triangle mesh has.
    """

    vertices: int
    """
    The number of vertices.
    """

    faces: int
    """
    The number of faces.
    """

    @classmethod
    def of_mesh_file(cls, path: Path) -> MeshSize:
        """
        :param path: A PLY mesh.
        :return: The size its header declares, read without reading the mesh itself.
        """
        with Path(path).open("rb") as stream:
            header = PlyData._parse_header(stream)
        return cls(vertices=header["vertex"].count, faces=header["face"].count)

    @classmethod
    def from_statistics_json(cls, data: dict[str, Any]) -> MeshSize:
        """
        :param data: ``data.stats`` of a ``parts.json``.
        :return: The size of the mesh the annotation records it was made on.
        """
        return cls(vertices=data["totalVertices"], faces=data["totalFaces"])


@dataclass(frozen=True)
class AnnotatedScene:
    """
    One scene's annotation, and the mesh it is laid on.
    """

    scene_id: str
    """
    The ScanNet++ scene.
    """

    parts: list[Articulate3DPart]
    """
    The annotated parts.
    """

    articulations: list[Articulate3DArticulation]
    """
    How the movable parts move.
    """

    recorded_mesh_size: MeshSize
    """
    The size of the mesh the annotation says it was made on.
    """

    mesh_size: MeshSize | None
    """
    The size of the ScanNet++ mesh at hand, or nothing where there is none.
    """

    @classmethod
    def read(
        cls, scene_id: str, articulate3d_directory: Path, scannetpp_directory: Path
    ) -> AnnotatedScene:
        """
        :param scene_id: The scene.
        :param articulate3d_directory: The directory holding Articulate3D's annotation
            files.
        :param scannetpp_directory: ScanNet++'s data directory, one folder per scene.
        :return: The scene's annotation and the size of its mesh.
        """
        parts_path = Articulate3DAnnotationFile.PARTS.path(
            articulate3d_directory, scene_id
        )
        articulations = json.loads(
            Articulate3DAnnotationFile.ARTICULATIONS.path(
                articulate3d_directory, scene_id
            ).read_text()
        )
        mesh_path = Path(scannetpp_directory) / scene_id / ScanNetPPSceneFile.MESH
        return cls(
            scene_id=scene_id,
            parts=Articulate3DPart.all_in_file(parts_path),
            articulations=[
                Articulate3DArticulation.from_json(articulation)
                for articulation in articulations["data"]["articulations"]
            ],
            recorded_mesh_size=MeshSize.from_statistics_json(
                json.loads(parts_path.read_text())["data"]["stats"]
            ),
            mesh_size=(
                MeshSize.of_mesh_file(mesh_path) if mesh_path.is_file() else None
            ),
        )


# %% rules


class AnnotationRule(JsonRecord, ABC):
    """
    A rule about how Articulate3D's annotation is read.
    """

    @property
    @abstractmethod
    def statement(self) -> str:
        """
        What the rule does, as one sentence for a reader of the evaluation.
        """

    @property
    @abstractmethod
    def reason(self) -> str:
        """
        Why the rule is needed, as one sentence for a reader of the evaluation.
        """


class SceneExclusion(AnnotationRule, ABC):
    """
    A reason to leave a whole scene out of the evaluation.
    """

    @abstractmethod
    def finding(self, scene: AnnotatedScene) -> str | None:
        """
        :param scene: A scene.
        :return: What makes the scene fall under this rule, or nothing where it does not.
        """


class AnnotationCorrection(AnnotationRule, ABC):
    """
    A change to what the annotation says about a scene.
    """

    @abstractmethod
    def correct(self, scene: AnnotatedScene) -> CorrectedScene | None:
        """
        :param scene: A scene.
        :return: The scene as corrected, or nothing where the correction does not
            concern it.
        """


@dataclass(frozen=True)
class CorrectedScene:
    """
    A scene a correction changed, and what it changed.
    """

    scene: AnnotatedScene
    """
    The scene after the correction.
    """

    finding: str
    """
    What the correction changed.
    """


# %% scenes left out


@dataclass(frozen=True)
class MeshMissing(SceneExclusion):
    """
    The scene has no ScanNet++ mesh, so its annotation labels nothing.
    """

    @property
    def statement(self) -> str:
        return "A scene is left out when ScanNet++ has no mesh for it."

    @property
    def reason(self) -> str:
        return (
            "Articulate3D labels the faces of a ScanNet++ mesh; without the mesh, the "
            "labels mark nothing."
        )

    def finding(self, scene: AnnotatedScene) -> str | None:
        if scene.mesh_size is not None:
            return None
        return "ScanNet++ holds no mesh for the scene."


@dataclass(frozen=True)
class MeshChanged(SceneExclusion):
    """
    The scene's mesh is not the one that was annotated, so the annotated faces lie
    elsewhere on it.

    The mesh was annotated if its number of vertices is the number the annotation
    records.
    """

    @property
    def statement(self) -> str:
        return (
            "A scene is left out when its ScanNet++ mesh has another number of vertices "
            "than the mesh the annotation records it was made on."
        )

    @property
    def reason(self) -> str:
        return (
            "The mesh changed between ScanNet++ versions, so the annotated face "
            "numbers mark other geometry than was annotated."
        )

    def finding(self, scene: AnnotatedScene) -> str | None:
        if (
            scene.mesh_size is None
            or scene.mesh_size.vertices == scene.recorded_mesh_size.vertices
        ):
            return None
        return (
            f"The annotation was made on a mesh of "
            f"{scene.recorded_mesh_size.vertices} vertices; the mesh has "
            f"{scene.mesh_size.vertices}."
        )


@dataclass(frozen=True)
class FaceCountMisrecorded(SceneExclusion):
    """
    The annotation was made on the scene's mesh but records a wrong number of faces for
    it.
    """

    @property
    def statement(self) -> str:
        return (
            "A scene is left out when its mesh has the number of vertices the annotation "
            "records, but another number of faces."
        )

    @property
    def reason(self) -> str:
        return (
            "The annotation contradicts itself: it labels faces beyond the number of "
            "faces it records. Its labels fit the mesh, but Articulate3D's own loader "
            "drops every face beyond the recorded number."
        )

    def finding(self, scene: AnnotatedScene) -> str | None:
        if (
            scene.mesh_size is None
            or scene.mesh_size.vertices != scene.recorded_mesh_size.vertices
            or scene.mesh_size.faces == scene.recorded_mesh_size.faces
        ):
            return None
        return (
            f"The annotation records {scene.recorded_mesh_size.faces} faces for a mesh "
            f"of {scene.mesh_size.faces}."
        )


@dataclass(frozen=True)
class ArticulationContradictsHierarchy(SceneExclusion):
    """
    An articulation of the scene does not fit its part hierarchy: the part that moves is
    not annotated, a base is not annotated, the bases leave out the part's parent in the
    hierarchy, or a part at the top of the hierarchy moves relative to a base.
    """

    @property
    def statement(self) -> str:
        return (
            "A scene is left out when one of its articulations does not fit its part "
            "hierarchy: the moving part or one of its bases is not annotated, its bases "
            "leave out its parent in the hierarchy, or it is at the top of the hierarchy "
            "and still moves relative to a base."
        )

    @property
    def reason(self) -> str:
        return (
            "Articulations and the part hierarchy both say what a part is attached to; "
            "where they disagree, the annotation does not say which is right."
        )

    def finding(self, scene: AnnotatedScene) -> str | None:
        contradicting = [
            articulation.part_id
            for articulation in scene.articulations
            if not self._fits(articulation, scene.parts)
        ]
        if not contradicting:
            return None
        return (
            f"{len(contradicting)} of {len(scene.articulations)} articulations do not "
            f"fit the part hierarchy, those of the parts {contradicting}."
        )

    @staticmethod
    def _fits(
        articulation: Articulate3DArticulation, parts: list[Articulate3DPart]
    ) -> bool:
        """
        :param articulation: An articulation of the scene.
        :param parts: The scene's parts.
        :return: Whether the articulation fits the part hierarchy.
        """
        by_id = {part.part_id: part for part in parts}
        if articulation.part_id not in by_id or any(
            base not in by_id for base in articulation.base_part_ids
        ):
            return False
        parent_path = by_id[articulation.part_id].parent_hierarchy_path
        if parent_path is None:
            return not articulation.base_part_ids
        return parent_path in [
            by_id[base].hierarchy_path for base in articulation.base_part_ids
        ]


@dataclass(frozen=True)
class AmbiguousWhole(SceneExclusion):
    """
    Several parts of the scene share one place in the part hierarchy and parts lie below
    that place, so which of them those parts belong to is not annotated.
    """

    @property
    def statement(self) -> str:
        return (
            "A scene is left out when several parts share one place in its part "
            "hierarchy and other parts lie below that place."
        )

    @property
    def reason(self) -> str:
        return (
            "A part belongs to the part at the place above it; where several parts "
            "share that place, the annotation does not say which one it belongs to."
        )

    def finding(self, scene: AnnotatedScene) -> str | None:
        at_place: dict[str, list[str]] = {}
        for part in scene.parts:
            if part.hierarchy_path is not None:
                at_place.setdefault(part.hierarchy_path, []).append(part.label)
        wholes = {part.parent_hierarchy_path for part in scene.parts}
        ambiguous = [
            labels
            for hierarchy_path, labels in at_place.items()
            if len(labels) > 1 and hierarchy_path in wholes
        ]
        if not ambiguous:
            return None
        return f"Parts lie below places several parts share: {ambiguous}."


# %% part names corrected


class PartNameRewrite(AnnotationCorrection, ABC):
    """
    A correction of how part names are written, made to every part of every scene.
    """

    @abstractmethod
    def rewritten(self, part_name: str) -> str:
        """
        :param part_name: A part name as annotators wrote it.
        :return: The name as this correction writes it, the same name where the
            correction does not concern it.
        """

    def correct(self, scene: AnnotatedScene) -> CorrectedScene | None:
        renamed = {
            part.part_id: self.rewritten(part.part_name)
            for part in scene.parts
            if self.rewritten(part.part_name) != part.part_name
        }
        if not renamed:
            return None
        return CorrectedScene(
            scene=replace(
                scene,
                parts=[
                    (
                        replace(
                            part,
                            label=renamed[part.part_id]
                            + part.label.removeprefix(part.part_name),
                        )
                        if part.part_id in renamed
                        else part
                    )
                    for part in scene.parts
                ],
            ),
            finding=self._finding(
                [
                    (part.part_name, renamed[part.part_id])
                    for part in scene.parts
                    if part.part_id in renamed
                ]
            ),
        )

    @staticmethod
    def _finding(renamings: list[tuple[str, str]]) -> str:
        """
        :param renamings: The name each renamed part had and the name it was given.
        :return: What was renamed, grouped by the name given, such as "'box_1' and
            'box_2' read as 'box' (2 parts)."
        """
        written_as: dict[str, list[str]] = {}
        for written, meant in renamings:
            written_as.setdefault(meant, []).append(written)
        sentences = []
        for meant, written in written_as.items():
            names = [f"'{name}'" for name in sorted(set(written))]
            listed = (
                names[0]
                if len(names) == 1
                else f"{', '.join(names[:-1])} and {names[-1]}"
            )
            count = len(written)
            sentences.append(
                f"{listed} read as '{meant}' ({count} {'part' if count == 1 else 'parts'})"
            )
        return "; ".join(sentences) + "."


@dataclass(frozen=True)
class SpacingCorrection(PartNameRewrite):
    """
    Spaces around a part name are removed, and several spaces in a row inside it become
    one, so ``"switch  base "`` becomes ``"switch base"``.
    """

    @property
    def statement(self) -> str:
        return (
            "Spaces around a part name are removed, and several spaces in a row inside "
            "it become one."
        )

    @property
    def reason(self) -> str:
        return "Names that differ only in spacing name the same thing."

    def rewritten(self, part_name: str) -> str:
        return " ".join(part_name.split())


@dataclass(frozen=True)
class InstanceNumberCorrection(PartNameRewrite):
    """
    A number counting the instances of one kind, written at the end of a part name, is
    removed, so ``"box_1"`` and ``"hinge1"`` become ``"box"`` and ``"hinge"``.
    """

    instance_number: str = r"_?\d+$"
    """
    The regular expression a counting number at the end of a name matches.
    """

    @property
    def statement(self) -> str:
        return (
            "A number counting instances, written at the end of a part name, is "
            "removed, so 'box_1' and 'hinge1' are read as 'box' and 'hinge'."
        )

    @property
    def reason(self) -> str:
        return (
            "The number tells instances of one kind apart, which the part hierarchy "
            "already does; it says nothing about what the part is."
        )

    def rewritten(self, part_name: str) -> str:
        return re.sub(self.instance_number, "", part_name)


@dataclass(frozen=True)
class PartNameCorrection(PartNameRewrite):
    """
    One part name was written other than meant, as by a spelling mistake, and is
    corrected wherever it occurs.
    """

    written: str
    """
    The name as written.
    """

    meant: str
    """
    The name meant.
    """

    mistake: str
    """
    How the name was written other than meant, completing "The name was written with
    ...", such as "a spelling mistake".
    """

    @property
    def statement(self) -> str:
        return f"The part name '{self.written}' is read as '{self.meant}'."

    @property
    def reason(self) -> str:
        return f"The name was written with {self.mistake}."

    def rewritten(self, part_name: str) -> str:
        if part_name != self.written:
            return part_name
        return self.meant


# %% applying the rules


@dataclass(frozen=True)
class RuleFinding(JsonRecord):
    """
    One rule that applied to one scene.
    """

    scene_id: str
    """
    The scene.
    """

    rule: AnnotationRule
    """
    The rule that applied.
    """

    finding: str
    """
    What made it apply, or what it changed.
    """


@dataclass(frozen=True)
class RuledScene:
    """
    A scene after the rules were applied to it.
    """

    scene: AnnotatedScene
    """
    The scene with every correction made, or as annotated where it is left out.
    """

    exclusions: list[RuleFinding] = field(default_factory=list)
    """
    The exclusions that leave the scene out, none where it is evaluated.
    """

    corrections: list[RuleFinding] = field(default_factory=list)
    """
    The corrections made to it, in the order they were made.
    """

    @property
    def excluded(self) -> bool:
        """
        Whether the scene is left out of the evaluation.
        """
        return bool(self.exclusions)


@dataclass(frozen=True)
class AnnotationRules(JsonRecord):
    """
    Every rule about how Articulate3D's annotation is read.
    """

    exclusions: list[SceneExclusion] = field(default_factory=list)
    """
    The reasons to leave a scene out.
    """

    corrections: list[AnnotationCorrection] = field(default_factory=list)
    """
    The corrections, made in this order.
    """

    def without_spelling_corrections(self) -> AnnotationRules:
        """
        :return: These rules, leaving every part name spelled as its annotator wrote it.
        """
        return replace(
            self,
            corrections=[
                one
                for one in self.corrections
                if not isinstance(one, PartNameCorrection)
            ],
        )

    def apply(self, scene: AnnotatedScene) -> RuledScene:
        """
        :param scene: A scene as annotated.
        :return: The scene with every exclusion that applies to it, and, where none
            does, with every correction made.
        """
        exclusions = [
            RuleFinding(scene_id=scene.scene_id, rule=exclusion, finding=finding)
            for exclusion in self.exclusions
            if (finding := exclusion.finding(scene)) is not None
        ]
        if exclusions:
            return RuledScene(scene=scene, exclusions=exclusions)
        corrections = []
        for correction in self.corrections:
            corrected = correction.correct(scene)
            if corrected is None:
                continue
            scene = corrected.scene
            corrections.append(
                RuleFinding(
                    scene_id=scene.scene_id,
                    rule=correction,
                    finding=corrected.finding,
                )
            )
        return RuledScene(scene=scene, corrections=corrections)

    def apply_to_scenes(
        self,
        scene_ids: list[str],
        articulate3d_directory: Path,
        scannetpp_directory: Path,
    ) -> list[RuledScene]:
        """
        :param scene_ids: The scenes.
        :param articulate3d_directory: The directory holding Articulate3D's annotation
            files.
        :param scannetpp_directory: ScanNet++'s data directory, one folder per scene.
        :return: Every scene after the rules were applied to it, in the given order.
        """
        return [
            self.apply(
                AnnotatedScene.read(
                    scene_id, articulate3d_directory, scannetpp_directory
                )
            )
            for scene_id in scene_ids
        ]


@dataclass(frozen=True)
class RuleFindings(JsonRecord):
    """
    Every rule that applied to a set of scenes, as the paper reports them.
    """

    exclusions: list[RuleFinding] = field(default_factory=list)
    """
    The scenes left out, and why.
    """

    corrections: list[RuleFinding] = field(default_factory=list)
    """
    The corrections made to the scenes kept.
    """

    @classmethod
    def of(cls, ruled_scenes: list[RuledScene]) -> RuleFindings:
        """
        :param ruled_scenes: Scenes the rules were applied to.
        :return: Every rule that applied to them.
        """
        return cls(
            exclusions=[
                finding for scene in ruled_scenes for finding in scene.exclusions
            ],
            corrections=[
                finding for scene in ruled_scenes for finding in scene.corrections
            ],
        )

    def write(self, path: Path) -> None:
        """
        :param path: Where to write the findings.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=2))
