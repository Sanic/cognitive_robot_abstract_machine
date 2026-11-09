"""Build a portable, database-independent spatial inspection of two exported worlds."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

import numpy as np
import trimesh
from experiments.warsaw.bases import HasLogger, JsonRecord
from experiments.warsaw.evaluation.size import ObjectSize
from experiments.warsaw.evaluation.spatial_comparison import (
    ComparisonObject,
    ComparisonSettings,
    SpatialComparison,
)
from experiments.warsaw.pipeline.records import SplitRecord
from experiments.warsaw.pipeline.run import Run, RunFile
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from typing_extensions import Any  # noqa: UP035

# %% export payload and file names


class ExportKey(StrEnum):
    """Fields read from the existing mesh export format."""

    IDENTIFIER = "id"
    NAME = "name"
    PARENT = "parent_id"
    ANNOTATIONS = "direct_semantic_annotation_ids"
    TYPE = "type_name"
    MESH = "mesh_file"
    BODIES = "bodies"
    SEMANTICS = "semantic_annotations"
    SCENE = "scene"


class ComparisonFile(StrEnum):
    """Files and directories of a portable comparison bundle."""

    MANIFEST = "manifest.json"
    TABLE = "comparison.json"
    PREDICTION = "prediction"
    GROUND_TRUTH = "ground_truth"
    ALIGNMENT = "alignment.json"


@dataclass(frozen=True)
class ExportBody:
    """Identity and geometry location in a mesh export manifest."""

    identifier: int
    """The export-local identifier."""
    name: str
    """The display name; matching never parses it."""
    parent_identifier: int | None
    """The exported kinematic parent."""
    annotation_identifiers: list[int]
    """Own semantic annotations, excluding enclosing annotations."""
    mesh_file: str | None
    """The world-aligned PLY path, absent for bodies without geometry."""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExportBody:
        """Mirror the mesh export's field names at its file boundary."""
        return cls(
            identifier=payload[ExportKey.IDENTIFIER],
            name=payload[ExportKey.NAME],
            parent_identifier=payload[ExportKey.PARENT],
            annotation_identifiers=payload[ExportKey.ANNOTATIONS],
            mesh_file=payload[ExportKey.MESH],
        )


@dataclass(frozen=True)
class LandmarkAlignment(JsonRecord):
    """A fitted transform from raw scan coordinates into metric GT coordinates."""

    matrix: list[list[float]]
    """The similarity transform, including the reconstruction's scale correction."""


# %% geometry read from existing exports


@dataclass
class ExportedWorld:
    """The objects and measured geometry of an existing world mesh export."""

    directory: Path
    """The directory containing its manifest and world-aligned body meshes."""
    bodies: list[ExportBody]
    """Every body, including the synthetic world root."""
    annotation_classes: dict[int, str]
    """Semantic annotation identifiers resolved once at the manifest boundary."""
    scene_file: str
    """The GLB path relative to the export directory."""
    manifest_digest: str
    """The SHA-256 identity of the exact source manifest."""

    @classmethod
    def read(cls, directory: Path) -> ExportedWorld:
        """Read an export without importing worlds, generated classes or the ORM."""
        data = (directory / ComparisonFile.MANIFEST).read_bytes()
        payload = json.loads(data)
        return cls(
            directory=directory,
            bodies=[ExportBody.from_json(body) for body in payload[ExportKey.BODIES]],
            annotation_classes={
                annotation[ExportKey.IDENTIFIER]: annotation[ExportKey.TYPE]
                for annotation in payload[ExportKey.SEMANTICS]
            },
            scene_file=payload[ExportKey.SCENE],
            manifest_digest=sha256(data).hexdigest(),
        )

    def measure(self, ground_truth_T_export: np.ndarray) -> list[ComparisonObject]:
        """Measure annotated objects in GT coordinates, retaining geometry-free groups."""
        meshes = {}
        for body in self.bodies:
            if body.mesh_file is None:
                continue
            mesh = trimesh.load_mesh(self.directory / body.mesh_file, process=False)
            mesh.apply_transform(ground_truth_T_export)
            meshes[body.identifier] = mesh
        by_identifier = {body.identifier: body for body in self.bodies}
        eligible = {
            body.identifier
            for body in self.bodies
            if body.annotation_identifiers and body.parent_identifier is not None
        }
        measured = []
        for body in self.bodies:
            if body.identifier not in eligible:
                continue
            mesh = meshes.get(body.identifier)
            if mesh is None:
                descendants = [body.identifier]
                for identifier in descendants:
                    descendants.extend(
                        child.identifier
                        for child in self.bodies
                        if child.parent_identifier == identifier
                    )
                pieces = [
                    meshes[identifier]
                    for identifier in descendants
                    if identifier in meshes
                ]
                mesh = trimesh.util.concatenate(pieces) if pieces else None
            parent = by_identifier.get(body.parent_identifier)
            while parent is not None and parent.identifier not in eligible:
                parent = by_identifier.get(parent.parent_identifier)
            measured.append(
                ComparisonObject(
                    identifier=body.identifier,
                    name=body.name,
                    classes=[
                        self.annotation_classes[identifier]
                        for identifier in body.annotation_identifiers
                    ],
                    parent_identifier=parent.identifier if parent is not None else None,
                    bounds=mesh.bounds.tolist() if mesh is not None else None,
                    size=ObjectSize.of(mesh),
                )
            )
        return measured

    def copy_view(self, directory: Path) -> None:
        """Copy the manifest and GLB required by the browser into a comparison bundle."""
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.directory / ComparisonFile.MANIFEST,
            directory / ComparisonFile.MANIFEST,
        )
        target = directory / self.scene_file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.directory / self.scene_file, target)


# %% portable comparison products


@dataclass(frozen=True)
class ComparisonBundle(JsonRecord):
    """The saved matching table and exact exported inputs used to produce it."""

    comparison: SpatialComparison
    """Geometry-ranked provisional associations and disagreement flags."""
    prediction_objects: list[ComparisonObject]
    """The measured prediction objects in GT coordinates."""
    ground_truth_objects: list[ComparisonObject]
    """The measured GT objects, including objects beyond the reconstructed area."""
    ground_truth_T_prediction: list[list[float]]
    """The transform from the pipeline export frame into metric GT coordinates."""
    prediction_source: str
    """The source export's absolute directory, for provenance."""
    ground_truth_source: str
    """The selected GT export's absolute directory, for provenance."""
    prediction_digest: str
    """The SHA-256 identity of the source prediction manifest."""
    ground_truth_digest: str
    """The SHA-256 identity of the source GT manifest."""


@dataclass
class ComparisonExporter(HasLogger):
    """Create a comparison beside the experiment without modifying either input world."""

    run_directory: Path
    """The completed pipeline run whose exported world is inspected."""
    ground_truth_directory: Path
    """The selected GT mesh export."""
    alignment_file: Path
    """The accepted raw-scan-to-GT landmark fit."""
    settings: ComparisonSettings = field(default_factory=ComparisonSettings)
    """The inspection policy, saved together with the resulting table."""

    def write(self, output_directory: Path) -> Path:
        """Align and compare the exports, then save their portable browser inputs."""
        run = Run(directory=self.run_directory)
        split = run.read_record(RunFile.SPLIT, SplitRecord)
        alignment = LandmarkAlignment.from_json(
            json.loads(self.alignment_file.read_text())
        )
        ground_truth_T_prediction = np.asarray(
            alignment.matrix
        ) @ HomogeneousTransformationMatrix.invert_numeric_matrix(
            np.asarray(split.world_T_source)
        )
        prediction = ExportedWorld.read(run.path(RunFile.WORLD_MESH))
        ground_truth = ExportedWorld.read(self.ground_truth_directory)
        self.logger.info("measuring the prediction in metric GT coordinates ...")
        predicted_objects = prediction.measure(ground_truth_T_prediction)
        self.logger.info("measuring the ground truth ...")
        ground_truth_objects = ground_truth.measure(np.eye(4))
        comparison = SpatialComparison.between(
            predicted_objects, ground_truth_objects, self.settings
        )
        bundle = ComparisonBundle(
            comparison=comparison,
            prediction_objects=predicted_objects,
            ground_truth_objects=ground_truth_objects,
            ground_truth_T_prediction=ground_truth_T_prediction.tolist(),
            prediction_source=str(prediction.directory.resolve()),
            ground_truth_source=str(ground_truth.directory.resolve()),
            prediction_digest=prediction.manifest_digest,
            ground_truth_digest=ground_truth.manifest_digest,
        )
        prediction.copy_view(output_directory / ComparisonFile.PREDICTION)
        ground_truth.copy_view(output_directory / ComparisonFile.GROUND_TRUTH)
        shutil.copy2(self.alignment_file, output_directory / ComparisonFile.ALIGNMENT)
        path = output_directory / ComparisonFile.TABLE
        path.write_text(json.dumps(bundle.to_json(), indent=2))
        return path


# %% command-line entry point


class ComparisonArgument(StrEnum):
    """The input and output directories accepted by the comparison command."""

    RUN = "run"
    GROUND_TRUTH = "ground_truth"
    ALIGNMENT = "alignment"
    OUTPUT = "output"


def main() -> None:
    """Write a portable comparison from a run, GT export and landmark alignment."""
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ComparisonArgument:
        parser.add_argument(argument.value, type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = ComparisonExporter(
        arguments.run, arguments.ground_truth, arguments.alignment
    ).write(arguments.output)
    logging.getLogger(__name__).info("comparison written to %s", written)


if __name__ == "__main__":
    main()
