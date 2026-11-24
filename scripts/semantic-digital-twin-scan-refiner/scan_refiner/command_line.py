"""
Local project initialization, automatic alignment and browser inspection commands.
"""

from __future__ import annotations

# %% Imports
import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import trimesh

from scan_refiner.inspection import InspectionViewer
from scan_refiner.exceptions import ExistingProject, InvalidEvidence
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole
from scan_refiner.registration import RegistrationSearch
from scan_refiner.world import BodyMesh, WorldSnapshot
from scan_refiner.scan_sources import E57PointSource, PreviewPointSource
from scan_refiner.spatial_index import IndexFile, ScanRegion, SpatialScanIndex
from scan_refiner.point_records import PointField
from scan_refiner.records import ProjectFile
from scan_refiner.web_service import InspectionService
from semantic_digital_twin.adapters.urdf import URDFParser


# %% Command vocabulary
class Command(StrEnum):
    """
    Project operations available without a browser or language model.
    """

    INIT = "init"
    ALIGN = "align"
    VIEW = "view"
    ACCEPT = "accept"
    SERVE = "serve"
    DEMO = "demo"
    INDEX = "index"
    CROP = "crop"
    UNDO = "undo"
    REDO = "redo"
    HISTORY = "history"


class Option(StrEnum):
    """
    Public command-line switches of the initial local workflow.
    """

    PROJECT = "--project"
    SCAN = "--scan"
    SCALE = "--scan-scale"
    URDF = "--urdf"
    PROVIDER = "--world-provider"
    UPRIGHT = "--upright"
    CELL_WIDTH = "--cell-width"
    YAW_STEP = "--yaw-step"
    GEOMETRY = "--geometry"
    CANDIDATE = "--candidate"
    HEATMAP = "--heatmap"
    DISPLAY_COUNT = "--display-count"
    PORT = "--port"
    E57 = "--e57"
    PREVIEW = "--preview"
    READER_PYTHON = "--reader-python"
    BATCH_CAPACITY = "--batch-capacity"
    LOWER = "--lower"
    UPPER = "--upper"
    OUTPUT = "--output"
    MAXIMUM_COUNT = "--maximum-count"


# %% Command dispatch
@dataclass(kw_only=True)
class CommandLine:
    """
    User-invoked operations on one local scan-refinement project.
    """

    @classmethod
    def run(cls) -> None:
        """
        Parse an operation and print the resulting artifact or local viewer URL.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in Command:
            command_parser = subparsers.add_parser(command)
            command_parser.add_argument(Option.PROJECT, type=Path, required=True)
            if command == Command.INIT:
                command_parser.add_argument(Option.SCAN, type=Path, required=True)
                command_parser.add_argument(
                    Option.SCALE,
                    type=float,
                    default=1.0,
                    help="Source-to-metre multiplier.",
                )
                model = command_parser.add_mutually_exclusive_group(required=True)
                model.add_argument(Option.URDF, type=Path)
                model.add_argument(
                    Option.PROVIDER,
                    help="Trusted module.path:WorldClass with create_world().",
                )
            elif command == Command.ALIGN:
                command_parser.add_argument(
                    Option.UPRIGHT,
                    action="store_true",
                    required=True,
                    help="Confirm both inputs already have Z aligned with gravity.",
                )
                command_parser.add_argument(Option.CELL_WIDTH, type=float, default=0.12)
                command_parser.add_argument(Option.YAW_STEP, type=float, default=10.0)
                command_parser.add_argument(
                    Option.GEOMETRY,
                    choices=list(GeometryRole),
                    default=GeometryRole.VISUAL,
                )
            elif command in (Command.VIEW, Command.ACCEPT):
                command_parser.add_argument(
                    Option.CANDIDATE, type=int, required=command == Command.ACCEPT
                )
                if command == Command.VIEW:
                    command_parser.add_argument(Option.HEATMAP, action="store_true")
                    command_parser.add_argument(
                        Option.DISPLAY_COUNT, type=int, default=20000
                    )
            elif command == Command.SERVE:
                command_parser.add_argument(Option.PORT, type=int, default=8765)
            elif command == Command.INDEX:
                source = command_parser.add_mutually_exclusive_group(required=True)
                source.add_argument(Option.E57, type=Path)
                source.add_argument(Option.PREVIEW, action="store_true")
                command_parser.add_argument(Option.READER_PYTHON, type=Path)
                command_parser.add_argument(
                    Option.BATCH_CAPACITY, type=int, default=250000
                )
                command_parser.add_argument(Option.CELL_WIDTH, type=float, default=1.0)
            elif command == Command.CROP:
                command_parser.add_argument(
                    Option.LOWER, nargs=3, type=float, required=True
                )
                command_parser.add_argument(
                    Option.UPPER, nargs=3, type=float, required=True
                )
                command_parser.add_argument(Option.OUTPUT, type=Path, required=True)
                command_parser.add_argument(
                    Option.MAXIMUM_COUNT, type=int, default=50000
                )
        arguments = parser.parse_args()
        cls().execute(arguments=arguments)

    def execute(self, *, arguments: argparse.Namespace) -> None:
        """
        Dispatch parsed commands without changing external model or scan sources.
        """
        project = RefinementProject(directory=arguments.project.resolve())
        command = Command(arguments.command)
        if command == Command.INIT:
            if arguments.urdf is not None:
                snapshot = WorldSnapshot.from_world(
                    world=URDFParser.from_file(
                        str(arguments.urdf.resolve())
                    ).create_world()
                )
                source = str(arguments.urdf.resolve())
            else:
                snapshot = WorldSnapshot.from_provider(
                    identifier=arguments.world_provider
                )
                source = arguments.world_provider
            project = RefinementProject.create(
                directory=project.directory,
                scan_path=arguments.scan,
                snapshot=snapshot,
                model_source=source,
                scan_scale=arguments.scan_scale,
            )
        elif command == Command.DEMO:
            project = self.create_demo(directory=project.directory)
            project.align(search=RegistrationSearch())
        elif command == Command.ALIGN:
            result = project.align(
                search=RegistrationSearch(
                    cell_width=arguments.cell_width, yaw_step_degrees=arguments.yaw_step
                ),
                role=GeometryRole(arguments.geometry),
            )
            for index, candidate in enumerate(result.candidates):
                print(
                    f"{index}: support={candidate.support_fraction:.3f}, median={candidate.median_distance:.4f} m"
                )
        elif command == Command.UNDO:
            project.undo()
        elif command == Command.REDO:
            project.redo()
        elif command == Command.HISTORY:
            history = project.history.load()
            for revision in history.past:
                print(revision.label)
            print(
                f"Current: {history.current.label}; {len(history.future)} redo actions"
            )
            return
        elif command == Command.ACCEPT:
            project.accept(candidate_index=arguments.candidate)
        elif command == Command.VIEW:
            print(
                InspectionViewer(
                    project=project, display_count=arguments.display_count
                ).write(candidate_index=arguments.candidate, heatmap=arguments.heatmap)
            )
            return
        elif command == Command.INDEX:
            self.index_scan(project=project, arguments=arguments)
            return
        elif command == Command.CROP:
            index = SpatialScanIndex(
                directory=project.directory / ProjectFile.SCAN_INDEX
            )
            region = ScanRegion(
                lower=np.array(arguments.lower), upper=np.array(arguments.upper)
            )
            sampled = index.sample(region=region, maximum_count=arguments.maximum_count)
            if arguments.output.exists():
                raise ExistingProject(directory=arguments.output)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = arguments.output.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, **{str(key): sampled.records[key] for key in PointField}
                )
            temporary.replace(arguments.output)
            print(
                f"Saved {len(sampled.records):,} original records to {arguments.output}"
            )
            return
        elif command == Command.SERVE:
            with InspectionService(project=project).create_server(
                port=arguments.port
            ) as server:
                print(f"http://127.0.0.1:{server.server_port}/", flush=True)
                server.serve_forever()
            return
        print(InspectionViewer(project=project).write())

    def index_scan(
        self, *, project: RefinementProject, arguments: argparse.Namespace
    ) -> None:
        """
        Attach a new spatial index without replacing a completed or partial import.
        """
        directory = project.directory / ProjectFile.SCAN_INDEX
        if directory.exists():
            raise ExistingProject(directory=directory)
        if arguments.e57 is not None:
            if arguments.reader_python is None:
                raise InvalidEvidence(
                    reason="E57 indexing requires --reader-python pointing to an existing pye57 environment."
                )
            source = E57PointSource(
                source=arguments.e57.resolve(),
                interpreter=arguments.reader_python.absolute(),
                metadata=directory / IndexFile.STATIONS,
                batch_capacity=arguments.batch_capacity,
            )
            identity = (
                f"{arguments.e57.resolve()}#sha256={project.digest(path=arguments.e57)}"
            )
        else:
            source = PreviewPointSource(
                preview=project.scan, batch_capacity=arguments.batch_capacity
            )
            identity = f"preview#sha256={project.record.scan_sha256}#scale={project.record.scan_scale}"
        index = SpatialScanIndex.build(
            directory=directory,
            batches=source.batches(),
            source_identity=identity,
            cell_width=arguments.cell_width,
        )
        print(
            f"Completed spatial index: {index.record.point_count:,} records at {directory}"
        )

    def create_demo(self, *, directory: Path) -> RefinementProject:
        """
        Create synthetic scan evidence for a portable end-to-end smoke test.
        """
        from_scan = directory.parent / f"{directory.name}-synthetic-scan.npz"
        if directory.exists() or from_scan.exists():
            raise ExistingProject(directory=directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        meshes = []
        for index, (extents, translation) in enumerate(
            (
                ((4.0, 3.0, 0.1), (0, 0, -0.05)),
                ((0.8, 1.2, 1.5), (-1.0, 0.6, 0.75)),
                ((1.2, 0.5, 0.7), (0.9, -0.8, 0.35)),
            )
        ):
            mesh = trimesh.creation.box(extents)
            mesh.apply_translation(translation)
            for role in GeometryRole:
                meshes.append(
                    BodyMesh(
                        body_name=f"demo/object-{index}", role=role, mesh=mesh.copy()
                    )
                )
        snapshot = WorldSnapshot(meshes=meshes)
        scan_T_root = trimesh.transformations.rotation_matrix(np.radians(37), (0, 0, 1))
        scan_T_root[:3, 3] = (3, -1.7, 0.4)
        points = trimesh.transform_points(
            snapshot.samples(role=GeometryRole.VISUAL, count=16000), scan_T_root
        )
        points += np.random.default_rng(0).normal(0, 0.002, size=points.shape)
        colors = np.tile(np.array((110, 150, 175), dtype=np.uint8), (len(points), 1))
        np.savez_compressed(from_scan, points=points, colors=colors)
        return RefinementProject.create(
            directory=directory,
            scan_path=from_scan,
            snapshot=snapshot,
            model_source="synthetic-demo",
        )
