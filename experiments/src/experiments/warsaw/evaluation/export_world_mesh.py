"""
Write a world as a scene a modelling tool can open, one named object per body.

RViz shows a world but is a poor place to pick it apart. A GLB opens in Blender with
every body its own object in the outliner: click one and its name is in front of you,
hide it and see what is behind, and no ROS is involved at all.

Each node is named ``<body> [<Class>]``, so a click says both what the run called the
object and what it decided the object is. Only the exported scene is named that way; the
world itself is read afresh and never written back.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from semantic_digital_twin.adapters.world_mesh_exporter import (
    GeometrySource,
    WorldMeshExporter,
    WorldMeshExtractor,
    WorldMeshSnapshot,
)
from semantic_digital_twin.world import World
from typing_extensions import List, Optional

from experiments.warsaw.painting import Coloring, paint

# %% naming a body by what it is


def named_by_class(snapshot: WorldMeshSnapshot) -> WorldMeshSnapshot:
    """
    Name every body for what it is as well as for which one it is.

    :param snapshot: What was extracted from the world.
    :return: The same, with each body named ``<body> [<Class>]`` where the world named
        it anything.
    """
    class_of = {
        annotation.root_body_identifier: annotation.type_name
        for annotation in snapshot.semantic_annotations
        if annotation.root_body_identifier is not None
    }
    return replace(
        snapshot,
        body_meshes=tuple(
            (
                replace(body, name=f"{body.name} [{class_of[body.identifier]}]")
                if body.identifier in class_of
                else body
            )
            for body in snapshot.body_meshes
        ),
    )


# %% writing one world out


def write_world_mesh(
    world: World,
    output_directory: Path,
    coloring: Coloring = Coloring.BY_CLASS,
    geometry_source: GeometrySource = GeometrySource.VISUAL_WITH_COLLISION_FALLBACK,
) -> Path:
    """
    Write a world as a GLB scene beside the manifest saying what each body is.

    :param world: The world to write.
    :param output_directory: Where to write it.
    :param coloring: What the bodies are painted by.
    :param geometry_source: Which of a body's shape collections to take.
    :return: The GLB that was written.
    """
    paint(world, coloring)
    snapshot = named_by_class(
        WorldMeshExtractor(geometry_source=geometry_source).extract(world)
    )
    return WorldMeshExporter().export(snapshot, output_directory).scene


def world_of_run(directory: Path) -> World:
    """
    Read back the world a finished run built.

    :param directory: The run's directory.
    :return: The annotated world it wrote.
    """
    # The run's own classes and schema have to be in place before anything reaches the
    # ORM, which is why these are imported here rather than at the top of the module.
    from experiments.warsaw.pipeline.database.orm_rebuild import OrmRebuild
    from experiments.warsaw.pipeline.database.run_schema import RunSchema
    from experiments.warsaw.pipeline.run import Run, RunFile
    from experiments.warsaw.pipeline.run_classes import GeneratedClasses
    from experiments.warsaw.pipeline.records import SplitRecord

    generated = GeneratedClasses(directory=directory)
    if generated.were_generated:
        logging.getLogger(__name__).info(
            "rebuilding the ORM for this run's classes ..."
        )
        OrmRebuild(directory=directory).run_in_new_interpreter()
    generated.use()
    RunSchema.for_run(directory).use()

    from experiments.warsaw.pipeline.database.world_store import WorldStore

    split = Run(directory=directory).read_record(RunFile.SPLIT, SplitRecord)
    return WorldStore().read(split.annotated_world_id)


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for writing a world out as a scene.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="A finished run's directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="The directory to write into"
    )
    parser.add_argument(
        "--coloring",
        type=Coloring,
        choices=list(Coloring),
        default=Coloring.BY_CLASS,
        help="What the bodies are painted by",
    )
    return parser


def main(arguments: Optional[List[str]] = None) -> int:
    """
    Write a finished run's world as a scene a modelling tool can open.

    :param arguments: Command-line arguments without the program name.
    :return: Zero after the scene is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    world = world_of_run(parsed.run)
    written = write_world_mesh(world, parsed.output, parsed.coloring)
    logging.getLogger(__name__).info(
        "wrote %s bodies to %s -- open it with Blender's glTF 2.0 import",
        len(world.bodies),
        written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
