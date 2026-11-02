"""
List the part names Articulate3D's annotators used, so the protocol can rule on each.

Each name is counted by how often it occurs and in how many scenes. Given a protocol,
every name it does not list yet is added to it as undecided, and every ruling it already
holds is kept.

Run ``python -m experiments.warsaw.articulate3d.labels --help`` for its options.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from semantic_digital_twin.adapters.articulate3d_dataset.schema import (
    Articulate3DAnnotationFile,
    Articulate3DPart,
)

from experiments.warsaw.articulate3d.protocol import EvaluationProtocol

# %% how often each name is used


@dataclass(frozen=True)
class PartNameUse:
    """
    How often annotators used one part name.
    """

    part_name: str
    """
    The name, without its position in the part hierarchy.
    """

    parts: int
    """
    How many parts carry it.
    """

    scenes: int
    """
    How many scenes hold at least one of them.
    """


@dataclass(frozen=True)
class PartNameCensus:
    """
    Every part name a set of scenes uses, commonest first.
    """

    uses: list[PartNameUse] = field(default_factory=list)
    """
    How often each name is used.
    """

    @classmethod
    def of(cls, directory: Path, scene_ids: list[str]) -> PartNameCensus:
        """
        :param directory: The directory holding Articulate3D's annotation files.
        :param scene_ids: The scenes to count over.
        :return: How often each part name is used in them.
        """
        parts: Counter = Counter()
        scenes: Counter = Counter()
        for scene_id in scene_ids:
            names = [
                part.part_name
                for part in Articulate3DPart.all_in_file(
                    Articulate3DAnnotationFile.PARTS.path(directory, scene_id)
                )
            ]
            parts.update(names)
            scenes.update(set(names))
        return cls(
            uses=[
                PartNameUse(part_name=name, parts=count, scenes=scenes[name])
                for name, count in sorted(
                    parts.items(), key=lambda one: (-one[1], one[0])
                )
            ]
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for listing part names.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="The directory holding Articulate3D's annotation files",
    )
    parser.add_argument(
        "--scenes",
        type=Path,
        help="A file of scene ids, one per line; every annotated scene by default",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="A protocol to add the names it does not list yet to, made if missing",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    List the part names a set of scenes uses.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the names are listed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parsed = argument_parser().parse_args(arguments)
    scene_ids = (
        parsed.scenes.read_text().split()
        if parsed.scenes
        else Articulate3DAnnotationFile.PARTS.scene_ids(parsed.dataset)
    )
    census = PartNameCensus.of(parsed.dataset, scene_ids)
    logger.info("%s part names over %s scenes", len(census.uses), len(scene_ids))
    for use in census.uses:
        logger.info("  %6s parts  %4s scenes  %s", use.parts, use.scenes, use.part_name)
    if parsed.protocol is None:
        return 0
    protocol = (
        EvaluationProtocol.read(parsed.protocol)
        if parsed.protocol.is_file()
        else EvaluationProtocol()
    )
    protocol.listing(use.part_name for use in census.uses).write(parsed.protocol)
    logger.info("listed in %s", parsed.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
