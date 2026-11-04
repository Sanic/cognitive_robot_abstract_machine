"""
The evaluation protocol as a reader of the paper reads it: every rule with what it does
and why, every scene and part it applied to, and what every part name is taken to mean.

Run ``python -m experiments.warsaw.articulate3d.protocol_report --help`` for its
options.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from semantic_digital_twin.semantic_annotations.part_whole import part_whole_fields
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    load_annotation_modules,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from typing_extensions import Callable

from experiments.warsaw.articulate3d.ground_truth import (
    Expressibility,
    GroundTruthBuilder,
    SceneGroundTruth,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol, PartNameRuling
from experiments.warsaw.articulate3d.rules import (
    AnnotationRule,
    PartNameCorrection,
    RuledScene,
    RuleFinding,
)
from experiments.warsaw.pipeline.templates import PipelineTemplates

# %% the report


@dataclass(frozen=True)
class RuleInstance:
    """
    One scene a rule applied to.
    """

    split: str
    """
    The split the scene belongs to.
    """

    scene_id: str
    """
    The scene.
    """

    finding: str
    """
    What made the rule apply, or what it changed.
    """


@dataclass(frozen=True)
class RuleUse:
    """
    One rule, and every scene it applied to.
    """

    rule: AnnotationRule
    """
    The rule.
    """

    instances: list[RuleInstance]
    """
    The scenes it applied to, by split.
    """

    @property
    def title(self) -> str:
        """
        The rule's class name as words, such as "Mesh changed".
        """
        return re.sub(r"(?<!^)(?=[A-Z])", " ", type(self.rule).__name__).capitalize()


@dataclass(frozen=True)
class PartNameUse:
    """
    One part name, what it is taken to mean, and how many parts carry it per split.
    """

    part_name: str
    """
    The name, as read after the corrections.
    """

    ruling: PartNameRuling
    """
    What the name is taken to mean.
    """

    semantic_class: str | None
    """
    The class it means, where it means one.
    """

    parts: dict[str, int]
    """
    How many parts carry the name, per split.
    """


@dataclass(frozen=True)
class UnheldPair:
    """
    A pair of classes whose relations no field of the ontology can hold.
    """

    whole_class: str
    """
    The class of the annotated whole.
    """

    part_class: str
    """
    The class of the annotated part.
    """

    relations: dict[str, int]
    """
    How many relations of the pair the kept scenes hold, per split.
    """


@dataclass(frozen=True)
class CarryingField:
    """
    A field whose parts carry the whole that holds them.
    """

    whole_class: str
    """
    The class holding the field.
    """

    field_name: str
    """
    The field.
    """

    part_class: str
    """
    The class of the parts it holds.
    """


@dataclass(frozen=True)
class ProtocolReport:
    """
    A protocol, applied to the scenes of every split.
    """

    protocol: EvaluationProtocol
    """
    The protocol.
    """

    ruled_scenes: dict[str, list[RuledScene]]
    """
    Every scene of every split, after the protocol's rules were applied to it.
    """

    ontology: dict[str, type]
    """
    The ontology's classes by name.
    """

    @cached_property
    def ground_truth(self) -> dict[str, list[SceneGroundTruth]]:
        """
        The ground truth of every kept scene, per split.
        """
        builder = GroundTruthBuilder(protocol=self.protocol, ontology=self.ontology)
        return {
            split: [
                builder.build(ruled.scene.scene_id, ruled.scene.parts)
                for ruled in scenes
                if not ruled.excluded
            ]
            for split, scenes in self.ruled_scenes.items()
        }

    @property
    def expressibility(self) -> dict[str, Counter]:
        """
        How many relations of the kept scenes the ontology can hold, and how, per split.
        """
        return {
            split: Counter(
                relation.expressibility
                for scene in scenes
                for relation in scene.relations
            )
            for split, scenes in self.ground_truth.items()
        }

    @property
    def unheld_pairs(self) -> list[UnheldPair]:
        """
        Every pair of classes whose relations no field can hold, most relations first.
        """
        counts: dict[tuple[str, str], Counter] = {}
        for split, scenes in self.ground_truth.items():
            for scene in scenes:
                class_of = {part.part_id: part.semantic_class for part in scene.parts}
                for relation in scene.relations:
                    if relation.expressibility is Expressibility.INEXPRESSIBLE:
                        pair = (class_of[relation.whole], class_of[relation.part])
                        counts.setdefault(pair, Counter())[split] += 1
        return [
            UnheldPair(
                whole_class=whole,
                part_class=part,
                relations={split: count[split] for split in self.ground_truth},
            )
            for (whole, part), count in sorted(
                counts.items(), key=lambda one: -sum(one[1].values())
            )
        ]

    @property
    def carrying_fields(self) -> list[CarryingField]:
        """
        Every field whose parts carry their whole, of the classes the protocol rules
        part names to.
        """
        ruled_classes = sorted(
            {
                one.semantic_class
                for one in self.protocol.part_names
                if one.semantic_class
            }
        )
        return [
            CarryingField(
                whole_class=class_name,
                field_name=one.field_name,
                part_class=one.part.__name__,
            )
            for class_name in ruled_classes
            for one in part_whole_fields(self.ontology[class_name])
            if one.part_carries_whole
        ]

    @property
    def exclusions(self) -> list[RuleUse]:
        """
        Every exclusion of the protocol, with the scenes it leaves out.
        """
        return [
            self._use(rule, lambda scene: scene.exclusions)
            for rule in self.protocol.annotation_rules.exclusions
        ]

    @property
    def general_corrections(self) -> list[RuleUse]:
        """
        Every correction made to all names alike, with the scenes it changed.
        """
        return [
            self._use(rule, lambda scene: scene.corrections)
            for rule in self.protocol.annotation_rules.corrections
            if not isinstance(rule, PartNameCorrection)
        ]

    @property
    def name_corrections(self) -> list[tuple[PartNameCorrection, dict[str, int]]]:
        """
        Every correction of one name, with how many scenes it changed per split.
        """
        return [
            (
                rule,
                Counter(
                    instance.split
                    for instance in self._use(
                        rule, lambda scene: scene.corrections
                    ).instances
                ),
            )
            for rule in self.protocol.annotation_rules.corrections
            if isinstance(rule, PartNameCorrection)
        ]

    @property
    def part_names(self) -> list[PartNameUse]:
        """
        Every part name the protocol rules on, by class and then by name.
        """
        parts = {
            split: Counter(
                part.part_name
                for ruled in scenes
                if not ruled.excluded
                for part in ruled.scene.parts
            )
            for split, scenes in self.ruled_scenes.items()
        }
        return sorted(
            (
                PartNameUse(
                    part_name=meaning.part_name,
                    ruling=meaning.ruling,
                    semantic_class=meaning.semantic_class,
                    parts={split: parts[split][meaning.part_name] for split in parts},
                )
                for meaning in self.protocol.part_names
            ),
            key=lambda use: (use.semantic_class or "", use.part_name),
        )

    def _use(
        self,
        rule: AnnotationRule,
        findings: Callable[[RuledScene], list[RuleFinding]],
    ) -> RuleUse:
        """
        :param rule: A rule of the protocol.
        :param findings: Reads the findings of the kind the rule makes off a scene.
        :return: The rule with every scene it applied to.
        """
        return RuleUse(
            rule=rule,
            instances=[
                RuleInstance(
                    split=split, scene_id=finding.scene_id, finding=finding.finding
                )
                for split, scenes in self.ruled_scenes.items()
                for scene in scenes
                for finding in findings(scene)
                if finding.rule == rule
            ],
        )

    def markdown(self) -> str:
        """
        :return: The report as one Markdown document.
        """
        return PipelineTemplates(
            directory=Path(__file__).resolve().parent / "templates"
        ).render(
            "protocol_report.md.jinja",
            ruled_scenes=self.ruled_scenes,
            exclusions=self.exclusions,
            general_corrections=self.general_corrections,
            name_corrections=self.name_corrections,
            part_names=self.part_names,
            ruling=PartNameRuling,
            expressibility=self.expressibility,
            unheld_pairs=self.unheld_pairs,
            carrying_fields=self.carrying_fields,
            kinds=Expressibility,
        )


# %% command-line entry point


def argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for writing the report.
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
        "--protocol", type=Path, required=True, help="The protocol to report"
    )
    parser.add_argument(
        "--splits",
        type=Path,
        nargs="+",
        required=True,
        help="Files of scene ids, one per line, each a split named by its file name",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the report"
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """
    Write the report of a protocol.

    :param arguments: Command-line arguments without the program name.
    :return: Zero once the report is written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = argument_parser().parse_args(arguments)
    protocol = EvaluationProtocol.read(parsed.protocol)
    load_annotation_modules()
    report = ProtocolReport(
        protocol=protocol,
        ontology=annotation_classes(SemanticAnnotation),
        ruled_scenes={
            split.stem: protocol.annotation_rules.apply_to_scenes(
                split.read_text().split(), parsed.dataset, parsed.scannetpp
            )
            for split in parsed.splits
        },
    )
    parsed.output.write_text(report.markdown())
    logging.getLogger(__name__).info("wrote %s", parsed.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
