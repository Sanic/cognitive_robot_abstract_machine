"""
The report of a protocol: every rule with what it does and why, and where it applied.
"""

from __future__ import annotations

from collections import Counter

import pytest

from experiments.warsaw.articulate3d.ground_truth import (
    Expressibility,
    GroundTruthBuilder,
)
from experiments.warsaw.articulate3d.protocol import EvaluationProtocol
from experiments.warsaw.articulate3d.protocol_report import main
from experiments.warsaw.articulate3d.rules import (
    AnnotatedScene,
    AnnotationRules,
    MeshMissing,
    MeshSize,
    PartNameCorrection,
)

from .scenes import (
    SceneName,
    kitchen_corner_parts,
    kitchen_corner_protocol,
    part,
    row_of_squares,
)

# %% a report of two scenes


def report_rules() -> AnnotationRules:
    """
    :return: A rule leaving out scenes without a mesh, and a correction of one name.
    """
    return AnnotationRules(
        exclusions=[MeshMissing()],
        corrections=[
            PartNameCorrection(
                written="unknown", meant="handle", mistake="a spelling mistake"
            )
        ],
    )


@pytest.fixture
def report(dataset, tmp_path):
    """
    :return: The report of the kitchen corner and of a scene without a mesh, each its
        own split.
    """
    dataset.write_scene(
        SceneName.KITCHEN_CORNER, kitchen_corner_parts(), row_of_squares(8)
    )
    dataset.write_scene(
        "hallway",
        [part(1, "door.1.1", [0])],
        mesh=None,
        recorded_mesh_size=MeshSize(vertices=3, faces=1),
    )
    protocol_path = tmp_path / "protocol.json"
    EvaluationProtocol(
        part_names=kitchen_corner_protocol().part_names,
        annotation_rules=report_rules(),
    ).write(protocol_path)
    splits = []
    for split, scene_id in (("train", SceneName.KITCHEN_CORNER), ("val", "hallway")):
        splits.append(tmp_path / f"{split}.txt")
        splits[-1].write_text(scene_id)
    output = tmp_path / "report.md"
    main(
        [
            "--dataset",
            str(dataset.articulate3d_directory),
            "--scannetpp",
            str(dataset.scannetpp_directory),
            "--protocol",
            str(protocol_path),
            "--splits",
            *map(str, splits),
            "--output",
            str(output),
        ]
    )
    return output.read_text()


# %% what the report says


@pytest.mark.parametrize(
    "rule",
    report_rules().exclusions + report_rules().corrections,
    ids=["exclusion", "correction"],
)
def test_every_rule_is_stated_with_its_reason(report, rule):
    assert (rule.statement in report, rule.reason in report) == (True, True)


def test_a_scene_left_out_is_listed_with_its_split_and_finding(report):
    finding = MeshMissing().finding(
        AnnotatedScene(
            scene_id="hallway",
            parts=[],
            articulations=[],
            recorded_mesh_size=MeshSize(vertices=3, faces=1),
            mesh_size=None,
        )
    )

    assert f"| val | `hallway` | {finding} |" in report


def test_the_relations_of_the_kept_scenes_are_counted_by_what_the_ontology_makes_of_them(
    report, ontology
):
    protocol = EvaluationProtocol(
        part_names=kitchen_corner_protocol().part_names,
        annotation_rules=report_rules(),
    )
    corrected = report_rules().apply(
        AnnotatedScene(
            scene_id=SceneName.KITCHEN_CORNER,
            parts=kitchen_corner_parts(),
            articulations=[],
            recorded_mesh_size=MeshSize(vertices=48, faces=16),
            mesh_size=MeshSize(vertices=48, faces=16),
        )
    )
    built = GroundTruthBuilder(protocol=protocol, ontology=ontology).build(
        SceneName.KITCHEN_CORNER, corrected.scene.parts
    )
    counts = Counter(relation.expressibility for relation in built.relations)

    assert (
        f"| train | {counts[Expressibility.EXPRESSIBLE]} | {counts[Expressibility.CARRIED]}"
        f" | {counts[Expressibility.INEXPRESSIBLE]} | {counts[Expressibility.UNDETERMINED]} |"
    ) in report
