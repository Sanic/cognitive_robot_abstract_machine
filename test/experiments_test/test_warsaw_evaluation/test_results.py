"""
What any finished run comes to, and tables of many of them for a document.

The runs here are written by hand, each holding only what the results read, so every
number a test asserts is the one its run was written with. Nothing here asks where the
scene came from: a run of a scene with no ground truth reports all but its scores.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

from experiments.experiment_definitions import MeanAndStandardDeviation
from experiments.warsaw.articulate3d.score import (
    ClassAgreement,
    FaceCorrespondence,
    JudgedClass,
    LinkAgreement,
    SceneScore,
    ScoreFile,
)
from experiments.warsaw.evaluation.exceptions import RunNotFinishedError
from experiments.warsaw.evaluation.graph import EvaluationGraph, EvaluationNode
from experiments.warsaw.evaluation.results import (
    ClassResult,
    EffortResult,
    LinkResult,
    ModelUse,
    ResultTables,
    RunComparison,
    RunGroup,
    RunMeasurements,
    StructureResult,
    main,
)
from experiments.warsaw.pipeline.model_calls import ModelCallTrace
from experiments.warsaw.pipeline.records import (
    Adjudications,
    BodyAnswer,
    Classifications,
    LabelAnswer,
    RefusedMount,
    SplitBody,
    SplitRecord,
    StepDuration,
    StepDurations,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import Model
from experiments.warsaw.scene_split import Pairing

# %% a run written by hand


def written_run(
    runs_directory: pathlib.Path,
    name: str,
    model: str = "a/model",
    scene: str = "kitchen",
    commit: str = "abcdef0123456789",
    usage: dict | None = None,
    seconds: float = 10.0,
    pictures: bool = True,
    contested_faces_pictured: bool | None = None,
) -> Run:
    """
    :return: A finished run with two labels, two bodies, one part mounted and one refused,
        and one question answered at its second attempt.
    """
    run = Run(directory=runs_directory / name)
    run.directory.mkdir(parents=True)
    run.write_json(
        RunFile.PROVENANCE,
        {
            "settings": {
                "scene_directory": f"/datasets/{scene}",
                "model": model,
                "show_the_pictures": pictures,
                **(
                    {}
                    if contested_faces_pictured is None
                    else {"show_the_contested_faces": contested_faces_pictured}
                ),
                "runs_directory": f"/somewhere/{name}",
            },
            "source": {"commit": commit, "dirty": False, "patch_sha256": None},
        },
    )
    run.write_record(
        RunFile.VOCABULARY,
        Vocabulary(
            scene="kitchen.ply",
            model=model,
            labels=[
                LabelAnswer(label="cabinet", class_name="Cabinet"),
                LabelAnswer(label="tray", class_name="Tray", is_new_class=True),
            ],
        ),
    )
    run.write_record(
        RunFile.ADJUDICATIONS, Adjudications(scene="kitchen.ply", model=model)
    )
    tray_in_cabinet = Pairing(whole="cabinet_1", part="tray_1", field_name="")
    run.write_record(
        RunFile.SPLIT,
        SplitRecord(
            scene="kitchen.ply",
            bodies=[
                SplitBody(name="cabinet_1", label="cabinet", faces=4, body_id="a"),
                SplitBody(name="tray_1", label="tray", faces=2, body_id="b"),
            ],
            pairings=[
                tray_in_cabinet,
                Pairing(whole="tray_1", part="cabinet_1", field_name=""),
            ],
            refused=[RefusedMount(pairing=tray_in_cabinet, reason="does not fit")],
        ),
    )
    run.write_record(
        RunFile.CLASSIFICATIONS,
        Classifications(
            scene="kitchen.ply",
            model=model,
            bodies=[
                BodyAnswer(name="cabinet_1", label="cabinet", class_name="Cabinet"),
                BodyAnswer(name="tray_1", label="tray", class_name="Tray"),
            ],
        ),
    )
    run.write_record(
        RunFile.EVALUATION_GRAPH,
        EvaluationGraph(
            source_world_id=None,
            annotated_world_id=None,
            nodes=[
                EvaluationNode(
                    name="cabinet_1",
                    input_label="cabinet",
                    predicted_class="Cabinet",
                    faces=4,
                    body_id="a",
                    annotation_applied=True,
                    parent=None,
                ),
                EvaluationNode(
                    name="tray_1",
                    input_label="tray",
                    predicted_class="Tray",
                    faces=2,
                    body_id="b",
                    annotation_applied=False,
                    parent=None,
                ),
            ],
        ),
    )
    run.write_record(
        RunFile.STEP_DURATIONS,
        StepDurations(
            scene="kitchen.ply",
            steps=[
                StepDuration(step="measure", seconds=seconds),
                StepDuration(step="ask", seconds=2.0),
            ],
        ),
    )
    for attempt in (1, 2):
        attempts = run.path(RunFile.MODEL_CALLS) / "vocabulary_answers" / "cabinet"
        attempts.mkdir(parents=True, exist_ok=True)
        trace = ModelCallTrace(
            question="cabinet",
            attempt=attempt,
            requested_model=model,
            system_prompt="say what this is",
            message_parts=[],
            response={"usage": usage} if usage else {},
            problems=[],
            started_at="2026-11-06T10:00:00+00:00",
            elapsed_seconds=1.5,
        )
        (attempts / f"attempt_{attempt}.json").write_text(json.dumps(trace.to_json()))
    return run


def scored(run: Run, correct: bool = True) -> Run:
    """
    :return: The run, holding a score that gives its one judged part the right class or
        the wrong one.
    """
    score = SceneScore(
        run=run.directory.name,
        scene_id="kitchen",
        correspondence=FaceCorrespondence(minimum_overlap=0.5),
        classes=ClassAgreement(
            judged=[
                JudgedClass(
                    part_id=1,
                    part_name="cabinet",
                    true_class="Cabinet",
                    predicted_class="Cabinet" if correct else "Tray",
                    projected_class="Cabinet" if correct else "Tray",
                    shared_ancestry=1 if correct else 0,
                    predicted_ancestry=1,
                    true_ancestry=1,
                )
            ],
            classified_parts=1,
        ),
        links=LinkAgreement(),
    )
    (run.directory / ScoreFile.SCENE).write_text(json.dumps(score.to_json()))
    return run


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.015}


@pytest.fixture
def runs_directory(tmp_path) -> pathlib.Path:
    """
    :return: Where the runs of a test are written.
    """
    return tmp_path / "runs"


# %% what one run comes to


def test_a_run_reports_what_it_built(runs_directory):
    measured = RunMeasurements.of(written_run(runs_directory, "one"))

    assert (
        measured.labels,
        measured.labels_mapped,
        measured.new_classes,
        measured.bodies,
        measured.bodies_annotated,
        measured.distinct_classes,
        measured.parts_mounted,
        measured.mounts_refused,
    ) == (2, 2, 1, 2, 1, 2, 1, 1)


def test_a_run_reports_what_it_asked_and_what_that_cost(runs_directory):
    used = ModelUse.of(written_run(runs_directory, "one", usage=USAGE))

    assert used == ModelUse(
        questions=1,
        calls=2,
        prompt_tokens=200,
        completion_tokens=40,
        cost_cents=3.0,
        seconds=3.0,
    )


def test_a_service_that_names_no_price_costs_nothing_known(runs_directory):
    usage = {"prompt_tokens": 100, "completion_tokens": 20}

    used = ModelUse.of(written_run(runs_directory, "one", usage=usage))

    assert (used.prompt_tokens, used.cost_cents) == (200, None)


def test_a_run_reports_how_long_its_steps_took(runs_directory):
    measured = RunMeasurements.of(written_run(runs_directory, "one", seconds=10.0))

    assert measured.total_seconds == 12.0


def test_a_run_of_a_scene_with_no_ground_truth_has_no_score(runs_directory):
    measured = RunMeasurements.of(written_run(runs_directory, "one"))

    assert measured.score is None


def test_a_run_that_was_scored_reports_its_score(runs_directory):
    measured = RunMeasurements.of(scored(written_run(runs_directory, "one")))

    assert measured.score.classes.accuracy.value == 1.0


def test_a_run_that_did_not_finish_is_refused(runs_directory):
    """
    Averaging in the zeros of a run that stopped halfway would be read as a bad result.
    """
    run = written_run(runs_directory, "one")
    run.path(RunFile.EVALUATION_GRAPH).unlink()

    with pytest.raises(RunNotFinishedError):
        RunMeasurements.of(run)


# %% which runs were made under the same condition


def comparison_of(*runs: Run, **kept) -> RunComparison:
    """
    :return: The comparison of those runs.
    """
    return RunComparison([RunMeasurements.of(one) for one in runs], **kept)


def names_and_runs(groups: list[RunGroup]) -> dict[str, list[str]]:
    """
    :return: Per group, the names of its runs.
    """
    return {group.name: [one.run for one in group.runs] for group in groups}


def test_runs_differing_in_a_setting_are_under_different_conditions(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", model="a/model"),
        written_run(runs_directory, "two", model="b/model"),
        written_run(runs_directory, "three", model="a/model"),
    )

    assert names_and_runs(comparison.by_condition) == {
        "a/model": ["one", "three"],
        "b/model": ["two"],
    }


def test_runs_of_other_scenes_are_still_under_the_same_condition(runs_directory):
    """
    Neither the scene nor where a run is kept is what a condition compares.
    """
    comparison = comparison_of(
        written_run(runs_directory, "one", scene="kitchen"),
        written_run(runs_directory, "two", scene="office"),
    )

    assert [len(group.runs) for group in comparison.by_condition] == [2]


def test_runs_of_different_code_are_under_different_conditions(runs_directory):
    """
    The same settings on another commit may have been asked with other prompts, and a
    version is numbered from the earliest run on.
    """
    comparison = comparison_of(
        written_run(runs_directory, "one", commit="abcdef0123456789"),
        written_run(runs_directory, "two", commit="123456abcdef0000"),
    )

    assert names_and_runs(comparison.by_condition) == {"v1": ["one"], "v2": ["two"]}


def test_a_known_model_is_called_by_its_short_name(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", model=Model.QWEN3_8_27B_DENVER.value),
        written_run(runs_directory, "two", model=Model.QWEN3_VL_30B.value),
    )

    assert sorted(group.name for group in comparison.by_condition) == sorted(
        [Model.QWEN3_8_27B_DENVER.short_name, Model.QWEN3_VL_30B.short_name]
    )


def test_a_run_without_pictures_is_called_text_and_with_them_images(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", pictures=True),
        written_run(runs_directory, "two", pictures=False),
    )

    assert names_and_runs(comparison.by_condition) == {
        "images": ["one"],
        "text": ["two"],
    }


def test_contested_faces_shown_like_the_pictures_are_not_said_twice(runs_directory):
    comparison = comparison_of(
        written_run(
            runs_directory, "one", pictures=True, contested_faces_pictured=True
        ),
        written_run(
            runs_directory, "two", pictures=False, contested_faces_pictured=False
        ),
    )

    assert sorted(group.name for group in comparison.by_condition) == [
        "images",
        "text",
    ]


def test_a_method_is_defined_by_every_setting_it_is_told_apart_by(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", model="a/model", pictures=True),
        written_run(runs_directory, "two", model="b/model", pictures=False),
    )

    assert comparison.definition_of(comparison.runs[0]) == (
        "model a/model; show the pictures True"
    )


def test_settings_every_method_shares_are_recorded_once(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", model="a/model"),
        written_run(runs_directory, "two", model="b/model"),
    )

    assert comparison.shared_settings == {"show_the_pictures": True}


def test_a_condition_can_be_given_a_name_for_the_reader(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", model="a/model"),
        written_run(runs_directory, "two", model="b/model"),
        renamed_conditions={"a/model": "Local"},
    )

    assert sorted(group.name for group in comparison.by_condition) == [
        "Local",
        "b/model",
    ]


def test_each_run_is_named_with_its_condition(runs_directory):
    comparison = comparison_of(
        written_run(runs_directory, "one", model="a/model"),
        written_run(runs_directory, "two", model="b/model"),
    )

    assert [group.name for group in comparison.by_run] == [
        "a/model / one",
        "b/model / two",
    ]


# %% a row over a group of runs


def group_of(*runs: Run) -> RunGroup:
    """
    :return: Those runs, as one group.
    """
    return RunGroup("compared", [RunMeasurements.of(one) for one in runs])


def test_one_run_reports_its_value_without_a_spread_it_cannot_have(runs_directory):
    row = EffortResult.of(group_of(written_run(runs_directory, "one", seconds=10.0)))

    assert row.total_minutes == pytest.approx(0.2)


def test_several_runs_report_mean_and_standard_deviation(runs_directory):
    row = EffortResult.of(
        group_of(
            written_run(runs_directory, "one", seconds=58.0),
            written_run(runs_directory, "two", seconds=118.0),
        )
    )

    assert row.total_minutes == MeanAndStandardDeviation.from_measurements([1.0, 2.0])


def test_a_group_counts_its_runs_and_names_its_only_scene(runs_directory):
    row = StructureResult.of(
        group_of(written_run(runs_directory, "one"), written_run(runs_directory, "two"))
    )

    assert (row.method, row.runs, row.scenes) == ("compared", 2, "kitchen")


def test_a_group_of_several_scenes_counts_them(runs_directory):
    row = StructureResult.of(
        group_of(
            written_run(runs_directory, "one", scene="kitchen"),
            written_run(runs_directory, "two", scene="office"),
        )
    )

    assert row.scenes == 2


def test_a_run_with_no_score_is_left_out_of_a_mean_of_scores(runs_directory):
    row = ClassResult.of(
        group_of(
            scored(written_run(runs_directory, "one"), correct=True),
            written_run(runs_directory, "two"),
        )
    )

    assert row.class_accuracy == 1.0


def test_scores_nobody_has_are_absent(runs_directory):
    row = LinkResult.of(group_of(written_run(runs_directory, "one")))

    assert (row.link_f1, row.triplet_f1) == (None, None)


def test_what_a_run_built_is_reported_as_shares_of_the_scene(runs_directory):
    """
    The run written here reads both labels as a class, annotates one body of two, and
    has one of its two mounts refused.
    """
    row = StructureResult.of(group_of(written_run(runs_directory, "one")))

    assert (
        row.labels_mapped,
        row.new_classes,
        row.bodies_annotated,
        row.bodies_mounted,
        row.mounts_accepted,
    ) == (1.0, 1, 0.5, 0.5, 0.5)


def test_what_a_run_asked_is_reported_compactly(runs_directory):
    row = EffortResult.of(group_of(written_run(runs_directory, "one", usage=USAGE)))

    assert (row.questions, row.corrected_answers, row.thousand_tokens) == (
        1,
        1,
        pytest.approx(0.24),
    )


# %% the tables


def tables_in(directory: pathlib.Path) -> set[str]:
    """
    :return: The names of the LaTeX tables written there.
    """
    return {one.stem for one in directory.glob("*.tex")}


def test_runs_with_no_ground_truth_are_given_every_table_but_the_scores(
    runs_directory, tmp_path
):
    output = tmp_path / "tables"

    ResultTables(comparison_of(written_run(runs_directory, "one"))).write(output)

    assert tables_in(output) == {
        "all_tables",
        "methods",
        "conditions_structure",
        "conditions_effort",
        "runs_structure",
        "runs_effort",
    }


def test_scored_runs_are_also_given_the_tables_of_their_scores(
    runs_directory, tmp_path
):
    output = tmp_path / "tables"

    ResultTables(comparison_of(scored(written_run(runs_directory, "one")))).write(
        output
    )

    assert {
        "conditions_classes",
        "conditions_links",
        "runs_classes",
        "runs_links",
    } <= tables_in(output)


def conditions_table(runs_directory, tmp_path, *runs) -> str:
    """
    :return: The markup of the effort table per condition of those runs.
    """
    output = tmp_path / "tables"
    ResultTables(comparison_of(*runs)).write(output)
    return (output / "conditions_effort.tex").read_text()


def test_what_holds_for_every_method_moves_from_the_columns_to_the_caption(
    runs_directory, tmp_path
):
    table = conditions_table(
        runs_directory,
        tmp_path,
        written_run(runs_directory, "one", model="a/model", seconds=58.0),
        written_run(runs_directory, "two", model="b/model", seconds=118.0),
    )
    caption, columns = table.split(r"\begin{tabular}")

    assert "Runs 1" in caption and "Scenes kitchen" in caption
    assert r"\textbf{Runs}" not in columns and r"\textbf{Scenes}" not in columns
    assert r"\textbf{Total Minutes}" in columns


def test_methods_with_unequal_runs_keep_the_column_saying_so(runs_directory, tmp_path):
    table = conditions_table(
        runs_directory,
        tmp_path,
        written_run(runs_directory, "one", model="a/model"),
        written_run(runs_directory, "two", model="a/model"),
        written_run(runs_directory, "three", model="b/model"),
    )

    assert r"\textbf{Runs}" in table.split(r"\begin{tabular}")[1]


def test_one_method_still_reports_its_measurements_in_columns(runs_directory, tmp_path):
    table = conditions_table(
        runs_directory, tmp_path, written_run(runs_directory, "one")
    )
    columns = table.split(r"\begin{tabular}")[1]

    assert r"\textbf{Total Minutes}" in columns
    assert r"\textbf{Runs}" not in columns


def test_a_measurement_no_method_has_is_not_mentioned(runs_directory, tmp_path):
    table = conditions_table(
        runs_directory,
        tmp_path,
        written_run(runs_directory, "one", model="a/model"),
        written_run(runs_directory, "two", model="b/model"),
    )

    assert "Cost" not in table


def test_the_methods_are_defined_in_a_table_of_their_own(runs_directory, tmp_path):
    output = tmp_path / "tables"

    ResultTables(
        comparison_of(
            written_run(runs_directory, "one", model="a/model"),
            written_run(runs_directory, "two", model="b/model"),
        )
    ).write(output)

    table = (output / "methods.tex").read_text()
    assert "model a/model" in table
    assert (output / "methods.json").is_file()


def test_every_table_is_recorded_beside_it(runs_directory, tmp_path):
    output = tmp_path / "tables"

    ResultTables(comparison_of(written_run(runs_directory, "one"))).write(output)

    [recorded] = json.loads((output / "conditions_structure.json").read_text())
    assert (recorded["bodies_annotated"], recorded["runs"]) == (0.5, 1)


def test_the_command_writes_the_tables_of_the_runs_it_is_given(
    runs_directory, tmp_path
):
    run = written_run(runs_directory, "one", model="a/model")
    other = written_run(runs_directory, "two", model="b/model")
    output = tmp_path / "tables"

    main(
        [
            str(run.directory),
            str(other.directory),
            "--output",
            str(output),
            "--rename",
            "a/model=Local",
        ]
    )

    assert "Local" in (output / "conditions_effort.tex").read_text()


@pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="no LaTeX to compile the tables with"
)
def test_all_the_tables_compile_together(runs_directory, tmp_path):
    output = tmp_path / "tables"
    ResultTables(
        comparison_of(
            scored(written_run(runs_directory, "one_1", model="a_model")),
            scored(written_run(runs_directory, "two_2", model="a_model")),
            written_run(runs_directory, "three_3", model="b_model", usage=USAGE),
        )
    ).write(output)
    document = pathlib.Path(__file__).parents[1] / "dataset" / "experiment_tables"
    shutil.copy(document / "all_tables_document.tex", output / "document.tex")

    compiled = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "document.tex"],
        cwd=output,
        capture_output=True,
        text=True,
    )

    assert compiled.returncode == 0, compiled.stdout
