"""
Comparison exports must respect the run's already-applied source transform.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from experiments.warsaw.evaluation.export_comparison import (
    ComparisonBundle,
    ComparisonExporter,
    ComparisonFile,
    ExportedWorld,
    LandmarkAlignment,
)
from experiments.warsaw.evaluation.export_world_mesh import write_world_mesh
from experiments.warsaw.pipeline.records import SplitRecord
from experiments.warsaw.pipeline.run import Run, RunFile

from .test_export_world_mesh import world_with_a_drawer_and_an_unnamed_body

# %% source and export frame composition


def test_exported_geometry_is_not_transformed_from_the_source_frame_twice(
    tmp_path: Path,
) -> None:
    """
    Remove the pipeline transform before applying the raw-scan landmark fit.
    """
    run = Run.create(tmp_path / "runs")
    world = world_with_a_drawer_and_an_unnamed_body()
    write_world_mesh(world, run.path(RunFile.WORLD_MESH))
    ground_truth = tmp_path / "ground_truth"
    write_world_mesh(world, ground_truth)
    world_T_source = np.eye(4)
    world_T_source[0, 3] = 1.0
    ground_truth_T_source = np.eye(4)
    ground_truth_T_source[0, 3] = 2.0
    run.write_record(
        RunFile.SPLIT,
        SplitRecord(scene="fixture", world_T_source=world_T_source.tolist()),
    )
    alignment = tmp_path / ComparisonFile.ALIGNMENT
    alignment.write_text(
        json.dumps(LandmarkAlignment(matrix=ground_truth_T_source.tolist()).to_json())
    )
    output = tmp_path / "comparison"

    written = ComparisonExporter(run.directory, ground_truth, alignment).write(output)
    bundle = ComparisonBundle.from_json(json.loads(written.read_text()))

    expected = ground_truth_T_source @ np.linalg.inv(world_T_source)
    np.testing.assert_allclose(bundle.ground_truth_T_prediction, expected)
    original = ExportedWorld.read(run.path(RunFile.WORLD_MESH)).measure(np.eye(4))
    np.testing.assert_allclose(
        bundle.prediction_objects[0].bounds,
        np.asarray(original[0].bounds) + expected[:3, 3],
    )
    assert (
        output / ComparisonFile.PREDICTION / ComparisonFile.MANIFEST
    ).read_bytes() == (
        run.path(RunFile.WORLD_MESH) / ComparisonFile.MANIFEST
    ).read_bytes()
    assert (
        output / ComparisonFile.GROUND_TRUTH / ComparisonFile.MANIFEST
    ).read_bytes() == (ground_truth / ComparisonFile.MANIFEST).read_bytes()
