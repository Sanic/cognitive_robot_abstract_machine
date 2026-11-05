"""
The splits, and the lock that keeps the validation scenes closed until the code is
frozen.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from experiments.warsaw.articulate3d.convert import (
    ConvertedScene,
    LabelDisclosure,
    LabelOverlap,
    SceneConversion,
)
from experiments.warsaw.articulate3d.exceptions import ValidationSceneLockedError
from experiments.warsaw.articulate3d.score import SceneScore
from experiments.warsaw.articulate3d.splits import SceneSplit, ValidationLock
from experiments.warsaw.pipeline.pipeline import WarsawPipeline
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.settings import PipelineSettings

from .scenes import kitchen_corner

# %% a checkout to freeze


def git(repository, *arguments):
    """
    Run a Git command in *repository*, as nobody in particular.
    """
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evaluation Test",
            "-c",
            "user.email=evaluation@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path):
    """
    :return: A Git work tree with one commit, no tag and no local changes.
    """
    repository = tmp_path / "checkout"
    repository.mkdir()
    git(repository, "init", "--quiet")
    (repository / "pipeline.py").write_text("frozen = False\n")
    git(repository, "add", "pipeline.py")
    git(repository, "commit", "--quiet", "-m", "the pipeline")
    return repository


@pytest.fixture
def validation_scene():
    """
    :return: A scene of the validation split.
    """
    return sorted(SceneSplit.VALIDATION.scene_ids)[0]


# %% the splits


def test_no_scene_is_in_both_splits():
    assert SceneSplit.TRAIN.scene_ids.isdisjoint(SceneSplit.VALIDATION.scene_ids)


# %% the lock


def test_a_train_scene_is_open_in_a_checkout_that_is_not_frozen(checkout):
    (checkout / "pipeline.py").write_text("tuning = True\n")

    ValidationLock(repository=checkout).check(sorted(SceneSplit.TRAIN.scene_ids)[0])


def test_a_validation_scene_is_locked_in_a_checkout_without_a_freezing_tag(
    checkout, validation_scene
):
    with pytest.raises(ValidationSceneLockedError):
        ValidationLock(repository=checkout).check(validation_scene)


def test_a_tag_of_another_name_does_not_freeze(checkout, validation_scene):
    git(checkout, "tag", "tuned-a-little")

    with pytest.raises(ValidationSceneLockedError):
        ValidationLock(repository=checkout).check(validation_scene)


def test_a_freezing_tag_does_not_cover_local_changes(checkout, validation_scene):
    """
    Changes made after the freeze would be evaluated under the frozen commit's name.
    """
    lock = ValidationLock(repository=checkout)
    git(checkout, "tag", f"{lock.tag_prefix}paper")
    (checkout / "pipeline.py").write_text("frozen = False, but quietly\n")

    with pytest.raises(ValidationSceneLockedError):
        lock.check(validation_scene)


def test_a_freezing_tag_does_not_cover_untracked_files(checkout, validation_scene):
    lock = ValidationLock(repository=checkout)
    git(checkout, "tag", f"{lock.tag_prefix}paper")
    (checkout / "a_new_prompt.md").write_text("one more hint\n")

    with pytest.raises(ValidationSceneLockedError):
        lock.check(validation_scene)


def test_a_validation_scene_opens_at_a_freezing_tag_with_no_local_changes(
    checkout, validation_scene
):
    lock = ValidationLock(repository=checkout)
    git(checkout, "tag", f"{lock.tag_prefix}paper")

    lock.check(validation_scene)


def test_a_validation_scene_is_locked_where_there_is_no_checkout(
    tmp_path, validation_scene
):
    with pytest.raises(ValidationSceneLockedError):
        ValidationLock(repository=tmp_path / "nowhere").check(validation_scene)


# %% every way into a validation scene


def written_as(directory, scene_id):
    """
    :return: *directory*, holding the record of a scene converted from *scene_id*.
    """
    directory.mkdir()
    ConvertedScene(
        scene_id=scene_id,
        disclosure=LabelDisclosure.WORDS,
        overlap=LabelOverlap.DISJOINT,
        seed=0,
    ).write_beside(directory)
    return directory


def test_a_validation_scene_is_not_converted(checkout, validation_scene, tmp_path):
    scene = kitchen_corner()
    scene = replace(
        scene,
        scannetpp_scene=replace(scene.scannetpp_scene, scene_id=validation_scene),
    )

    with pytest.raises(ValidationSceneLockedError):
        SceneConversion(lock=ValidationLock(repository=checkout)).write(
            scene, tmp_path / "converted"
        )

    assert not (tmp_path / "converted").exists()


def test_a_validation_scene_is_not_run(checkout, validation_scene, tmp_path):
    runs = tmp_path / "runs"
    settings = PipelineSettings(
        scene_directory=written_as(tmp_path / "scene", validation_scene),
        repository=checkout,
        runs_directory=runs,
        headless=True,
    )

    with pytest.raises(ValidationSceneLockedError):
        WarsawPipeline(settings=settings).carry_out()

    assert not runs.exists()


def test_a_run_of_a_validation_scene_is_not_scored(
    checkout, validation_scene, tmp_path
):
    run = Run(directory=tmp_path / "run")
    run.directory.mkdir()
    scene = written_as(tmp_path / "scene", validation_scene)
    run.write_json(RunFile.PROVENANCE, {"settings": {"scene_directory": str(scene)}})

    with pytest.raises(ValidationSceneLockedError):
        SceneScore.of(
            run, tmp_path / "ground_truth", lock=ValidationLock(repository=checkout)
        )
