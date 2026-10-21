# %% imports

from __future__ import annotations

from unittest.mock import Mock

import cv2

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from robokudo.experiments.camera_handoff.textured_models import (
    ModelAsset,
    TexturedModel,
)
from robokudo.experiments.camera_handoff.textured_experiment import (
    TexturedHandoffExperiment,
)
from robokudo.cas import CASViews
from robokudo.experiments.camera_handoff.feature_tracking import FeatureTargetTracker
from robokudo.experiments.camera_handoff.perception import CameraObservation
from robokudo.experiments.camera_handoff.perception import ImageRegion

if TYPE_CHECKING:
    from pathlib import Path


# %% textured assets


@pytest.mark.parametrize("asset", list(ModelAsset))
def test_model_loading_preserves_texture_and_metric_scale(asset: ModelAsset) -> None:
    """
    The repository models retain UV maps and use the episode's metric scale.
    """
    model = TexturedModel(asset)
    shape = model.load()
    assert shape.mesh.visual.kind == "texture"
    assert shape.mesh.visual.uv.shape == (len(shape.mesh.vertices), 2)
    np.testing.assert_allclose(
        shape.mesh.extents, shape.unscaled_mesh.extents * model.meters_per_unit
    )


def test_textured_box_feedback_completes_and_saves_appearance(tmp_path: Path) -> None:
    """
    A textured box supplies feature correspondences throughout the approach.
    """
    experiment = TexturedHandoffExperiment(
        head_resolution=320, hand_resolution=160, forward_step=0.03
    )
    result = experiment.run(tmp_path)
    assert result.feedback.completed
    assert (
        result.feedback.final_lateral_error < result.head_only.final_lateral_error / 3
    )
    assert (tmp_path / experiment.model.asset.preview_filename).is_file()


def test_table_mask_does_not_raise_estimated_box_center() -> None:
    """
    Excluding tabletop pixels must not shift the known box's vertical center.
    """
    experiment = TexturedHandoffExperiment(head_resolution=320, hand_resolution=80)
    cameras = experiment.create_cameras()
    observation = cameras.head.capture()
    bounds = experiment.model.measure_bounds(experiment.model.load())
    dimensions = bounds[1] - bounds[0]
    target = experiment.feature_tracker.acquire(
        observation,
        cameras.head.pipeline.cas.get(CASViews.DEPTH_IMAGE).astype(float) / 1000,
        dimensions,
        np.zeros(3),
    )
    assert target.world_center[2] == pytest.approx(dimensions[2] / 2, abs=0.002)


def test_head_seed_selects_cup_instead_of_larger_distractor() -> None:
    """
    The head image's target seed preserves identity despite a larger distractor.
    """
    experiment = TexturedHandoffExperiment(model=TexturedModel(ModelAsset.CUP))
    cameras = experiment.create_cameras()
    observation = cameras.head.capture()
    bounds = experiment.model.measure_bounds(experiment.model.load())
    target = experiment.feature_tracker.acquire(
        observation,
        cameras.head.pipeline.cas.get(CASViews.DEPTH_IMAGE).astype(float) / 1000,
        bounds[1] - bounds[0],
        np.zeros(3),
    )
    assert target.world_center[0] == pytest.approx(0, abs=0.01)


def test_head_initialization_needs_no_texture_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A textureless head observation can initialize independent hand tracking.
    """
    monkeypatch.setattr(
        cv2,
        "SIFT_create",
        Mock(
            side_effect=AssertionError(
                "Head initialization must not extract SIFT features"
            )
        ),
    )
    monkeypatch.setattr(
        cv2,
        "goodFeaturesToTrack",
        Mock(
            side_effect=AssertionError(
                "Head initialization must not extract tracking features"
            )
        ),
    )
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[30:70, 30:70] = [0, 0, 200]
    depth = np.zeros(color.shape[:2])
    depth[30:70, 30:70] = 1
    observation = CameraObservation(
        color, np.array([[100.0, 0, 49.5], [0, 100.0, 49.5], [0, 0, 1]]), np.eye(4)
    )
    dimensions = np.full(3, 0.1)
    target = FeatureTargetTracker().acquire(observation, depth, dimensions, np.zeros(3))
    assert target.world_center[2] == pytest.approx(1 - dimensions[2] / 2)


def test_temporal_tracking_uses_only_hand_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Hand-image motion is measured without any cross-camera feature matcher.
    """
    monkeypatch.setattr(
        cv2,
        "SIFT_create",
        Mock(
            side_effect=AssertionError("No cross-camera feature extraction is needed")
        ),
    )
    monkeypatch.setattr(
        cv2,
        "BFMatcher",
        Mock(side_effect=AssertionError("No cross-camera feature matching is needed")),
    )
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[30:70, 30:70] = [0, 0, 200]
    for row in range(34, 66, 6):
        for column in range(34, 66, 6):
            color[row : row + 2, column : column + 2] = 255
    tracker = FeatureTargetTracker(target_hue=0)
    region = ImageRegion(10, 10, 90, 90)
    tracker.track(color, region)
    translation = np.array([5.0, 3.0])
    shifted = cv2.warpAffine(
        color,
        np.column_stack((np.eye(2), translation)),
        (color.shape[1], color.shape[0]),
    )
    measurement = tracker.track(shifted, region)
    assert measurement is not None
    np.testing.assert_allclose(
        measurement.center, np.array([49.5, 49.5]) + translation, atol=0.1
    )
