# %% imports

from __future__ import annotations

import cv2
import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.experiments.camera_handoff.experiment import (
    ApproachEvaluator,
    CameraHandoffExperiment,
    LateralFeedback,
)
from robokudo.experiments.camera_handoff.exceptions import TargetNotAcquired
from robokudo.experiments.camera_handoff.perception import (
    CameraObservation,
    ColorTargetTracker,
    ImageRegion,
    TargetEstimate,
)
from robokudo.experiments.camera_handoff.recording import ApproachFiles, ImageRecorder

if TYPE_CHECKING:
    from pathlib import Path

    from rclpy.node import Node

# %% projection and association


@pytest.fixture
def observation() -> CameraObservation:
    """
    A calibrated optical camera facing a centered target plane.
    """
    return CameraObservation(
        color=np.zeros((100, 100, 3), dtype=np.uint8),
        intrinsic_matrix=np.array([[100.0, 0, 49.5], [0, 100.0, 49.5], [0, 0, 1]]),
        world_T_camera=np.eye(4),
    )


def test_projection_accounts_for_camera_translation(
    observation: CameraObservation,
) -> None:
    """
    A translated camera projects the stationary target into its own frame.
    """
    observation.world_T_camera[0, 3] = 0.1
    target = TargetEstimate(np.array([0.1, 0.0, 1.0]), 0.1)
    np.testing.assert_allclose(
        observation.project(target.world_center[None]), [[49.5, 49.5]]
    )


def test_projected_region_encloses_model_corners(
    observation: CameraObservation,
) -> None:
    """
    The search region includes the cube extent and configured margin.
    """
    target = TargetEstimate(np.array([0.0, 0.0, 1.0]), 0.1)
    assert target.project_region(observation, margin_pixels=5) == ImageRegion(
        39, 39, 61, 61
    )


def test_target_behind_camera_has_no_search_region(
    observation: CameraObservation,
) -> None:
    """
    A target behind the camera cannot initialize an image tracker.
    """
    target = TargetEstimate(np.array([0.0, 0.0, -1.0]), 0.1)
    assert target.project_region(observation, margin_pixels=5) is None


def test_color_tracking_ignores_objects_outside_search_region(
    observation: CameraObservation,
) -> None:
    """
    An unrelated patch outside the geometric gate cannot steal the target.
    """
    tracker = ColorTargetTracker()
    observation.color[40:60, 40:60] = tracker.target_color
    observation.color[5:20, 5:20] = tracker.target_color
    measurement = tracker.track(observation.color, ImageRegion(30, 30, 70, 70))
    assert measurement is not None
    np.testing.assert_allclose(measurement.center, [49.5, 49.5])
    assert measurement.region == ImageRegion(40, 40, 60, 60)


def test_ambiguous_color_tracking_returns_no_measurement(
    observation: CameraObservation,
) -> None:
    """
    Two matching candidates inside the gate do not produce a guessed identity.
    """
    tracker = ColorTargetTracker()
    observation.color[30:40, 30:40] = tracker.target_color
    observation.color[60:70, 60:70] = tracker.target_color
    assert tracker.track(observation.color, ImageRegion(20, 20, 80, 80)) is None


def test_partial_target_returns_no_measurement(observation: CameraObservation) -> None:
    """
    A clipped patch cannot supply a misleading centroid for servoing.
    """
    tracker = ColorTargetTracker()
    observation.color[30:60, 30:60] = tracker.target_color
    assert tracker.track(observation.color, ImageRegion(40, 20, 80, 80)) is None


# %% rendering, feedback and recording


def test_hand_pipeline_exposes_no_depth_or_scene_truth(node: Node) -> None:
    """
    The moving camera supplies only monocular observations to perception.
    """
    experiment = CameraHandoffExperiment(head_resolution=80, hand_resolution=40)
    cameras = experiment.create_cameras()
    cameras.hand.capture()
    forbidden_views = (
        CASViews.DEPTH_IMAGE,
        CASViews.OBJECT_IMAGE,
        CASViews.OBJECT_COLOR_MAP,
        CASViews.GROUND_TRUTH_WORLD_REFERENCE,
    )
    assert all(not cameras.hand.pipeline.cas.contains(view) for view in forbidden_views)
    assert cameras.head.camera.world is cameras.hand.camera.world


def test_hand_feedback_reduces_lateral_error_and_records_images(
    node: Node, tmp_path: Path
) -> None:
    """
    Image feedback improves alignment relative to the biased head-only approach.
    """
    experiment = CameraHandoffExperiment(head_resolution=120, hand_resolution=80)
    result = experiment.run(tmp_path)
    assert result.feedback.completed
    assert result.head_only.completed
    assert (
        result.feedback.final_lateral_error < result.head_only.final_lateral_error / 5
    )
    assert result.feedback.final_lateral_error < experiment.alignment_tolerance
    assert (
        len(list((tmp_path / ApproachFiles.FEEDBACK).glob("*.png")))
        == len(result.feedback.samples) * 2
    )
    assert (tmp_path / ApproachFiles.HEAD_IMAGE).is_file()
    assert (tmp_path / ApproachFiles.MEASUREMENTS).is_file()


def test_image_recorder_preserves_raw_pixels(
    tmp_path: Path, observation: CameraObservation
) -> None:
    """
    Saving an overlay does not change the recorded sensor image.
    """
    recorder = ImageRecorder(tmp_path)
    region = ImageRegion(20, 20, 80, 80)
    recorder.write_frame(ApproachFiles.FEEDBACK, 0, observation, region, None)
    image_path = next((tmp_path / ApproachFiles.FEEDBACK).glob("*_raw.png"))
    np.testing.assert_array_equal(cv2.imread(str(image_path)), observation.color)


def test_default_hand_resolution_keeps_target_inside_search_region(
    node: Node, tmp_path: Path
) -> None:
    """
    The uncertainty gate covers the biased target throughout a default approach.
    """
    result = CameraHandoffExperiment(head_resolution=120).run(tmp_path)
    assert result.head_only.completed
    assert result.feedback.completed


def test_missing_target_stops_without_advancing(node: Node, tmp_path: Path) -> None:
    """
    Lost visual association holds the camera at the last captured position.
    """
    experiment = CameraHandoffExperiment(head_resolution=80, hand_resolution=40)
    cameras = experiment.create_cameras()
    experiment.tracker.target_color = np.array([0, 255, 0], dtype=np.uint8)
    target = TargetEstimate(
        np.array([0.0, 0.0, experiment.scene.cube_edge / 2]), experiment.scene.cube_edge
    )
    result = experiment.approach(
        cameras.hand,
        target,
        ImageRecorder(tmp_path),
        ApproachFiles.FEEDBACK,
        ApproachEvaluator(target.world_center),
    )
    assert not result.completed
    assert len(result.samples) == 1
    assert cameras.hand.camera.camera_config.camera_y == -experiment.starting_distance


def test_head_acquisition_rejects_missing_depth(observation: CameraObservation) -> None:
    """
    A color match without depth cannot initialize a metric target estimate.
    """
    tracker = ColorTargetTracker()
    observation.color[40:60, 40:60] = tracker.target_color
    with pytest.raises(TargetNotAcquired):
        tracker.acquire(
            observation, np.zeros(observation.color.shape[:2]), edge_length=0.1
        )


def test_feedback_respects_lateral_step_limit(observation: CameraObservation) -> None:
    """
    A large image error produces a bounded lateral movement without approach motion.
    """
    tracker = ColorTargetTracker()
    observation.color[40:60, 70:90] = tracker.target_color
    measurement = tracker.track(observation.color, ImageRegion(10, 10, 95, 95))
    assert measurement is not None
    controller = LateralFeedback()
    correction = controller.compute_translation(observation, measurement, depth=1.0)
    np.testing.assert_allclose(correction, [controller.maximum_step, 0, 0])


def test_head_acquisition_uses_measured_surface_and_known_cube_size(
    observation: CameraObservation,
) -> None:
    """
    A face-on RGB-D observation locates the center behind the visible surface.
    """
    tracker = ColorTargetTracker()
    observation.color[40:60, 40:60] = tracker.target_color
    edge_length = 0.1
    target = tracker.acquire(
        observation, np.ones(observation.color.shape[:2]), edge_length
    )
    np.testing.assert_allclose(target.world_center, [0, 0, 1 + edge_length / 2])
