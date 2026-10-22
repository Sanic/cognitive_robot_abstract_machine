# %% imports

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.experiments.camera_handoff.feature_tracking import FeatureTargetTracker
from robokudo.experiments.camera_handoff.perception import (
    CameraObservation,
    ImageRegion,
    TargetEstimate,
)
from robokudo.experiments.camera_handoff.recording import ApproachFiles, ImageRecorder
from robokudo.experiments.camera_handoff.textured_experiment import (
    TexturedHandoffExperiment,
)
from robokudo.experiments.camera_handoff.textured_models import (
    ModelAsset,
    TexturedModel,
)
from robokudo.types.annotation import BoundingBox3DAnnotation, Plane, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import get_transform_matrix_from_q

if TYPE_CHECKING:
    from pathlib import Path

    from robokudo.experiments.camera_handoff.simulation import CameraPair


# %% detected tabletop fixtures


@dataclass
class DetectedTabletop:
    """
    A rendered head observation processed by the standard tabletop annotators.
    """

    experiment: TexturedHandoffExperiment
    """Experiment whose head pipeline produced the observation."""
    cameras: CameraPair
    """
    Camera pipelines observing the tabletop scene.
    """

    observation: CameraObservation
    """Calibrated head image associated with the pipeline CAS."""


@pytest.fixture
def detected_tabletop() -> DetectedTabletop:
    """
    Capture an oblique RGB-D head view containing a target and distractor.
    """
    experiment = TexturedHandoffExperiment(head_resolution=320, hand_resolution=80)
    cameras = experiment.create_cameras()
    return DetectedTabletop(experiment, cameras, cameras.head.capture())


# %% pipeline integration


def test_head_uses_demo_tabletop_annotators(
    detected_tabletop: DetectedTabletop,
) -> None:
    """
    Head acquisition executes segmentation and bounding-box pose estimation.
    """
    pipeline = detected_tabletop.cameras.head.pipeline
    assert tuple(type(annotator) for annotator in pipeline.get_annotators()) == (
        CollectionReaderAnnotator,
        ImagePreprocessorAnnotator,
        PointcloudCropAnnotator,
        PlaneAnnotator,
        PointCloudClusterExtractor,
        ClusterPoseBBAnnotator,
    )
    assert len(pipeline.cas.filter_annotations_by_type(Plane)) == 1
    assert len(pipeline.cas.filter_annotations_by_type(ObjectHypothesis)) == 2


def test_initial_estimate_uses_cluster_pose_and_extents(
    detected_tabletop: DetectedTabletop,
) -> None:
    """
    The handoff preserves the measured annotation's pose and box dimensions.
    """
    cas = detected_tabletop.cameras.head.pipeline.cas
    selected = detected_tabletop.experiment.head_initializer.estimate(
        cas, detected_tabletop.observation
    )
    pose = cas.filter_by_type(PoseAnnotation, selected.hypothesis.annotations)[0]
    bounding_box = cas.filter_by_type(
        BoundingBox3DAnnotation, selected.hypothesis.annotations
    )[0]
    expected_pose = (
        detected_tabletop.observation.world_T_camera
        @ get_transform_matrix_from_q(pose.rotation, pose.translation)
    )
    np.testing.assert_allclose(selected.estimate.world_center, expected_pose[:3, 3])
    np.testing.assert_allclose(selected.estimate.world_R_target, expected_pose[:3, :3])
    np.testing.assert_array_equal(
        selected.estimate.dimensions,
        [bounding_box.x_length, bounding_box.y_length, bounding_box.z_length],
    )


def test_segmented_handoff_needs_no_model_dimensions_or_depth(
    detected_tabletop: DetectedTabletop,
) -> None:
    """
    Once a cluster pose exists, tracker initialization consumes its measured box.
    """
    experiment = detected_tabletop.experiment
    selected = experiment.head_initializer.estimate(
        detected_tabletop.cameras.head.pipeline.cas, detected_tabletop.observation
    )
    estimate = FeatureTargetTracker().acquire(
        detected_tabletop.observation,
        np.zeros(detected_tabletop.observation.color.shape[:2]),
        np.full(3, np.nan),
        experiment.position_bias,
        segmented_target=selected,
    )
    np.testing.assert_allclose(
        estimate.world_center, selected.estimate.world_center + experiment.position_bias
    )
    np.testing.assert_array_equal(estimate.dimensions, selected.estimate.dimensions)


def test_head_recording_preserves_sensor_image(
    detected_tabletop: DetectedTabletop, tmp_path: Path
) -> None:
    """
    Head segmentation diagnostics leave the captured RGB image unchanged.
    """
    cas = detected_tabletop.cameras.head.pipeline.cas
    observation = detected_tabletop.observation
    selected = detected_tabletop.experiment.head_initializer.estimate(cas, observation)
    original = observation.color.copy()
    ImageRecorder(tmp_path).write_initial_estimate(cas, observation, selected)
    np.testing.assert_array_equal(observation.color, original)
    pose = cas.filter_by_type(PoseAnnotation, selected.hypothesis.annotations)[0]
    assert pose.source in (tmp_path / ApproachFiles.HEAD_ESTIMATE).read_text()
    assert (tmp_path / ApproachFiles.HEAD_SEGMENTATION).is_file()


@pytest.mark.parametrize("asset", list(ModelAsset))
def test_segmented_seed_selects_target_not_distractor(asset: ModelAsset) -> None:
    """
    A head-image seed selects the intended cluster independently of cluster size.
    """
    experiment = TexturedHandoffExperiment(
        model=TexturedModel(asset), head_resolution=320
    )
    cameras = experiment.create_cameras()
    observation = cameras.head.capture()
    selected = experiment.head_initializer.estimate(
        cameras.head.pipeline.cas, observation
    )
    assert selected.estimate.world_center[0] == pytest.approx(0, abs=0.02)


# %% oriented projection


def test_projected_region_preserves_box_orientation() -> None:
    """
    A quarter turn swaps the projected box's width and height.
    """
    observation = CameraObservation(
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.array([[100.0, 0, 49.5], [0, 100.0, 49.5], [0, 0, 1]]),
        np.eye(4),
    )
    target = TargetEstimate(
        np.array([0.0, 0.0, 1.0]),
        0.2,
        np.array([0.2, 0.1, 0.1]),
        world_R_target=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert target.project_region(observation, margin_pixels=0) == ImageRegion(
        44, 38, 56, 62
    )


def test_front_depth_preserves_box_orientation() -> None:
    """
    Optical depth uses the box axis that actually faces the camera.
    """
    observation = CameraObservation(
        np.zeros((100, 100, 3), dtype=np.uint8), np.eye(3), np.eye(4)
    )
    dimensions = np.array([0.2, 0.1, 0.1])
    target = TargetEstimate(
        np.array([0.0, 0.0, 1.0]),
        0.2,
        dimensions,
        world_R_target=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    assert target.front_depth(observation) == 1 - dimensions[0] / 2
