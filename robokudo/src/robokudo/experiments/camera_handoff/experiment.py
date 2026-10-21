# %% imports

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.experiments.camera_handoff.exceptions import InvalidExperiment
from robokudo.experiments.camera_handoff.perception import (
    ColorTargetTracker,
)
from robokudo.experiments.camera_handoff.recording import ApproachFiles, ImageRecorder
from robokudo.experiments.camera_handoff.simulation import (
    CameraPair,
    CameraPipeline,
    SceneNames,
    SharedSceneCamera,
    TabletopScene,
)

if TYPE_CHECKING:
    from pathlib import Path

    from robokudo.experiments.camera_handoff.perception import (
        CameraObservation,
        TargetEstimate,
        TargetMeasurement,
        ImageTargetTracker,
    )

# %% feedback and measurements


@dataclass
class LateralFeedback:
    """Translate a camera laterally using a target bearing and depth prior."""

    gain: float = 0.6
    """Fraction of the estimated lateral error corrected per step."""
    maximum_step: float = 0.015
    """Maximum lateral movement in meters per step."""

    def compute_translation(
        self,
        observation: CameraObservation,
        measurement: TargetMeasurement,
        depth: float,
    ) -> np.ndarray:
        """Return a world-frame correction perpendicular to the optical axis."""
        pixels = np.array([measurement.center, observation.intrinsic_matrix[:2, 2]])
        world_points = observation.unproject(pixels, np.full(2, depth))
        world_V_correction = (world_points[0] - world_points[1]) * self.gain
        length = float(np.linalg.norm(world_V_correction))
        if length > self.maximum_step:
            world_V_correction *= self.maximum_step / length
        return world_V_correction


@dataclass
class ApproachSample:
    """Camera state and alignment measurements at one captured approach step."""

    approach: str
    """Name of the approach receiving this measurement."""
    step: int
    """Zero-based captured frame number."""
    camera_x: float
    """Camera position along the world x axis in meters."""
    camera_y: float
    """Camera position along the world y axis in meters."""
    camera_z: float
    """Camera position along the world z axis in meters."""
    pixel_error: float | None
    """Distance from observed centroid to desired image alignment."""
    lateral_error: float
    """Ground-truth lateral alignment error, for evaluation only."""
    target_visible: bool
    """Whether a unique complete target patch was measured."""


@dataclass
class ApproachResult:
    """Recorded outcome of one simulated approach."""

    completed: bool
    """Whether the approach reached its stopping distance with a visible target."""
    samples: list[ApproachSample]
    """Measurements recorded before each camera movement."""

    final_lateral_error: float = field(init=False)
    """Ground-truth lateral error of the last captured camera position."""

    def __post_init__(self) -> None:
        """Extract final alignment from a nonempty captured trajectory."""
        if not self.samples:
            raise InvalidExperiment(
                "An approach must contain at least one captured frame."
            )
        self.final_lateral_error = self.samples[-1].lateral_error


@dataclass
class ComparisonResult:
    """Outcomes from identical initial conditions with and without feedback."""

    head_only: ApproachResult
    """Approach that retains the biased head-camera alignment."""
    feedback: ApproachResult
    """Approach corrected by monocular hand observations."""


# %% evaluation


@dataclass
class ApproachEvaluator:
    """Measure alignment independently of the perception and feedback components."""

    truth_center: np.ndarray
    """Ground-truth cube center, accessible only for evaluation."""

    def measure(
        self,
        approach: ApproachFiles,
        step: int,
        observation: CameraObservation,
        pixel_error: float | None,
    ) -> ApproachSample:
        """Record image error and ground-truth lateral error at a captured pose."""
        camera_T_world = np.linalg.inv(observation.world_T_camera)
        camera_P_truth = camera_T_world @ np.append(self.truth_center, 1)
        return ApproachSample(
            str(approach),
            step,
            *observation.world_T_camera[:3, 3].tolist(),
            pixel_error,
            float(np.linalg.norm(camera_P_truth[:2])),
            pixel_error is not None,
        )


# %% experiment orchestration


@dataclass
class CameraHandoffExperiment:
    """Compare head-only and monocular-feedback approaches to a stationary cube."""

    head_resolution: int = 512
    """Square head-image width and height in pixels."""
    hand_resolution: int = 240
    """Square hand-image width and height in pixels."""
    head_height: float = 0.85
    """Height of the downward-looking head camera above the table."""
    starting_distance: float = 0.55
    """Initial hand-camera distance to the cube center along world y."""
    stopping_distance: float = 0.22
    """Final hand-camera distance to the estimated cube center along world y."""
    forward_step: float = 0.01
    """Forward movement in meters between captures."""
    position_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.04, 0.0, 0.025])
    )
    """Deliberate world-frame error added to the head-derived target estimate."""
    search_margin: int = 24
    """Extra pixels around projected model bounds for target acquisition."""
    search_uncertainty: float = 0.06
    """Position uncertainty in meters added around each side of the model."""
    alignment_tolerance: float = 0.005
    """Lateral error in meters used to assess the demonstration."""
    maximum_frames: int = 100
    """Bound on captures in each approach, including stationary alignment steps."""
    pixel_tolerance: float = 1.0
    """Acceptable final image alignment error in pixels."""
    scene: TabletopScene = field(default_factory=TabletopScene)
    """Stationary tabletop geometry and known target dimensions."""
    tracker: ColorTargetTracker = field(default_factory=ColorTargetTracker)
    """Target acquisition shared across head and hand observations."""
    controller: LateralFeedback = field(default_factory=LateralFeedback)
    """Image-feedback translation policy."""

    def create_cameras(self) -> CameraPair:
        """Create two independent CollectionReader pipelines sharing one scene."""
        if (
            not 0 < self.forward_step < self.stopping_distance < self.starting_distance
            or self.maximum_frames <= 0
        ):
            raise InvalidExperiment("Approach distances or capture limit are invalid.")
        world = self.scene.create_world(self.tracker.target_color)
        head_config = SemDTRayTracerCameraConfig(
            camera_frame=SceneNames.HEAD,
            resolution=self.head_resolution,
            camera_x=0,
            camera_y=0,
            camera_z=self.head_height,
            camera_roll=pi,
            camera_pitch=0,
            camera_yaw=0,
        )
        hand_config = SemDTRayTracerCameraConfig(
            camera_frame=SceneNames.HAND,
            resolution=self.hand_resolution,
            camera_x=0,
            camera_y=-self.starting_distance,
            camera_z=self.scene.cube_edge / 2,
            camera_roll=-pi / 2,
            camera_pitch=0,
            camera_yaw=0,
        )
        return CameraPair(
            CameraPipeline(SharedSceneCamera(head_config, world, include_depth=True)),
            CameraPipeline(SharedSceneCamera(hand_config, world)),
        )

    def run(self, output_directory: Path) -> ComparisonResult:
        """Execute both approaches and save their images and measured errors."""
        cameras = self.create_cameras()
        head = cameras.head.capture()
        depth = cameras.head.pipeline.cas.get(CASViews.DEPTH_IMAGE).astype(float) / 1000
        target = self.tracker.acquire(head, depth, self.scene.cube_edge)
        target.world_center += self.position_bias
        recorder = ImageRecorder(output_directory)
        world_T_target = cameras.hand.camera.world.compute_forward_kinematics_np(
            cameras.hand.camera.world.root,
            cameras.hand.camera.world.get_body_by_name(SceneNames.TARGET),
        )
        evaluator = ApproachEvaluator(world_T_target[:3, 3])
        comparison = ComparisonResult(
            self.approach(
                cameras.hand,
                target,
                recorder,
                ApproachFiles.HEAD_ONLY,
                evaluator,
            ),
            self.approach(
                cameras.hand,
                target,
                recorder,
                ApproachFiles.FEEDBACK,
                evaluator,
            ),
        )
        recorder.write_comparison(comparison, head)
        return comparison

    def approach(
        self,
        camera: CameraPipeline,
        target: TargetEstimate,
        recorder: ImageRecorder,
        approach: ApproachFiles,
        evaluator: ApproachEvaluator,
        tracker: ImageTargetTracker | None = None,
    ) -> ApproachResult:
        """Advance toward the estimated target, stopping on lost or ambiguous vision."""
        config = camera.camera.camera_config
        tracker = self.tracker if tracker is None else tracker
        config.camera_x = float(target.world_center[0])
        config.camera_y = float(target.world_center[1] - self.starting_distance)
        config.camera_z = float(target.world_center[2])
        samples: list[ApproachSample] = []
        for step in range(self.maximum_frames):
            observation = camera.capture()
            region = target.project_region(
                observation, self.search_margin, self.search_uncertainty
            )
            measurement = tracker.track(observation.color, region)
            recorder.write_frame(approach, step, observation, region, measurement)
            pixel_error = (
                None
                if measurement is None
                else float(
                    np.linalg.norm(
                        measurement.center - observation.intrinsic_matrix[:2, 2]
                    )
                )
            )
            samples.append(evaluator.measure(approach, step, observation, pixel_error))
            if measurement is None:
                return ApproachResult(False, samples)
            remaining = float(
                target.world_center[1] - config.camera_y - self.stopping_distance
            )
            if remaining <= np.finfo(float).eps and (
                approach == ApproachFiles.HEAD_ONLY
                or pixel_error <= self.pixel_tolerance
            ):
                return ApproachResult(True, samples)
            if step + 1 == self.maximum_frames:
                return ApproachResult(False, samples)
            if approach == ApproachFiles.FEEDBACK:
                depth_prior = target.front_depth(observation)
                correction = self.controller.compute_translation(
                    observation, measurement, depth_prior
                )
                config.camera_x += float(correction[0])
                config.camera_y += float(correction[1])
                config.camera_z += float(correction[2])
            config.camera_y += min(self.forward_step, max(remaining, 0))
        return ApproachResult(False, samples)
