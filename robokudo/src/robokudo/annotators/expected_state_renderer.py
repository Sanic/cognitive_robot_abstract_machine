"""Render a simple expected world state from current camera perspective."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np
from py_trees.common import Status
from sensor_msgs.msg import CameraInfo

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, Encoding, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.semdt_ground_truth import (
    get_ground_truth_world_ref,
    get_gt_pose_from_runtime_world,
)
from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Scale, Color


@dataclass
class OutlineMatchResult:
    shape_distance: float
    shape_score: float
    outline_iou: float
    outline_dice: float
    combined_score: float


@dataclass
class RefinementResult:
    center_world: np.ndarray
    render_bgr: np.ndarray
    expected_mask: np.ndarray
    outline_match: OutlineMatchResult
    pixel_error: float
    iterations: int
    converged: bool
    stop_reason: str
    score_history: list[float]
    center_history_world: list[list[float]]
    pixel_error_history: list[float]


@dataclass
class GroundTruthPoseEvaluation:
    body_name: str
    gt_center_world: np.ndarray
    initial_translation_error_m: float
    final_translation_error_m: float
    translation_error_history_m: list[float]


@dataclass
class GroundTruthObjectModel:
    body_name: str
    shape_type: str
    box_scale: Scale | None
    cylinder_width: float | None
    cylinder_height: float | None
    color: Color
    rotation_world: np.ndarray


class ExpectedStateRendererAnnotator(BaseAnnotator):
    """Render a stubbed expected world state containing only one known object."""

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                """Define tunable parameters for expected-object rendering and matching."""
                #: Classification label used to select the detected object hypothesis.
                self.target_classname: str = "box_blue"
                #: Name assigned to the expected object in the synthesized world model.
                self.expected_object_name: str = "expected_blue_box"
                #: Minimum accepted object depth from camera in meters.
                self.min_distance: float = 0.05
                #: Maximum accepted object depth from camera in meters.
                self.max_distance: float = 8.0
                #: PoseAnnotation source name used to read initial target pose.
                self.pose_annotation_source: str = "ClusterPoseBBAnnotator"
                #: Pixel thickness for drawing expected/detected outlines.
                self.contour_thickness: int = 1
                #: Max random translation magnitude applied to initialize refinement (m).
                self.random_offset_translation_m: float = 0.12
                #: Upper bound on optimization iterations per update.
                self.refinement_max_iterations: int = 25
                #: Finite-difference perturbation size for Jacobian estimation (m).
                self.refinement_jacobian_delta_m: float = 0.005
                #: Maximum translation step applied per refinement iteration (m).
                self.refinement_max_step_m: float = 0.05
                #: Minimum iterations before any convergence criterion can terminate.
                self.refinement_min_iterations_before_convergence: int = 3
                #: Convergence threshold for centroid error in pixels.
                self.refinement_convergence_pixel_error: float = 0.15
                #: Convergence threshold for score change between iterations.
                self.refinement_convergence_score_delta: float = 0.001
                #: Require px_err to be below this value before score-delta stop can trigger.
                self.refinement_score_delta_pixel_error_gate: float = 5.0
                #: Stop when px_err shows no meaningful improvement for this many iterations.
                self.refinement_pixel_error_patience_iterations: int = 8
                #: Minimum px_err decrease counted as meaningful improvement for patience.
                self.refinement_pixel_error_patience_min_delta: float = 0.03
                #: GT body name used as expected object model and quality reference.
                self.ground_truth_body_name: str = "box_blue"
                #: Fixed RNG seed for deterministic random initialization (-1 disables).
                self.random_seed: int = -1
                #: Enable writing visualization and mask images for each run.
                self.save_run_images: bool = True
                #: Directory where run and per-iteration images are written.
                self.run_image_output_dir: str = "/tmp"
                #: Target threshold for translation error metrics (meters).
                self.translation_error_goal_m: float = 0.02
                #: Enable appending one JSON log line per update for tuning.
                self.save_tuning_log_jsonl: bool = True
                #: JSONL file path used to store run-level tuning records.
                self.tuning_log_jsonl_path: str = "/tmp/expected_state_tuning_log.jsonl"

        parameters = Parameters()

    def __init__(
        self,
        name: str = "ExpectedStateRendererAnnotator",
        descriptor: "ExpectedStateRendererAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize renderer with descriptor-based expected-object settings."""
        super().__init__(name, descriptor)

    def update(self) -> Status:
        """Render expected object view, compare outlines, and publish visualization/metrics."""
        cam_info, cam_to_world_optical = self._read_camera_context()
        if cam_info is None or cam_to_world_optical is None:
            self.feedback_message = "No CameraInfo in CAS."
            return Status.FAILURE

        target_oh = self._find_target_object()
        if target_oh is None:
            self.feedback_message = f"No ObjectHypothesis with Classification '{self.descriptor.parameters.target_classname}'."
            blank = np.zeros(
                (int(cam_info.height), int(cam_info.width), 3), dtype=np.uint8
            )
            self.get_annotator_output_struct().set_image(blank)
            return Status.SUCCESS

        gt_object_model = self._resolve_ground_truth_object_model()
        if gt_object_model is None:
            self.feedback_message = (
                "No usable GT object model. Set 'ground_truth_body_name' and ensure CAS "
                "contains a valid GROUND_TRUTH_WORLD_REF."
            )
            blank = np.zeros(
                (int(cam_info.height), int(cam_info.width), 3), dtype=np.uint8
            )
            self.get_annotator_output_struct().set_image(blank)
            return Status.FAILURE

        target_center_world = self._target_center_world_from_pose_annotation(
            object_hypothesis=target_oh,
            cam_to_world_optical=cam_to_world_optical,
        )
        if target_center_world is None:
            self.feedback_message = f"Target object '{self.descriptor.parameters.target_classname}' has no usable PoseAnnotation."
            blank = np.zeros(
                (int(cam_info.height), int(cam_info.width), 3), dtype=np.uint8
            )
            self.get_annotator_output_struct().set_image(blank)
            return Status.FAILURE

        detected_mask = self._detected_mask_from_object_hypothesis(
            object_hypothesis=target_oh,
            image_width=int(cam_info.width),
            image_height=int(cam_info.height),
        )
        initial_center_world = self._apply_random_translation_offset(
            target_center_world
        )
        refinement = self._refine_expected_pose_translation(
            initial_center_world=initial_center_world,
            detected_mask=detected_mask,
            gt_object_model=gt_object_model,
            cam_info=cam_info,
            cam_to_world_optical=cam_to_world_optical,
        )
        gt_evaluation = self._evaluate_refinement_against_ground_truth(
            initial_center_world=initial_center_world,
            refinement=refinement,
        )

        self._store_outline_match_annotation(
            target_oh,
            refinement.outline_match,
            initial_center_world=initial_center_world,
            refined_center_world=refinement.center_world,
            iterations=refinement.iterations,
            converged=refinement.converged,
            pixel_error=refinement.pixel_error,
            score_history=refinement.score_history,
            center_history_world=refinement.center_history_world,
            pixel_error_history=refinement.pixel_error_history,
            gt_evaluation=gt_evaluation,
        )
        outline_match = refinement.outline_match
        render_bgr = refinement.render_bgr
        expected_mask = refinement.expected_mask
        render_bgr = self._draw_outlines(render_bgr, expected_mask, detected_mask)
        fov_deg = self._fov_deg_from_cam_info(cam_info)
        score_start = (
            float(refinement.score_history[0])
            if len(refinement.score_history) > 0
            else outline_match.combined_score
        )
        score_final = (
            float(refinement.score_history[-1])
            if len(refinement.score_history) > 0
            else outline_match.combined_score
        )
        self._log_refinement_summary(
            initial_center_world=initial_center_world,
            refinement=refinement,
            gt_evaluation=gt_evaluation,
        )
        self._log_tuning_snapshot(
            initial_center_world=initial_center_world,
            target_center_world=target_center_world,
            refinement=refinement,
            gt_evaluation=gt_evaluation,
        )

        cv2.putText(
            render_bgr,
            (
                f"expected: {self.descriptor.parameters.target_classname} | "
                f"outline_match: {outline_match.combined_score:.3f}"
            ),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            render_bgr,
            (
                f"score {score_start:.3f}->{score_final:.3f} | "
                f"iters {refinement.iterations} | px_err {refinement.pixel_error:.2f}"
            ),
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if gt_evaluation is not None:
            cv2.putText(
                render_bgr,
                (
                    f"gt_err_m {gt_evaluation.initial_translation_error_m:.4f}"
                    f"->{gt_evaluation.final_translation_error_m:.4f}"
                ),
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self._save_run_images(
            visualization_bgr=render_bgr,
            expected_mask=expected_mask,
            detected_mask=detected_mask,
        )

        self.get_annotator_output_struct().set_image(render_bgr)
        self.feedback_message = (
            f"Rendered expected state for '{self.descriptor.parameters.target_classname}' "
            f"(FOV {fov_deg:.1f} deg, outline_match={outline_match.combined_score:.3f}, "
            f"score {score_start:.3f}->{score_final:.3f}, iters={refinement.iterations}, "
            f"init_xyz={[round(float(v), 3) for v in initial_center_world]}, "
            f"final_xyz={[round(float(v), 3) for v in refinement.center_world]})."
        )
        if gt_evaluation is not None:
            goal_m = max(
                float(self.descriptor.parameters.translation_error_goal_m), 0.0
            )
            self.feedback_message += (
                f" GT '{gt_evaluation.body_name}' err_m "
                f"{gt_evaluation.initial_translation_error_m:.4f}"
                f"->{gt_evaluation.final_translation_error_m:.4f}, "
                f"goal={goal_m:.4f}, "
                f"init_ok={gt_evaluation.initial_translation_error_m <= goal_m}, "
                f"final_ok={gt_evaluation.final_translation_error_m <= goal_m}."
            )
        return Status.SUCCESS

    def _save_run_images(
        self,
        visualization_bgr: np.ndarray,
        expected_mask: np.ndarray,
        detected_mask: np.ndarray,
        file_suffix: str = "",
    ) -> None:
        """Persist this run's visualization and masks under /tmp."""
        if not bool(self.descriptor.parameters.save_run_images):
            return
        output_dir = Path(str(self.descriptor.parameters.run_image_output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = self._create_run_id()
        suffix = str(file_suffix)

        vis_path = output_dir / f"{run_id}{suffix}_vis.png"
        expected_path = output_dir / f"{run_id}{suffix}_expected_mask.png"
        detected_path = output_dir / f"{run_id}{suffix}_detected_mask.png"
        cv2.imwrite(str(vis_path), visualization_bgr)
        cv2.imwrite(str(expected_path), expected_mask)
        cv2.imwrite(str(detected_path), detected_mask)
        self.rk_logger.info(
            "ExpectedState saved run images: %s, %s, %s",
            str(vis_path),
            str(expected_path),
            str(detected_path),
        )

    def _create_run_id(self) -> str:
        """Create a stable unique id for one annotator update run."""
        cas = self.get_cas()
        timestamp_ns = getattr(cas, "data_timestamp", None)
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        return f"expected_state_{int(timestamp_ns)}"

    def _apply_random_translation_offset(self, center_world: np.ndarray) -> np.ndarray:
        """Return center pose with a uniformly sampled translation perturbation."""
        magnitude = float(self.descriptor.parameters.random_offset_translation_m)
        if magnitude <= 0.0:
            return center_world.copy()
        random_seed = int(self.descriptor.parameters.random_seed)
        if random_seed >= 0:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        random_direction = rng.normal(size=3).astype(np.float64)
        direction_norm = float(np.linalg.norm(random_direction))
        if direction_norm < 1e-9:
            random_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            direction_norm = 1.0
        random_direction /= direction_norm
        random_radius = float(rng.uniform(0.0, magnitude))
        offset = random_direction * random_radius
        return center_world + offset

    def _evaluate_refinement_against_ground_truth(
        self, initial_center_world: np.ndarray, refinement: RefinementResult
    ) -> GroundTruthPoseEvaluation | None:
        """Evaluate initial/final refinement translation error against GT world pose."""
        body_name = str(self.descriptor.parameters.ground_truth_body_name).strip()
        if body_name == "":
            return None
        pose_world = get_gt_pose_from_runtime_world(self.get_cas(), body_name)
        if pose_world is None:
            self.rk_logger.warning(
                "ExpectedState GT body '%s' not available in CAS ground-truth world.",
                body_name,
            )
            return None
        gt_center_world = np.asarray(pose_world[:3, 3], dtype=np.float64)
        initial_error = float(np.linalg.norm(initial_center_world - gt_center_world))
        final_error = float(np.linalg.norm(refinement.center_world - gt_center_world))
        error_history = [
            float(
                np.linalg.norm(np.asarray(center, dtype=np.float64) - gt_center_world)
            )
            for center in refinement.center_history_world
        ]
        return GroundTruthPoseEvaluation(
            body_name=body_name,
            gt_center_world=gt_center_world,
            initial_translation_error_m=initial_error,
            final_translation_error_m=final_error,
            translation_error_history_m=error_history,
        )

    def _resolve_ground_truth_object_model(self) -> GroundTruthObjectModel | None:
        """Read expected object box scale/color/orientation from GT world."""
        body_name = str(self.descriptor.parameters.ground_truth_body_name).strip()
        if body_name == "":
            self.rk_logger.warning(
                "ExpectedState ground_truth_body_name is empty; GT model is unavailable."
            )
            return None
        world = get_ground_truth_world_ref(self.get_cas())
        if world is None:
            self.rk_logger.warning(
                "ExpectedState CAS has no GROUND_TRUTH_WORLD_REF; GT model unavailable."
            )
            return None
        bodies = world.get_bodies_by_name(body_name)
        if len(bodies) == 0:
            self.rk_logger.warning(
                "ExpectedState GT body '%s' not found in ground-truth world.",
                body_name,
            )
            return None
        body = bodies[0]
        if len(body.collision) == 0:
            self.rk_logger.warning(
                "ExpectedState GT body '%s' has no collision geometry.",
                body_name,
            )
            return None

        pose_world = np.asarray(body.global_pose.to_np(), dtype=np.float64)
        if pose_world.shape != (4, 4) or not np.all(np.isfinite(pose_world)):
            self.rk_logger.warning(
                "ExpectedState GT body '%s' has invalid global pose.",
                body_name,
            )
            return None
        rotation_world = pose_world[:3, :3]
        if not np.all(np.isfinite(rotation_world)):
            self.rk_logger.warning(
                "ExpectedState GT body '%s' has invalid pose rotation.",
                body_name,
            )
            return None

        shape = body.collision[0]
        shape_type: str
        box_scale: Scale | None = None
        cylinder_width: float | None = None
        cylinder_height: float | None = None
        if isinstance(shape, Box):
            shape_type = "box"
            box_scale = shape.scale
            if (
                box_scale is None
                or box_scale.x <= 0.0
                or box_scale.y <= 0.0
                or box_scale.z <= 0.0
                or not np.all(np.isfinite([box_scale.x, box_scale.y, box_scale.z]))
            ):
                self.rk_logger.warning(
                    "ExpectedState GT body '%s' has invalid box scale.",
                    body_name,
                )
                return None
        elif isinstance(shape, Cylinder):
            shape_type = "cylinder"
            cylinder_width = float(shape.width)
            cylinder_height = float(shape.height)
            if (
                cylinder_width <= 0.0
                or cylinder_height <= 0.0
                or not np.all(np.isfinite([cylinder_width, cylinder_height]))
            ):
                self.rk_logger.warning(
                    "ExpectedState GT body '%s' has invalid cylinder dimensions.",
                    body_name,
                )
                return None
        else:
            self.rk_logger.warning(
                "ExpectedState GT body '%s' has unsupported shape type '%s'.",
                body_name,
                type(shape).__name__,
            )
            return None
        shape_color = shape.color

        return GroundTruthObjectModel(
            body_name=body_name,
            shape_type=shape_type,
            box_scale=(
                Scale(float(box_scale.x), float(box_scale.y), float(box_scale.z))
                if box_scale is not None
                else None
            ),
            cylinder_width=(
                float(cylinder_width) if cylinder_width is not None else None
            ),
            cylinder_height=(
                float(cylinder_height) if cylinder_height is not None else None
            ),
            color=Color(
                float(shape_color.R),
                float(shape_color.G),
                float(shape_color.B),
                float(shape_color.A),
            ),
            rotation_world=rotation_world.copy(),
        )

    def _refine_expected_pose_translation(
        self,
        initial_center_world: np.ndarray,
        detected_mask: np.ndarray,
        gt_object_model: GroundTruthObjectModel,
        cam_info: CameraInfo,
        cam_to_world_optical: np.ndarray,
    ) -> RefinementResult:
        """Iteratively shift expected object translation to maximize outline agreement."""
        max_iterations = max(
            int(self.descriptor.parameters.refinement_max_iterations), 1
        )
        convergence_pixel_error = float(
            self.descriptor.parameters.refinement_convergence_pixel_error
        )
        min_iterations_before_convergence = max(
            int(
                self.descriptor.parameters.refinement_min_iterations_before_convergence
            ),
            1,
        )
        convergence_score_delta = float(
            self.descriptor.parameters.refinement_convergence_score_delta
        )
        score_delta_pixel_error_gate = float(
            self.descriptor.parameters.refinement_score_delta_pixel_error_gate
        )
        pixel_error_patience_iterations = max(
            int(self.descriptor.parameters.refinement_pixel_error_patience_iterations),
            0,
        )
        pixel_error_patience_min_delta = max(
            float(self.descriptor.parameters.refinement_pixel_error_patience_min_delta),
            0.0,
        )

        current_center = initial_center_world.astype(np.float64).copy()
        previous_score = None
        best_result = None
        executed_iterations = 0
        score_history: list[float] = []
        center_history_world: list[list[float]] = []
        pixel_error_history: list[float] = []
        stop_reason = "max_iterations_reached"
        best_pixel_error = float("inf")
        last_pixel_error_improvement_iteration = 0

        for iteration in range(max_iterations):
            executed_iterations = iteration + 1
            render_bgr, expected_mask = self._render_expected_mask_for_center(
                object_center_world=current_center,
                gt_object_model=gt_object_model,
                cam_info=cam_info,
                cam_to_world_optical=cam_to_world_optical,
            )
            outline_match = self._compute_outline_match(expected_mask, detected_mask)
            centroid_shift_px = self._centroid_shift_px(
                expected_mask=expected_mask, detected_mask=detected_mask
            )
            pixel_error = (
                float(np.linalg.norm(centroid_shift_px))
                if centroid_shift_px is not None
                else float("inf")
            )
            score_history.append(float(outline_match.combined_score))
            center_history_world.append(list(current_center.astype(float)))
            pixel_error_history.append(float(pixel_error))
            self._log_refinement_iteration(
                iteration=iteration + 1,
                center_world=current_center,
                outline_match=outline_match,
                pixel_error=pixel_error,
            )
            iteration_vis = self._draw_outlines(
                render_bgr.copy(), expected_mask, detected_mask
            )
            cv2.putText(
                iteration_vis,
                (
                    f"iter {iteration + 1}/{max_iterations} | "
                    f"outline_match {outline_match.combined_score:.3f} | "
                    f"px_err {pixel_error:.2f}"
                ),
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                iteration_vis,
                (
                    f"iou {outline_match.outline_iou:.3f} | "
                    f"dice {outline_match.outline_dice:.3f} | "
                    f"shape_dist {outline_match.shape_distance:.2f}"
                ),
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self._save_run_images(
                visualization_bgr=iteration_vis,
                expected_mask=expected_mask,
                detected_mask=detected_mask,
                file_suffix=f"_iter_{iteration + 1:02d}",
            )

            current_result = RefinementResult(
                center_world=current_center.copy(),
                render_bgr=render_bgr,
                expected_mask=expected_mask,
                outline_match=outline_match,
                pixel_error=pixel_error,
                iterations=iteration + 1,
                converged=False,
                stop_reason="in_progress",
                score_history=list(score_history),
                center_history_world=[list(xyz) for xyz in center_history_world],
                pixel_error_history=list(pixel_error_history),
            )
            if (
                np.isfinite(pixel_error)
                and pixel_error + pixel_error_patience_min_delta < best_pixel_error
            ):
                best_pixel_error = pixel_error
                last_pixel_error_improvement_iteration = executed_iterations
            if (
                best_result is None
                or (
                    np.isfinite(current_result.pixel_error)
                    and not np.isfinite(best_result.pixel_error)
                )
                or (
                    np.isfinite(current_result.pixel_error)
                    and np.isfinite(best_result.pixel_error)
                    and current_result.pixel_error < best_result.pixel_error - 1e-3
                )
                or (
                    (
                        not np.isfinite(current_result.pixel_error)
                        or not np.isfinite(best_result.pixel_error)
                        or abs(current_result.pixel_error - best_result.pixel_error)
                        <= 1e-3
                    )
                    and current_result.outline_match.combined_score
                    > best_result.outline_match.combined_score
                )
            ):
                best_result = current_result

            score_delta = (
                abs(outline_match.combined_score - previous_score)
                if previous_score is not None
                else float("inf")
            )
            previous_score = outline_match.combined_score
            if (
                executed_iterations >= min_iterations_before_convergence
                and pixel_error <= convergence_pixel_error
            ):
                current_result.converged = True
                current_result.stop_reason = (
                    "pixel_error_converged "
                    f"(px_err={pixel_error:.2f} <= {convergence_pixel_error:.2f}, "
                    f"iter={executed_iterations} >= {min_iterations_before_convergence})"
                )
                current_result.score_history = list(score_history)
                current_result.center_history_world = [
                    list(xyz) for xyz in center_history_world
                ]
                current_result.pixel_error_history = list(pixel_error_history)
                return current_result
            if (
                executed_iterations >= min_iterations_before_convergence
                and score_delta <= convergence_score_delta
                and iteration > 0
                and pixel_error <= score_delta_pixel_error_gate
            ):
                if best_result is not None:
                    best_result.converged = True
                    best_result.stop_reason = (
                        "score_delta_converged "
                        f"(delta={score_delta:.6f} <= {convergence_score_delta:.6f}, "
                        f"px_err={pixel_error:.2f} <= {score_delta_pixel_error_gate:.2f}, "
                        f"iter={executed_iterations} >= {min_iterations_before_convergence})"
                    )
                    best_result.iterations = executed_iterations
                    best_result.score_history = list(score_history)
                    best_result.center_history_world = [
                        list(xyz) for xyz in center_history_world
                    ]
                    best_result.pixel_error_history = list(pixel_error_history)
                    return best_result
                current_result.converged = True
                current_result.stop_reason = (
                    "score_delta_converged "
                    f"(delta={score_delta:.6f} <= {convergence_score_delta:.6f}, "
                    f"px_err={pixel_error:.2f} <= {score_delta_pixel_error_gate:.2f}, "
                    f"iter={executed_iterations} >= {min_iterations_before_convergence})"
                )
                current_result.score_history = list(score_history)
                current_result.center_history_world = [
                    list(xyz) for xyz in center_history_world
                ]
                current_result.pixel_error_history = list(pixel_error_history)
                return current_result
            if (
                pixel_error_patience_iterations > 0
                and executed_iterations - last_pixel_error_improvement_iteration
                >= pixel_error_patience_iterations
                and pixel_error > convergence_pixel_error
            ):
                stop_reason = (
                    "pixel_error_plateau "
                    f"(no improvement >= {pixel_error_patience_min_delta:.2f}px for "
                    f"{pixel_error_patience_iterations} iterations)"
                )
                break

            if centroid_shift_px is None:
                stop_reason = "centroid_shift_unavailable"
                break

            shift_world = self._estimate_translation_shift_world(
                current_center_world=current_center,
                expected_mask=expected_mask,
                detected_mask=detected_mask,
                gt_object_model=gt_object_model,
                cam_info=cam_info,
                cam_to_world_optical=cam_to_world_optical,
            )
            if shift_world is None:
                stop_reason = "shift_estimation_failed"
                break
            if np.linalg.norm(shift_world) < 1e-6:
                stop_reason = "step_too_small"
                break
            current_center = current_center + shift_world

        if best_result is None:
            render_bgr, expected_mask = self._render_expected_mask_for_center(
                object_center_world=current_center,
                gt_object_model=gt_object_model,
                cam_info=cam_info,
                cam_to_world_optical=cam_to_world_optical,
            )
            outline_match = self._compute_outline_match(expected_mask, detected_mask)
            best_result = RefinementResult(
                center_world=current_center.copy(),
                render_bgr=render_bgr,
                expected_mask=expected_mask,
                outline_match=outline_match,
                pixel_error=float("inf"),
                iterations=executed_iterations,
                converged=False,
                stop_reason=stop_reason,
                score_history=list(score_history),
                center_history_world=[list(xyz) for xyz in center_history_world],
                pixel_error_history=list(pixel_error_history),
            )
        else:
            best_result.iterations = executed_iterations
            best_result.stop_reason = stop_reason
            best_result.score_history = list(score_history)
            best_result.center_history_world = [
                list(xyz) for xyz in center_history_world
            ]
            best_result.pixel_error_history = list(pixel_error_history)
        return best_result

    def _log_refinement_iteration(
        self,
        iteration: int,
        center_world: np.ndarray,
        outline_match: OutlineMatchResult,
        pixel_error: float,
    ) -> None:
        """Emit one refinement step with pose and score metrics."""
        self.rk_logger.info(
            (
                "ExpectedState refinement iter %d: "
                "pose=[%.4f, %.4f, %.4f], score=%.4f, iou=%.4f, dice=%.4f, px_err=%.2f"
            ),
            int(iteration),
            float(center_world[0]),
            float(center_world[1]),
            float(center_world[2]),
            float(outline_match.combined_score),
            float(outline_match.outline_iou),
            float(outline_match.outline_dice),
            float(pixel_error),
        )

    def _log_refinement_summary(
        self,
        initial_center_world: np.ndarray,
        refinement: RefinementResult,
        gt_evaluation: GroundTruthPoseEvaluation | None = None,
    ) -> None:
        """Emit final refinement summary with initial/final poses and score progression."""
        score_start = (
            float(refinement.score_history[0])
            if len(refinement.score_history) > 0
            else float(refinement.outline_match.combined_score)
        )
        score_final = (
            float(refinement.score_history[-1])
            if len(refinement.score_history) > 0
            else float(refinement.outline_match.combined_score)
        )
        self.rk_logger.info(
            (
                "ExpectedState refinement summary: "
                "initial=[%.4f, %.4f, %.4f], final=[%.4f, %.4f, %.4f], "
                "score=%.4f->%.4f, iterations=%d, converged=%s, final_px_err=%.2f, stop=%s"
            ),
            float(initial_center_world[0]),
            float(initial_center_world[1]),
            float(initial_center_world[2]),
            float(refinement.center_world[0]),
            float(refinement.center_world[1]),
            float(refinement.center_world[2]),
            score_start,
            score_final,
            int(refinement.iterations),
            bool(refinement.converged),
            float(refinement.pixel_error),
            str(refinement.stop_reason),
        )
        if gt_evaluation is not None:
            self.rk_logger.info(
                (
                    "ExpectedState GT eval (%s): gt=[%.4f, %.4f, %.4f], "
                    "err_m=%.4f->%.4f"
                ),
                gt_evaluation.body_name,
                float(gt_evaluation.gt_center_world[0]),
                float(gt_evaluation.gt_center_world[1]),
                float(gt_evaluation.gt_center_world[2]),
                float(gt_evaluation.initial_translation_error_m),
                float(gt_evaluation.final_translation_error_m),
            )

    def _log_tuning_snapshot(
        self,
        initial_center_world: np.ndarray,
        target_center_world: np.ndarray,
        refinement: RefinementResult,
        gt_evaluation: GroundTruthPoseEvaluation | None,
    ) -> None:
        """Emit and persist a structured tuning snapshot for offline parameter search."""
        goal_m = max(float(self.descriptor.parameters.translation_error_goal_m), 0.0)
        run_id = self._create_run_id()
        score_start = (
            float(refinement.score_history[0])
            if len(refinement.score_history) > 0
            else float(refinement.outline_match.combined_score)
        )
        score_final = (
            float(refinement.score_history[-1])
            if len(refinement.score_history) > 0
            else float(refinement.outline_match.combined_score)
        )

        record = {
            "run_id": run_id,
            "target_classname": str(self.descriptor.parameters.target_classname),
            "ground_truth_body_name": str(
                self.descriptor.parameters.ground_truth_body_name
            ),
            "random_seed": int(self.descriptor.parameters.random_seed),
            "random_offset_translation_m": float(
                self.descriptor.parameters.random_offset_translation_m
            ),
            "refinement_max_iterations": int(
                self.descriptor.parameters.refinement_max_iterations
            ),
            "refinement_jacobian_delta_m": float(
                self.descriptor.parameters.refinement_jacobian_delta_m
            ),
            "refinement_max_step_m": float(
                self.descriptor.parameters.refinement_max_step_m
            ),
            "refinement_min_iterations_before_convergence": int(
                self.descriptor.parameters.refinement_min_iterations_before_convergence
            ),
            "refinement_convergence_pixel_error": float(
                self.descriptor.parameters.refinement_convergence_pixel_error
            ),
            "refinement_convergence_score_delta": float(
                self.descriptor.parameters.refinement_convergence_score_delta
            ),
            "refinement_score_delta_pixel_error_gate": float(
                self.descriptor.parameters.refinement_score_delta_pixel_error_gate
            ),
            "refinement_pixel_error_patience_iterations": int(
                self.descriptor.parameters.refinement_pixel_error_patience_iterations
            ),
            "refinement_pixel_error_patience_min_delta": float(
                self.descriptor.parameters.refinement_pixel_error_patience_min_delta
            ),
            "outline_score_start": float(score_start),
            "outline_score_final": float(score_final),
            "outline_score_best": float(refinement.outline_match.combined_score),
            "final_centroid_pixel_error": float(refinement.pixel_error),
            "refinement_iterations": int(refinement.iterations),
            "refinement_converged": bool(refinement.converged),
            "refinement_stop_reason": str(refinement.stop_reason),
            "initial_center_world_xyz": list(initial_center_world.astype(float)),
            "pose_annotation_center_world_xyz": list(target_center_world.astype(float)),
            "refined_center_world_xyz": list(refinement.center_world.astype(float)),
            "init_offset_from_pose_annotation_m": float(
                np.linalg.norm(initial_center_world - target_center_world)
            ),
            "translation_error_goal_m": float(goal_m),
        }

        if gt_evaluation is not None:
            pose_annotation_error = float(
                np.linalg.norm(target_center_world - gt_evaluation.gt_center_world)
            )
            record.update(
                {
                    "ground_truth_center_world_xyz": list(
                        gt_evaluation.gt_center_world.astype(float)
                    ),
                    "ground_truth_translation_error_initial_m": float(
                        gt_evaluation.initial_translation_error_m
                    ),
                    "ground_truth_translation_error_final_m": float(
                        gt_evaluation.final_translation_error_m
                    ),
                    "ground_truth_translation_error_pose_annotation_m": pose_annotation_error,
                    "ground_truth_initial_meets_goal": bool(
                        gt_evaluation.initial_translation_error_m <= goal_m
                    ),
                    "ground_truth_final_meets_goal": bool(
                        gt_evaluation.final_translation_error_m <= goal_m
                    ),
                }
            )
            self.rk_logger.info(
                (
                    "ExpectedState tuning: run=%s, init_err=%.4fm, final_err=%.4fm, "
                    "goal=%.4fm, init_ok=%s, final_ok=%s, stop=%s, iters=%d"
                ),
                run_id,
                float(gt_evaluation.initial_translation_error_m),
                float(gt_evaluation.final_translation_error_m),
                float(goal_m),
                bool(gt_evaluation.initial_translation_error_m <= goal_m),
                bool(gt_evaluation.final_translation_error_m <= goal_m),
                str(refinement.stop_reason),
                int(refinement.iterations),
            )
            if gt_evaluation.final_translation_error_m > goal_m:
                self.rk_logger.warning(
                    (
                        "ExpectedState final translation error above goal: "
                        "%.4fm > %.4fm (run=%s)"
                    ),
                    float(gt_evaluation.final_translation_error_m),
                    float(goal_m),
                    run_id,
                )
        else:
            self.rk_logger.info(
                "ExpectedState tuning: run=%s, no GT evaluation available.",
                run_id,
            )

        self._append_tuning_log_jsonl(record)

    def _append_tuning_log_jsonl(self, record: dict[str, object]) -> None:
        """Append one JSON object per line for later batch analysis."""
        if not bool(self.descriptor.parameters.save_tuning_log_jsonl):
            return
        log_path = Path(str(self.descriptor.parameters.tuning_log_jsonl_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True))
            fh.write("\n")

    def _render_expected_mask_for_center(
        self,
        object_center_world: np.ndarray,
        gt_object_model: GroundTruthObjectModel,
        cam_info: CameraInfo,
        cam_to_world_optical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render expected-world BGR and binary mask for a concrete object translation."""
        expected_world = self._build_expected_world(
            object_center_world=object_center_world,
            gt_object_model=gt_object_model,
        )
        render_bgr, expected_mask, _ = self._render_expected_world(
            expected_world=expected_world,
            cam_info=cam_info,
            cam_to_world_optical=cam_to_world_optical,
        )
        return render_bgr, expected_mask

    def _estimate_translation_shift_world(
        self,
        current_center_world: np.ndarray,
        expected_mask: np.ndarray,
        detected_mask: np.ndarray,
        gt_object_model: GroundTruthObjectModel,
        cam_info: CameraInfo,
        cam_to_world_optical: np.ndarray,
    ) -> np.ndarray | None:
        """Estimate world-frame translation step from centroid error and local Jacobian."""
        centroid_shift_px = self._centroid_shift_px(
            expected_mask=expected_mask, detected_mask=detected_mask
        )
        expected_center_px = self._mask_centroid(expected_mask)
        if centroid_shift_px is None or expected_center_px is None:
            return None

        delta_m = max(
            float(self.descriptor.parameters.refinement_jacobian_delta_m), 1e-4
        )
        jacobian = np.zeros((2, 3), dtype=np.float64)
        for axis in range(3):
            perturbed_center = current_center_world.copy()
            perturbed_center[axis] += delta_m
            _, perturbed_mask = self._render_expected_mask_for_center(
                object_center_world=perturbed_center,
                gt_object_model=gt_object_model,
                cam_info=cam_info,
                cam_to_world_optical=cam_to_world_optical,
            )
            perturbed_center_px = self._mask_centroid(perturbed_mask)
            if perturbed_center_px is None:
                continue
            jacobian[:, axis] = (perturbed_center_px - expected_center_px) / delta_m

        if np.linalg.norm(jacobian) < 1e-8:
            return None

        shift_world, _, _, _ = np.linalg.lstsq(jacobian, centroid_shift_px, rcond=None)
        if not np.all(np.isfinite(shift_world)):
            return None

        max_step_m = max(float(self.descriptor.parameters.refinement_max_step_m), 1e-4)
        step_norm = float(np.linalg.norm(shift_world))
        if step_norm > max_step_m:
            shift_world = shift_world * (max_step_m / step_norm)
        return shift_world

    def _centroid_shift_px(
        self, expected_mask: np.ndarray, detected_mask: np.ndarray
    ) -> np.ndarray | None:
        """Return detected-minus-expected centroid shift in pixel coordinates."""
        expected_centroid = self._mask_centroid(expected_mask)
        detected_centroid = self._mask_centroid(detected_mask)
        if expected_centroid is None or detected_centroid is None:
            return None
        return detected_centroid - expected_centroid

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> np.ndarray | None:
        """Compute contour centroid in pixels for the largest object mask component."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return np.array([cx, cy], dtype=np.float64)

    def _read_camera_context(
        self,
    ) -> tuple[CameraInfo | None, np.ndarray | None]:
        """Read camera intrinsics and cam-to-world optical transform from CAS."""
        cas = self.get_cas()
        cam_info = cas.get(CASViews.CAM_INFO)
        if not isinstance(cam_info, CameraInfo):
            return None, None
        cam_to_world_optical = cas.cam_to_world_transform
        if cam_to_world_optical is None:
            return cam_info, None
        return cam_info, cam_to_world_optical.to_np()

    def _find_target_object(self) -> ObjectHypothesis | None:
        """Return best-matching target hypothesis by classname and highest confidence."""
        target_class = self.descriptor.parameters.target_classname
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)
        if object_hypotheses is None:
            return None

        best_oh = None
        best_confidence = -1.0
        for object_hypothesis in object_hypotheses:
            for annotation in object_hypothesis.annotations:
                if (
                    isinstance(annotation, Classification)
                    and annotation.classname == target_class
                    and float(annotation.confidence) >= best_confidence
                ):
                    best_confidence = float(annotation.confidence)
                    best_oh = object_hypothesis

        return best_oh

    def _target_center_world_from_pose_annotation(
        self, object_hypothesis: ObjectHypothesis, cam_to_world_optical: np.ndarray
    ) -> np.ndarray | None:
        """Compute world-space target center from preferred PoseAnnotation on the object."""
        pose_annotations = self.get_cas().filter_by_type(
            PoseAnnotation, object_hypothesis.annotations
        )
        preferred_source = str(self.descriptor.parameters.pose_annotation_source)
        pose_annotation = None
        if preferred_source:
            for annotation in pose_annotations:
                if annotation.source == preferred_source:
                    pose_annotation = annotation
                    break
        if pose_annotation is None and len(pose_annotations) > 0:
            pose_annotation = pose_annotations[0]

        if (
            pose_annotation is not None
            and pose_annotation.translation is not None
            and len(pose_annotation.translation) >= 3
        ):
            center_cam = np.asarray(pose_annotation.translation[:3], dtype=np.float64)
            center_world = cam_to_world_optical @ np.array(
                [center_cam[0], center_cam[1], center_cam[2], 1.0], dtype=np.float64
            )
            return center_world[:3]
        return None

    def _render_expected_world(
        self,
        expected_world,
        cam_info: CameraInfo,
        cam_to_world_optical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Render expected world from current perspective and return image/mask/FOV."""
        resolution = int(min(cam_info.width, cam_info.height))
        fov_deg = self._fov_deg_from_cam_info(cam_info)
        camera_link_pose = HomogeneousTransformationMatrix(
            data=cam_to_world_optical @ self._camera_optical_to_link_np(),
            reference_frame=expected_world.root,
        )

        segmentation = expected_world.ray_tracer.create_segmentation_mask(
            camera_link_pose,
            resolution=resolution,
            min_distance=float(self.descriptor.parameters.min_distance),
            max_distance=float(self.descriptor.parameters.max_distance),
        )
        render_bgr = self._segmentation_to_bgr(
            expected_world=expected_world,
            segmentation=segmentation,
        )
        expected_mask = np.zeros(segmentation.shape, dtype=np.uint8)
        expected_mask[segmentation >= 0] = 255

        if render_bgr.shape[0] != int(cam_info.height) or render_bgr.shape[1] != int(
            cam_info.width
        ):
            render_bgr = cv2.resize(
                render_bgr,
                (int(cam_info.width), int(cam_info.height)),
                interpolation=cv2.INTER_NEAREST,
            )
            expected_mask = cv2.resize(
                expected_mask,
                (int(cam_info.width), int(cam_info.height)),
                interpolation=cv2.INTER_NEAREST,
            )

        return render_bgr, expected_mask, fov_deg

    @staticmethod
    def _detected_mask_from_object_hypothesis(
        object_hypothesis: ObjectHypothesis, image_width: int, image_height: int
    ) -> np.ndarray:
        """Build full-image binary mask for detected object from ROI and optional ROI mask."""
        mask_full = np.zeros((image_height, image_width), dtype=np.uint8)
        roi = object_hypothesis.roi.roi
        x = max(int(roi.pos.x), 0)
        y = max(int(roi.pos.y), 0)
        w = max(int(roi.width), 0)
        h = max(int(roi.height), 0)
        if w == 0 or h == 0:
            return mask_full
        x2 = min(x + w, image_width)
        y2 = min(y + h, image_height)
        if x2 <= x or y2 <= y:
            return mask_full

        roi_h = y2 - y
        roi_w = x2 - x
        if object_hypothesis.roi.mask is None:
            mask_full[y:y2, x:x2] = 255
            return mask_full

        roi_mask = object_hypothesis.roi.mask
        if roi_mask.shape[0] < roi_h or roi_mask.shape[1] < roi_w:
            mask_full[y:y2, x:x2] = 255
            return mask_full
        mask_full[y:y2, x:x2] = np.where(roi_mask[:roi_h, :roi_w] > 0, 255, 0)
        return mask_full

    def _compute_outline_match(
        self, expected_mask: np.ndarray, detected_mask: np.ndarray
    ) -> OutlineMatchResult:
        """Compute contour-shape and overlap metrics between expected and detected outlines."""
        expected_contour = self._largest_contour(expected_mask)
        detected_contour = self._largest_contour(detected_mask)
        if expected_contour is None or detected_contour is None:
            return OutlineMatchResult(
                shape_distance=1.0,
                shape_score=0.0,
                outline_iou=0.0,
                outline_dice=0.0,
                combined_score=0.0,
            )

        shape_distance = float(
            cv2.matchShapes(
                expected_contour, detected_contour, cv2.CONTOURS_MATCH_I1, 0.0
            )
        )
        shape_score = 1.0 / (1.0 + shape_distance)

        thickness = max(int(self.descriptor.parameters.contour_thickness), 1)
        expected_outline = np.zeros_like(expected_mask, dtype=np.uint8)
        detected_outline = np.zeros_like(detected_mask, dtype=np.uint8)
        cv2.drawContours(expected_outline, [expected_contour], -1, 255, thickness)
        cv2.drawContours(detected_outline, [detected_contour], -1, 255, thickness)

        expected_bool = expected_outline > 0
        detected_bool = detected_outline > 0
        intersection = int(np.count_nonzero(expected_bool & detected_bool))
        union = int(np.count_nonzero(expected_bool | detected_bool))
        expected_count = int(np.count_nonzero(expected_bool))
        detected_count = int(np.count_nonzero(detected_bool))

        outline_iou = float(intersection / union) if union > 0 else 0.0
        denom = expected_count + detected_count
        outline_dice = float((2.0 * intersection) / denom) if denom > 0 else 0.0
        combined_score = float(
            np.clip((shape_score + outline_iou + outline_dice) / 3.0, 0.0, 1.0)
        )

        return OutlineMatchResult(
            shape_distance=shape_distance,
            shape_score=float(np.clip(shape_score, 0.0, 1.0)),
            outline_iou=float(np.clip(outline_iou, 0.0, 1.0)),
            outline_dice=float(np.clip(outline_dice, 0.0, 1.0)),
            combined_score=combined_score,
        )

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        """Return largest external contour in mask, or None when no contour exists."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None
        return max(contours, key=cv2.contourArea)

    def _store_outline_match_annotation(
        self,
        object_hypothesis: ObjectHypothesis,
        outline_match: OutlineMatchResult,
        initial_center_world: np.ndarray,
        refined_center_world: np.ndarray,
        iterations: int,
        converged: bool,
        pixel_error: float,
        score_history: list[float],
        center_history_world: list[list[float]],
        pixel_error_history: list[float],
        gt_evaluation: GroundTruthPoseEvaluation | None = None,
    ) -> None:
        """Store outline-match metrics as an Encoding annotation on target object."""
        source = "ExpectedStateRendererOutlineMatch"
        object_hypothesis.annotations = [
            annotation
            for annotation in object_hypothesis.annotations
            if not (isinstance(annotation, Encoding) and annotation.source == source)
        ]
        annotation = Encoding()
        annotation.source = source
        annotation.encoding = {
            "shape_distance": outline_match.shape_distance,
            "shape_score": outline_match.shape_score,
            "outline_iou": outline_match.outline_iou,
            "outline_dice": outline_match.outline_dice,
            "combined_score": outline_match.combined_score,
            "initial_center_world_xyz": list(initial_center_world.astype(float)),
            "refined_center_world_xyz": list(refined_center_world.astype(float)),
            "refinement_iterations": int(iterations),
            "refinement_converged": bool(converged),
            "centroid_pixel_error": float(pixel_error),
            "refinement_score_history": [float(x) for x in score_history],
            "refinement_pose_history_world_xyz": [
                [float(v) for v in xyz] for xyz in center_history_world
            ],
            "refinement_pixel_error_history": [float(x) for x in pixel_error_history],
        }
        if gt_evaluation is not None:
            goal_m = max(
                float(self.descriptor.parameters.translation_error_goal_m), 0.0
            )
            annotation.encoding["ground_truth_body_name"] = gt_evaluation.body_name
            annotation.encoding["ground_truth_center_world_xyz"] = list(
                gt_evaluation.gt_center_world.astype(float)
            )
            annotation.encoding["ground_truth_translation_error_initial_m"] = float(
                gt_evaluation.initial_translation_error_m
            )
            annotation.encoding["ground_truth_translation_error_final_m"] = float(
                gt_evaluation.final_translation_error_m
            )
            annotation.encoding["ground_truth_translation_error_goal_m"] = float(goal_m)
            annotation.encoding["ground_truth_initial_meets_goal"] = bool(
                gt_evaluation.initial_translation_error_m <= goal_m
            )
            annotation.encoding["ground_truth_final_meets_goal"] = bool(
                gt_evaluation.final_translation_error_m <= goal_m
            )
            annotation.encoding["ground_truth_translation_error_history_m"] = [
                float(x) for x in gt_evaluation.translation_error_history_m
            ]
        object_hypothesis.annotations.append(annotation)

    def _draw_outlines(
        self,
        image_bgr: np.ndarray,
        expected_mask: np.ndarray,
        detected_mask: np.ndarray,
    ) -> np.ndarray:
        """Overlay expected and detected contours for quick visual comparison."""
        vis = image_bgr.copy()
        expected_contour = self._largest_contour(expected_mask)
        detected_contour = self._largest_contour(detected_mask)
        thickness = max(int(self.descriptor.parameters.contour_thickness), 1)
        if expected_contour is not None:
            cv2.drawContours(vis, [expected_contour], -1, (0, 255, 0), thickness)
        if detected_contour is not None:
            cv2.drawContours(vis, [detected_contour], -1, (0, 0, 255), thickness)
        return vis

    def _build_expected_world(
        self,
        object_center_world: np.ndarray,
        gt_object_model: GroundTruthObjectModel,
    ):
        """Create a temporary SemDT world with only the expected object asserted."""
        world_descriptor = BaseWorldDescriptor(
            root_name="expected_world_root",
            root_prefix="world",
        )
        root = world_descriptor.world.root
        pose_world = np.eye(4, dtype=np.float64)
        pose_world[:3, :3] = gt_object_model.rotation_world
        pose_world[:3, 3] = object_center_world.astype(np.float64)
        expected_object_kwargs = {
            "name": self.descriptor.parameters.expected_object_name,
            "color": Color(
                float(gt_object_model.color.R),
                float(gt_object_model.color.G),
                float(gt_object_model.color.B),
                float(gt_object_model.color.A),
            ),
            "pose": HomogeneousTransformationMatrix(
                data=pose_world,
                reference_frame=root,
            ),
        }
        if gt_object_model.shape_type == "box":
            if gt_object_model.box_scale is None:
                raise ValueError("GT object model for box requires box_scale.")
            expected_object = ObjectSpec(
                box_scale=Scale(
                    float(gt_object_model.box_scale.x),
                    float(gt_object_model.box_scale.y),
                    float(gt_object_model.box_scale.z),
                ),
                **expected_object_kwargs,
            )
        elif gt_object_model.shape_type == "cylinder":
            if (
                gt_object_model.cylinder_width is None
                or gt_object_model.cylinder_height is None
            ):
                raise ValueError(
                    "GT object model for cylinder requires width and height."
                )
            expected_object = ObjectSpec(
                cylinder_width=float(gt_object_model.cylinder_width),
                cylinder_height=float(gt_object_model.cylinder_height),
                **expected_object_kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported GT object model shape '{gt_object_model.shape_type}'."
            )
        world_descriptor.build_objects(root, [expected_object])
        return world_descriptor.world

    @staticmethod
    def _camera_optical_to_link_np() -> np.ndarray:
        """Return homogeneous transform mapping camera optical frame to camera-link frame."""
        return np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _fov_deg_from_cam_info(cam_info: CameraInfo) -> float:
        """Estimate horizontal FOV in degrees from CameraInfo width and fx."""
        width = float(cam_info.width)
        fx = float(cam_info.k[0])
        if fx <= 0.0 or width <= 0.0:
            return 90.0
        return float(np.rad2deg(2.0 * np.arctan(width / (2.0 * fx))))

    @staticmethod
    def _segmentation_to_bgr(expected_world, segmentation: np.ndarray) -> np.ndarray:
        """Colorize segmentation indices into a BGR image using body collision colors."""
        bgr = np.zeros(
            (segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8
        )
        unique_indices = np.unique(segmentation)
        for body_index in unique_indices:
            if body_index < 0:
                continue
            body = expected_world.kinematic_structure[int(body_index)]
            if len(body.collision) == 0:
                continue
            color = body.collision[0].color
            rgb = np.array(
                [
                    int(np.clip(round(color.R * 255.0), 0, 255)),
                    int(np.clip(round(color.G * 255.0), 0, 255)),
                    int(np.clip(round(color.B * 255.0), 0, 255)),
                ],
                dtype=np.uint8,
            )
            bgr[segmentation == body_index] = rgb[::-1]
        return bgr
