"""Render a simple expected world state from current camera perspective."""

from __future__ import annotations

from dataclasses import dataclass
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
from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale, Color


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
                #: Expected object size along x in meters.
                self.scale_x: float = 0.08
                #: Expected object size along y in meters.
                self.scale_y: float = 0.08
                #: Expected object size along z in meters.
                self.scale_z: float = 0.14
                #: Expected object color red channel in [0, 1].
                self.color_r: float = 0.22
                #: Expected object color green channel in [0, 1].
                self.color_g: float = 0.37
                #: Expected object color blue channel in [0, 1].
                self.color_b: float = 0.82
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
                self.refinement_max_iterations: int = 10
                #: Finite-difference perturbation size for Jacobian estimation (m).
                self.refinement_jacobian_delta_m: float = 0.01
                #: Maximum translation step applied per refinement iteration (m).
                self.refinement_max_step_m: float = 0.03
                #: Convergence threshold for centroid error in pixels.
                self.refinement_convergence_pixel_error: float = 2.0
                #: Convergence threshold for score change between iterations.
                self.refinement_convergence_score_delta: float = 0.001
                #: Enable writing visualization and mask images for each run.
                self.save_run_images: bool = True
                #: Directory where run and per-iteration images are written.
                self.run_image_output_dir: str = "/tmp"

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
            cam_info=cam_info,
            cam_to_world_optical=cam_to_world_optical,
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
        random_direction = np.random.normal(size=3).astype(np.float64)
        direction_norm = float(np.linalg.norm(random_direction))
        if direction_norm < 1e-9:
            random_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            direction_norm = 1.0
        random_direction /= direction_norm
        random_radius = float(np.random.uniform(0.0, magnitude))
        offset = random_direction * random_radius
        return center_world + offset

    def _refine_expected_pose_translation(
        self,
        initial_center_world: np.ndarray,
        detected_mask: np.ndarray,
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
        convergence_score_delta = float(
            self.descriptor.parameters.refinement_convergence_score_delta
        )

        current_center = initial_center_world.astype(np.float64).copy()
        previous_score = None
        best_result = None
        executed_iterations = 0
        score_history: list[float] = []
        center_history_world: list[list[float]] = []
        pixel_error_history: list[float] = []
        stop_reason = "max_iterations_reached"

        for iteration in range(max_iterations):
            executed_iterations = iteration + 1
            render_bgr, expected_mask = self._render_expected_mask_for_center(
                object_center_world=current_center,
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
                best_result is None
                or current_result.outline_match.combined_score
                > best_result.outline_match.combined_score
            ):
                best_result = current_result

            score_delta = (
                abs(outline_match.combined_score - previous_score)
                if previous_score is not None
                else float("inf")
            )
            previous_score = outline_match.combined_score
            if pixel_error <= convergence_pixel_error:
                current_result.converged = True
                current_result.stop_reason = (
                    "pixel_error_converged "
                    f"(px_err={pixel_error:.2f} <= {convergence_pixel_error:.2f})"
                )
                current_result.score_history = list(score_history)
                current_result.center_history_world = [
                    list(xyz) for xyz in center_history_world
                ]
                current_result.pixel_error_history = list(pixel_error_history)
                return current_result
            if score_delta <= convergence_score_delta and iteration > 0:
                if best_result is not None:
                    best_result.converged = True
                    best_result.stop_reason = (
                        "score_delta_converged "
                        f"(delta={score_delta:.6f} <= {convergence_score_delta:.6f})"
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
                    f"(delta={score_delta:.6f} <= {convergence_score_delta:.6f})"
                )
                current_result.score_history = list(score_history)
                current_result.center_history_world = [
                    list(xyz) for xyz in center_history_world
                ]
                current_result.pixel_error_history = list(pixel_error_history)
                return current_result

            if centroid_shift_px is None:
                stop_reason = "centroid_shift_unavailable"
                break

            shift_world = self._estimate_translation_shift_world(
                current_center_world=current_center,
                expected_mask=expected_mask,
                detected_mask=detected_mask,
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
        self, initial_center_world: np.ndarray, refinement: RefinementResult
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

    def _render_expected_mask_for_center(
        self,
        object_center_world: np.ndarray,
        cam_info: CameraInfo,
        cam_to_world_optical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render expected-world BGR and binary mask for a concrete object translation."""
        expected_world = self._build_expected_world(object_center_world)
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

    def _build_expected_world(self, object_center_world: np.ndarray):
        """Create a temporary SemDT world with only the expected object asserted."""
        world_descriptor = BaseWorldDescriptor(
            root_name="expected_world_root",
            root_prefix="world",
        )
        root = world_descriptor.world.root
        expected_object = ObjectSpec(
            name=self.descriptor.parameters.expected_object_name,
            box_scale=Scale(
                float(self.descriptor.parameters.scale_x),
                float(self.descriptor.parameters.scale_y),
                float(self.descriptor.parameters.scale_z),
            ),
            color=Color(
                float(self.descriptor.parameters.color_r),
                float(self.descriptor.parameters.color_g),
                float(self.descriptor.parameters.color_b),
                1.0,
            ),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(object_center_world[0]),
                y=float(object_center_world[1]),
                z=float(object_center_world[2]),
                reference_frame=root,
            ),
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
