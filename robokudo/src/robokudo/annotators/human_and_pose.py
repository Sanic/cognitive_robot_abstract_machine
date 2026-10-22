"""
Human detection with body keypoint estimation.

Detects humans with a YOLO pose model and annotates each one as a human hypothesis that
carries a person classification and its 2D body keypoints. With depth, each hypothesis
also gets the point cloud of its bounding box and, optionally, one point cloud per
clearly visible keypoint.

.. note::
   The model needs the packages of RoboKudo's optional ``dev`` extra.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any
from ultralytics import YOLO

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.annotators.exceptions import NotAPoseModelException
from robokudo.cas import CASViews
from robokudo.defs import PACKAGE_NAME
from robokudo.types.annotation import Classification, CloudAnnotation
from robokudo.types.cv import Rect
from robokudo.types.human import (
    BodyKeypoint,
    BodyKeypointCloud,
    KeypointAnnotation,
    KeypointKind,
)
from robokudo.types.scene import HumanHypothesis
from robokudo.utils.decorators import record_time
from robokudo.utils.image_region_projection_3d import (
    ImageCoordinates,
    ImageRegionProjection3D,
)
from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.o3d_helper import concatenate_clouds

if TYPE_CHECKING:
    import numpy.typing as npt
    from ultralytics.engine.results import Results


# %% body keypoints


class PoseKeypoint(IntEnum):
    """
    Index of each body keypoint in the keypoints a YOLO pose model reports.
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# %% annotator


class HumanAndPoseAnnotator(ThreadedAnnotator):
    """
    Detects humans with a YOLO pose model and annotates them with their keypoints.
    """

    class Descriptor(ThreadedAnnotator.Descriptor):
        """
        Descriptor for the HumanAndPoseAnnotator.
        """

        class Parameters:
            """
            Parameters for the HumanAndPoseAnnotator.
            """

            def __init__(self) -> None:
                self.models_directory: Path = (
                    ModuleLoader.get_module_path(PACKAGE_NAME) / "models"
                )
                """
                Directory the pose model weights are loaded from.

                Created during setup if it does not exist.
                """
                self.pretrained_model: str = "yolov8n-pose.pt"
                """
                File name of the YOLO pose model weights inside
                :attr:`models_directory`.

                A model ultralytics provides is downloaded there on first use; custom
                trained weights must already be there.

                .. note::
                   The model must report body keypoints, otherwise
                   :meth:`HumanAndPoseAnnotator.compute` raises
                   :class:`~robokudo.annotators.exceptions.NotAPoseModelException`.
                """
                self.person_class_name: str = "person"
                """
                Class name the human hypotheses are classified as.
                """
                self.confidence_threshold: float = 0.7
                """
                Minimum confidence of a human detection to be annotated.
                """
                self.non_maximum_suppression_overlap: float = 0.5
                """
                Intersection over union above which the model merges overlapping
                detections.
                """
                self.global_with_depth: bool = False
                """
                Whether to build point clouds from the depth image.
                """
                self.shrink_bounding_box: float | None = None
                """
                Fraction in [0, 1] of its width and height the bounding box of each
                human is shrunk by around its center, or None to keep it.

                Shrinking may interfere with activity recognition.
                """
                self.shrink_as_annotation: bool = False
                """
                Whether, with depth, the shrunk box only yields a cloud annotation
                instead of replacing the region of the human hypothesis.
                """
                self.generate_clouds_for_keypoints: bool = False
                """
                Whether, with depth, to build a point cloud around every keypoint.

                .. warning::
                   This is computationally expensive.
                """
                self.keypoint_confidence_threshold_for_clouds: float = 0.6
                """
                Minimum confidence of a keypoint to get a non-empty point cloud.
                """
                self.keypoint_masking_radius: int = 6
                """
                Radius in depth image pixels around each keypoint its cloud is built
                from.
                """
                self.warmup_annotator: bool = True
                """
                Whether setup runs the model once, so that the first real detection is
                not slowed down by its initialization.
                """
                self.warmup_image_shape: tuple[int, int, int] = (512, 512, 3)
                """
                Shape (height, width, channels) of the blank image the model is warmed
                up on.
                """
                self.depth_truncate: float = 3.5
                """
                Distance in meters beyond which depth measurements are left out of the
                box clouds, removing background from the human detection.
                """

    def __init__(
        self,
        name: str = "HumanAndPoseAnnotator",
        descriptor: HumanAndPoseAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)
        self.model: YOLO | None = None
        """
        The YOLO pose model, loaded during setup.
        """

    def setup(self, **kwargs: Any) -> bool:
        """
        Extends :meth:`ThreadedAnnotator.setup` by loading the pose model from
        :attr:`Descriptor.Parameters.models_directory`, creating that directory if it
        does not exist, and, if configured, warming the model up.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        parameters.models_directory.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(
            str(parameters.models_directory / parameters.pretrained_model)
        )
        if parameters.warmup_annotator:
            self._predict(np.zeros(parameters.warmup_image_shape, dtype=np.uint8))
        return super().setup(**kwargs)

    def _predict(self, image: npt.NDArray[np.uint8]) -> Results:
        """
        Run the pose model on an image with the configured confidence threshold and non-
        maximum suppression overlap.

        :param image: The color image to detect humans in.
        :return: The humans the model detects in the image, with their bounding boxes,
            confidences and keypoints.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        return self.model.predict(
            image,
            conf=parameters.confidence_threshold,
            iou=parameters.non_maximum_suppression_overlap,
            save=False,
            verbose=False,
        )[0]

    @record_time
    def compute(self) -> Status:
        """
        Detect humans in the color image of the CAS and add one human hypothesis with
        its keypoints per detection to the CAS.

        :return: :attr:`Status.SUCCESS` once the hypotheses are added.
        :raises NotAPoseModelException: If the model reports no keypoints for a
            detection.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        region_projection = (
            ImageRegionProjection3D.from_cas(self.get_cas())
            if parameters.global_with_depth
            else None
        )
        result = self._predict(self.get_cas().get(CASViews.COLOR_IMAGE))

        hypotheses = [
            self._hypothesis(result, index, region_projection)
            for index in range(len(result.boxes))
        ]
        self.get_cas().annotations.extend(hypotheses)

        if parameters.global_with_visualization:
            self._visualize(result, hypotheses)
        return Status.SUCCESS

    def _hypothesis(
        self,
        result: Results,
        index: int,
        region_projection: ImageRegionProjection3D | None,
    ) -> HumanHypothesis:
        """
        Turn one detection into a human hypothesis classified as a person and annotated
        with its 2D keypoints, and, with depth, with its point clouds.

        :param result: The detections of the pose model on the color image.
        :param index: Index of the detection in the result, which becomes the id of the
            hypothesis.
        :param region_projection: The depth measurements to build point clouds from, or
            None to build none.
        :return: The human hypothesis of the detection.
        :raises NotAPoseModelException: If the model reports no keypoints.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        if result.keypoints is None:
            raise NotAPoseModelException(
                annotator_name=self.name, model=parameters.pretrained_model
            )
        box = Rect.from_corner_points(*result.boxes.xyxy[index].cpu().numpy())
        shrunk_box = (
            box if parameters.shrink_bounding_box is None else self._shrink_box(box)
        )
        keeps_full_box = (
            region_projection is not None and parameters.shrink_as_annotation
        )

        hypothesis = HumanHypothesis()
        hypothesis.id = index
        hypothesis.source = self.name
        hypothesis.roi.roi = box if keeps_full_box else shrunk_box
        hypothesis.annotations.append(
            Classification(
                source=self.name,
                classname=parameters.person_class_name,
                confidence=float(result.boxes.conf[index]),
            )
        )

        keypoints = [
            BodyKeypoint(x=x, y=y, confidence=confidence)
            for x, y, confidence in result.keypoints.data[index].cpu().numpy().tolist()
        ]
        hypothesis.annotations.append(
            KeypointAnnotation(
                source=self.name,
                keypoints=keypoints,
                kind=KeypointKind.IMAGE_POSITION,
            )
        )
        if region_projection is None:
            return hypothesis

        hypothesis.points = region_projection.project_box(
            hypothesis.roi.roi, depth_truncate=parameters.depth_truncate
        )
        if parameters.shrink_bounding_box is not None and keeps_full_box:
            hypothesis.annotations.append(
                CloudAnnotation(
                    source=self.name,
                    points=region_projection.project_box(
                        shrunk_box, depth_truncate=parameters.depth_truncate
                    ),
                )
            )
        if parameters.generate_clouds_for_keypoints:
            hypothesis.annotations.append(
                KeypointAnnotation(
                    source=self.name,
                    keypoints=self._keypoint_clouds(keypoints, region_projection),
                    kind=KeypointKind.POINT_CLOUD,
                )
            )
        return hypothesis

    def _shrink_box(self, box: Rect) -> Rect:
        """
        Shrink a box around its center by
        :attr:`Descriptor.Parameters.shrink_bounding_box` of its width and height.

        :param box: The box in color image pixels.
        :return: The shrunk box in color image pixels.
        """
        shrink_fraction = self.descriptor.parameters.shrink_bounding_box
        horizontal_margin = box.width * shrink_fraction / 2
        vertical_margin = box.height * shrink_fraction / 2
        return Rect.from_corner_points(
            x1=box.pos.x + horizontal_margin,
            y1=box.pos.y + vertical_margin,
            x2=box.pos.x + box.width - horizontal_margin,
            y2=box.pos.y + box.height - vertical_margin,
        )

    def _keypoint_clouds(
        self, keypoints: list[BodyKeypoint], region_projection: ImageRegionProjection3D
    ) -> list[BodyKeypointCloud]:
        """
        Build a point cloud from the disc of
        :attr:`Descriptor.Parameters.keypoint_masking_radius` around each keypoint.

        :param keypoints: The body keypoints of one human.
        :param region_projection: The depth measurements to build the clouds from.
        :return: The cloud around each keypoint, empty for keypoints below
            :attr:`Descriptor.Parameters.keypoint_confidence_threshold_for_clouds`.
        """
        parameters: HumanAndPoseAnnotator.Descriptor.Parameters = (
            self.descriptor.parameters
        )
        return [
            BodyKeypointCloud(
                keypoint=keypoint,
                points=(
                    region_projection.project_disc(
                        self.get_cas()
                        .get(CASViews.COLOR_TO_DEPTH_RATIO)
                        .depth_pixel(keypoint.x, keypoint.y),
                        parameters.keypoint_masking_radius,
                        coordinates=ImageCoordinates.DEPTH,
                    )
                    if keypoint.confidence
                    >= parameters.keypoint_confidence_threshold_for_clouds
                    else o3d.geometry.PointCloud()
                ),
            )
            for keypoint in keypoints
        ]

    def _visualize(self, result: Results, hypotheses: list[HumanHypothesis]) -> None:
        """
        Output the color image with the detections drawn by the model and, if any
        hypothesis has one, the point clouds of the hypotheses merged into one.

        :param result: The detections of the pose model on the color image.
        :param hypotheses: The human hypotheses made from the detections.
        """
        self.get_annotator_output_struct().set_image(result.plot())
        clouds = [
            hypothesis.points
            for hypothesis in hypotheses
            if hypothesis.points is not None
        ]
        if len(clouds) > 0:
            self.get_annotator_output_struct().set_geometries(
                concatenate_clouds(clouds)
            )
