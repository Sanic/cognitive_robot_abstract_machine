"""
This module implements the 'SegmentAnythingAnnotator' which allows object segmentation
using Segment Anything Model (SAM) in a zero-shot manner and visualizes the results.

Overview:
---------
The Annotator expects to find ObjectHypotheses containing a region of interest. These
RoIs will be aggregated and processed as a batch.

> Note that this Annotator was not tested with a great number of ObjectHypotheses. It
may be necessary to implement sequential processing logic for use cases with more
than about 32 Object Hypotheses. This number may also depend on the available GPU RAM.

How does it work?
-----------------
The Segment Anything model by Meta AI is prompted using the region of interest of each
detected ObjectHypothesis. The model is trained on 1.1billion segmentation masks and
generalizes well to unseen objects.

Metadata:
---------
Author: Lennart Heinbokel
Created on: 2023-08-13
"""

from __future__ import annotations

import copy
import random

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from ultralytics import SAM

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils import annotator_helper, cv_helper
from robokudo.utils.decorators import timer_decorator
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy.typing as npt


class SegmentAnythingAnnotator(ThreadedAnnotator):
    """
    The SegmentAnythingAnnotator allows object segmentation using SAM (Segment Anything
    Model) and visualizes the results.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Descriptor for the SegmentAnythingAnnotator.
        """

        class Parameters:
            """
            This class contains all parameters that are necessary for the YoloAnnotator.

            Attributes:
                model_name:     Name of the SAM model. Either mobile_sam.pt or TODO
                cmap:           Color map used to visualize the object hypotheses
                filter_fn:      Function that can be used to filter the object hypotheses. Defaults to None.

                                    ```python
                                        def filter_fn(object_hypothesis):
                                            return object_hypothesis.classname == "cereal_box"
                                    ```

                                    This example would only classify object hypotheses that are
                                    cereal boxes. This might be useful if you want to classify
                                    only a subset of the object hypotheses.
                interactive_mode:   If set to True, the annotator will run in interactive mode.

                                        This means that you can either click on the image to
                                        segment an object or you can drag a bounding box around
                                        the object you want to segment.
            """

            def __init__(self) -> None:
                self.model_name: str = "mobile_sam.pt"
                self.interactive_mode: bool = False

                self.cmap: List[Tuple[int, int, int]] = [
                    (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    for _ in range(100)
                ]
                self.filter_fn: Optional[Callable[[ObjectHypothesis], bool]] = None

        parameters = Parameters()

    def __init__(
        self,
        name: str = "SAMAnnotator",
        descriptor: SegmentAnythingAnnotator.Descriptor = Descriptor(),
    ) -> None:
        super().__init__(name)
        self.parameters: SegmentAnythingAnnotator.Descriptor.Parameters = (
            descriptor.parameters
        )
        self.model: SAM = SAM(self.parameters.model_name)

        self.mouse_event: Optional[Dict[str, Any]] = None
        self.name2rgb: Dict[str, Tuple[int, int, int]] = {}
        self.counter: int = 0

        # Book keeping vars for BoundingBox selection mode.
        self.drawing_active: bool = False
        self.start_x: int = -1
        self.start_y: int = -1
        self.last_x: int = -1
        self.last_y: int = -1

    def _reset_drawing_vars(self) -> None:
        self.drawing_active = False
        self.start_x = -1
        self.start_y = -1
        self.last_x = -1
        self.last_y = -1

    def _get_ohs(self) -> List[ObjectHypothesis]:
        object_hypotheses_list = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )

        if self.parameters.filter_fn:
            object_hypotheses_list = [
                oh for oh in object_hypotheses_list if self.filter_fn(oh)
            ]

        return object_hypotheses_list

    @catch_and_raise_to_blackboard
    @timer_decorator
    def compute(self) -> Status:
        """
        Infer masks for all filtered object hypotheses.
        """
        self.camera_intrinsics = copy.deepcopy(
            self.get_cas().get(CASViews.CAMERA_INTRINSIC)
        )

        if self.parameters.interactive_mode:
            return self._interactive_mode()

        object_hypotheses_list = self._get_ohs()
        if len(object_hypotheses_list) == 0:
            self.rk_logger.warning("No bounding boxes found in SAM Annotator")
            return Status.SUCCESS

        oh_bboxes = [oh.roi.roi.get_corner_points() for oh in object_hypotheses_list]

        img_np = self.get_cas().get(CASViews.COLOR_IMAGE)
        model_output = self.model.predict(img_np, bboxes=oh_bboxes, labels=[1])
        predicted_masks = model_output[0].masks.data.cpu().numpy()

        for oh, mask in zip(object_hypotheses_list, predicted_masks):
            oh.roi.mask = mask

        self.visualize(object_hypotheses_list)
        return Status.SUCCESS

    def _get_color_for_classname(self, class_name: str) -> Tuple[int, int, int]:
        if class_name not in self.name2rgb:
            self.name2rgb[class_name] = self.parameters.cmap[self.counter]
            self.counter += 1

        return self.name2rgb[class_name]

    def _interactive_mode(self) -> Status:
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

        event_params = self.mouse_event
        if event_params is None:
            self.visualize(object_hypotheses)
            return Status.SUCCESS

        self.rk_logger.critical("Received mouse event!!! Starting to process")

        if event_params["event"] == cv2.EVENT_LBUTTONDBLCLK:
            self._predict_for_interaction(event_params["x"], event_params["y"])
        elif event_params["event"] == cv2.EVENT_LBUTTONUP:
            self._predict_for_interaction(
                self.start_x, self.start_y, event_params["x"], event_params["y"]
            )
        else:
            self.rk_logger.error(f"Unknown mouse event: {event_params}")

        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

        interactive_ohs = [
            oh
            for oh in object_hypotheses
            if "interactive" in oh.classification.classname
        ]
        self.rk_logger.critical(f"visualizing the results: {interactive_ohs}")
        self.visualize(object_hypotheses)
        self.mouse_event = None
        return Status.SUCCESS

    def resize_mask_to_depth(
        self, mask: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        The mask is potentially created after the input image has been scaled down.

        If that's the case, we have to bring it back to the original resolution.

        :param mask: A binary image.
        :return: The scaled version of mask, according to the COLOR2DEPTH_RATIO
        """
        color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)

        if not color2depth_ratio:
            raise RuntimeError("No Color to Depth Ratio set. Can't continue.")

        if color2depth_ratio == (1, 1):
            return mask
        else:
            c2d_ratio_x = color2depth_ratio[0]
            c2d_ratio_y = color2depth_ratio[1]
            resized_mask = cv2.resize(
                mask,
                None,
                fx=c2d_ratio_x,
                fy=c2d_ratio_y,
                interpolation=cv2.INTER_NEAREST,
            )

        return resized_mask

    def mouse_callback(self, event: int, x: int, y: int, flags, param) -> None:
        """
        Handle mouse events.
        """
        received_click_event = False

        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.rk_logger.info(
                "SAM Annotator mouse event: Double click at ({}, {})".format(x, y)
            )
            self._reset_drawing_vars()
            received_click_event = True
        # The following cases handle the BoundingBox selection mode.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.rk_logger.info(
                "SAM Annotator mouse event: Left button down at ({}, {})".format(x, y)
            )
            self.drawing_active = True
            self.start_x = self.last_x = x
            self.start_y = self.last_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            self.rk_logger.info(
                "SAM Annotator mouse event: Mouse move from ({}, {}) to ({}, {})".format(
                    x, y, self.start_x, self.start_y
                )
            )
            if self.drawing_active:
                self.last_x = x
                self.last_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.rk_logger.info(
                "SAM Annotator mouse event: Left button up at ({}, {})".format(x, y)
            )

            if self.drawing_active:
                self.drawing_active = False

                if self.start_x != x and self.start_y != y:
                    self.last_x = x
                    self.last_y = y
                    received_click_event = True

        if received_click_event:
            self.mouse_event = {
                "event": event,
                "x": x,
                "y": y,
                "flags": flags,
                "param": param,
            }

    def _predict_for_interaction(
        self, x: int, y: int, last_x: Optional[int] = None, last_y: Optional[int] = None
    ) -> None:
        """
        Predict mask for a single point or a BoundingBox defined by (x,y,last_x,last_y).
        """
        img_np = cv2.cvtColor(
            self.get_cas().get(CASViews.COLOR_IMAGE), cv2.COLOR_BGR2RGB
        )
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)

        if last_x and last_y:
            self.rk_logger.debug("Processing box prompt")
            predicted_masks = (
                self.model.predict(img_np, bboxes=[(x, y, last_x, last_y)], labels=[1])[
                    0
                ]
                .masks.data.cpu()
                .numpy()
            )
        else:
            self.rk_logger.debug("Processing point prompt")
            predicted_masks = (
                self.model.predict(img_np, points=[x, y], labels=[1])[0]
                .masks.data.cpu()
                .numpy()
            )

        try:
            mask = predicted_masks[0].astype(np.uint8)
        except IndexError:
            self.rk_logger.error(f"No mask predicted for point ({x}, {y})")
            return
        # resized_mask = mask.astype(np.uint8)
        resized_mask = self.resize_mask_to_depth(mask)

        # Calculate the BB dimensions from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)

        # Generate the Object Hypothesis
        oh = ObjectHypothesis()
        oh.classification = Classification()
        oh.classification.classname = "interactive_click"
        oh.roi.roi.pos.x = bbox_x
        oh.roi.roi.pos.y = bbox_y
        oh.roi.roi.width = bbox_w
        oh.roi.roi.height = bbox_h
        oh.roi.mask = mask

        # POINT GENERATION
        # TODO REFACTOR THIS INTO A SEPERATE ANNOTATOR
        depth_masked = copy.deepcopy(depth)
        depth_masked = np.where(resized_mask == 1, depth_masked, 0)

        resized_color = None
        try:
            resized_color = cv_helper.get_scaled_color_image_for_depth_image(
                self.get_cas(), img_np
            )
            annotator_helper.scale_camera_intrinsics(self)
        except RuntimeError as e:  # pylint: disable=invalid-name
            self.rk_logger.error(
                f"No color to depth ratio set by your camera driver! Can't preprocess: {e}"
            )

        o3d_color = o3d.geometry.Image(resized_color)
        o3d_depth = o3d.geometry.Image(depth_masked)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth
        )
        # Here we used the (potentially) scaled cam intrinsics to create the object cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_intrinsics
        )
        oh.points = pcd

        x = [
            {"name": "Cloud", "geometry": self.get_cas().get(CASViews.CLOUD)},
            {"name": "Object", "geometry": pcd},
        ]

        self.get_annotator_output_struct().set_geometries(x)
        # POINT GENERATION END

        self.get_cas().annotations.append(oh)

    def visualize(self, object_hypotheses: List[ObjectHypothesis]) -> None:
        """
        Visualize the object hypotheses' masks.
        """
        # self.rk_logger.critical("setting visualization image")
        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)

        # Check if drawing is active and draw rectangle based on the current data
        if self.drawing_active:
            visualization_img = cv2.rectangle(
                visualization_img,
                (int(self.start_x), int(self.start_y)),
                (int(self.last_x), int(self.last_y)),
                (255, 0, 255),
                1,
            )

        for oh in object_hypotheses:
            roi = oh.roi
            x1, y1, x2, y2 = map(int, roi.roi.get_corner_points())

            visualization_img = cv2.rectangle(
                visualization_img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            visualization_img[roi.mask == 1] = self._get_color_for_classname(
                oh.classification.classname
            )

            text = f"{oh.classification.classname}"
            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(
                visualization_img,
                text,
                (x1, (y1 - 5)),
                font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )

        self.get_annotator_output_struct().set_image(visualization_img)

    # def _predict_bbox(self, x1, y1, x2, y2):
    #     """Predict mask for a bounding box"""
    #     img_np = self.get_cas().get(CASViews.COLOR_IMAGE)
    #     predicted_masks = (
    #         self.model.predict(img_np, bboxes=[x1, y1, x2, y2], labels=[1])[0]
    #         .masks.data.cpu()
    #         .numpy()
    #     )
    #
    #     try:
    #         mask = predicted_masks[0]
    #     except IndexError:
    #         self.rk_logger.error(f"No mask predicted for bounding box ({x1}, {y1}, {x2}, {y2})")
    #         return
    #
    #     oh = robokudo.types.scene.ObjectHypothesis()
    #     oh.classification.classname = "interactive_bbox"
    #     oh.bbox = [x1, y1, x2, y2]
    #     oh.roi.roi.pos.x = x1
    #     oh.roi.roi.pos.y = y1
    #     oh.roi.roi.width = x2 - x1
    #     oh.roi.roi.height = y2 - y1
    #     oh.roi.mask = mask
