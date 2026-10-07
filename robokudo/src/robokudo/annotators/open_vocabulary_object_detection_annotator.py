import random
from timeit import default_timer

import cv2
import numpy as np
import py_trees
import requests
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from ultralytics import SAM

import robokudo.annotators.core
import robokudo.types
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
import robokudo.types.annotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


def box_text_function(object_hypothesis: robokudo.types.scene.ObjectHypothesis) -> str:
    return ""


def get_box_text(oh):
    max_conf = -1
    best_classification = None

    for oh_anno in oh.annotations:
        if isinstance(oh_anno, robokudo.types.annotation.Classification):
            if oh_anno.confidence > max_conf:
                max_conf = oh_anno.confidence
                best_classification = oh_anno

    if best_classification is None:
        return f"ROI-{oh.id}"
    else:
        return f"{oh.id}: {best_classification.classname}, {best_classification.confidence:.2f}"


class OpenVocabularyObjectDetectionAnnotator(
    robokudo.annotators.core.ThreadedAnnotator
):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.classes = ["Cat", "Dog"]
                # This refers to 'transformers' terminology
                self.detection_model = "google/owlv2-base-patch16-ensemble"
                self.detection_processor = "google/owlv2-base-patch16-ensemble"
                self.detection_threshold = 0.2

                # Use SAM in precision mode to generate masks
                self.sam_model = "mobile_sam.pt"
                self.precision_mode = False
                # Some object detectors might undersegment the object.
                # Applying SAM can sometimes fix this problem. If this is set to true,
                # use the BB as suggested by SAM
                self.precision_mode_can_fix_boundingbox = False

        parameters = Parameters()

    def __init__(
        self,
        name="OpenVocabObjectDetectionAnnotator",
        descriptor=Descriptor(),
    ) -> None:
        super(OpenVocabularyObjectDetectionAnnotator, self).__init__(name, descriptor)

        self.classes = self.descriptor.parameters.classes

        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.descriptor.parameters.detection_model
        )
        self.processor = Owlv2Processor.from_pretrained(
            self.descriptor.parameters.detection_processor
        )

        self.id2name = {
            str(i): name for i, name in enumerate(self.descriptor.parameters.classes)
        }
        self.id2rgb = {
            i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for i, _ in enumerate(self.descriptor.parameters.classes)
        }

        if self.descriptor.parameters.precision_mode:
            self.sam = SAM(self.descriptor.parameters.sam_model)

    @catch_and_raise_to_blackboard
    def compute(self):
        start_timer = default_timer()

        img = self.get_cas().get(CASViews.COLOR_IMAGE)
        img = Image.fromarray(img)

        object_hypotheses = []

        target_sizes = torch.Tensor([img.size[::-1]])

        inputs = self.processor(
            text=self.classes, images=img, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.descriptor.parameters.detection_threshold,
        )
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = self.classes
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            confidence = round(score.item(), 3)

            print(
                f"Detected {text[label]} with confidence {confidence} at location {box}"
            )
            xmin, ymin, xmax, ymax = box
            oh = ObjectHypothesis()
            oh.roi.roi.pos.x = int(xmin)
            oh.roi.roi.pos.y = int(ymin)
            oh.roi.roi.width = int(xmax - xmin)
            oh.roi.roi.height = int(ymax - ymin)

            if self.descriptor.parameters.precision_mode:
                masks = (
                    self.sam.predict(img, bboxes=[box], labels=[1])[0]
                    .masks.data.cpu()
                    .numpy()
                )
                mask = masks[0].astype(np.uint8)
                oh.roi.mask = self.resize_mask(
                    mask
                )  # scale mask back to original if necessary
                oh.roi.mask *= 255  # SAM outputs masks with 1. Scale to 255.

                if self.descriptor.parameters.precision_mode_can_fix_boundingbox:
                    contours, _ = cv2.findContours(
                        oh.roi.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        self.rk_logger.warning(f"Attempting to fix {oh.roi.roi}")
                        x, y, w, h = cv2.boundingRect(
                            contours[0]
                        )  # Largest/first contour
                        oh.roi.roi.pos.x = int(x)
                        oh.roi.roi.pos.y = int(y)
                        oh.roi.roi.width = int(w)
                        oh.roi.roi.height = int(h)
                        self.rk_logger.warning(f"AFTER FIX: {oh.roi.roi}")
                        # bbox = (y, x, y + h, x + w)  # Convert to (y1, x1, y2, x2)

                oh.roi.mask = robokudo.utils.cv_helper.crop_image(
                    oh.roi.mask,
                    (oh.roi.roi.pos.x, oh.roi.roi.pos.y),
                    (oh.roi.roi.width, oh.roi.roi.height),
                )

            classification_annotation = robokudo.types.annotation.Classification()
            classification_annotation.classname = text[label]
            classification_annotation.source = self.get_class_name()
            classification_annotation.confidence = confidence
            oh.annotations.append(classification_annotation)

            object_hypotheses.append(oh)

        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
            visualization_img, object_hypotheses, get_box_text
        )

        cmap = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [122, 0, 122], [90, 90, 0]]
        )

        if self.descriptor.parameters.precision_mode:
            for oh_idx, oh in enumerate(object_hypotheses):
                x1 = oh.roi.roi.pos.x
                y1 = oh.roi.roi.pos.y
                full_mask = np.zeros(
                    (visualization_img.shape[0], visualization_img.shape[1]),
                    dtype="uint8",
                )
                full_mask[
                    y1 : y1 + oh.roi.mask.shape[0], x1 : x1 + oh.roi.mask.shape[1]
                ] = oh.roi.mask
                visualization_img[full_mask == 255] = cmap[oh_idx % cmap.shape[0]]

        self.get_cas().annotations.extend(object_hypotheses)

        self.get_annotator_output_struct().set_image(visualization_img)
        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def resize_mask(self, mask):
        """
        Resize the mask according to the color-to-depth ratio.
        """
        if not self.descriptor.parameters.global_with_depth:
            return mask
        color2depth_ratio = self.get_cas().get(CASViews.COLOR2DEPTH_RATIO)
        if not color2depth_ratio:
            raise RuntimeError("No Color to Depth Ratio set. Can't continue.")
        if color2depth_ratio == (1, 1):
            return mask
        else:
            c2d_ratio_x, c2d_ratio_y = color2depth_ratio
            resized_mask = cv2.resize(
                mask,
                None,
                fx=1 / c2d_ratio_x,
                fy=1 / c2d_ratio_y,
                interpolation=cv2.INTER_NEAREST,
            )
            return resized_mask
