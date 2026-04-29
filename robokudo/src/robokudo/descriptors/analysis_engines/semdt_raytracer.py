"""Analysis engine for simulated RGB-D input from SemDT RayTracer.

This pipeline mirrors the standard tabletop segmentation flow but uses the
`semdt_raytracer` camera descriptor, which renders color/depth images from a
world descriptor instead of reading a physical camera stream.
"""

import numpy as np

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.expected_state_renderer import (
    ExpectedStateRendererAnnotator,
)
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.lambda_function import LambdaFunctionAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.cas import CASViews
from robokudo.descriptors import CrDescriptorFactory
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis


def classify_boxes_by_color(annotator: LambdaFunctionAnnotator) -> None:
    """Attach a simple color-based Classification annotation to each ObjectHypothesis."""
    cas = annotator.get_cas()
    color_image = cas.get(CASViews.COLOR_IMAGE)
    object_hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
    if color_image is None or object_hypotheses is None:
        return

    for object_hypothesis in object_hypotheses:
        mean_bgr = _mean_bgr_for_object(color_image, object_hypothesis)
        if mean_bgr is None:
            continue

        b, g, r = mean_bgr
        if b > r + 15.0 and b > g:
            class_name = "box_blue"
            class_id = 2
        elif r > b + 15.0 and r > g:
            class_name = "box_red"
            class_id = 1
        else:
            class_name = "box_unknown"
            class_id = 0

        confidence = abs(float(r) - float(b)) / max(float(r + b), 1.0)
        object_hypothesis.annotations = [
            annotation
            for annotation in object_hypothesis.annotations
            if not (
                isinstance(annotation, Classification)
                and annotation.source == "SemDTRayTracerColorClassifier"
            )
        ]

        classification = Classification(
            classification_type="CLASS",
            classname=class_name,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            class_id=class_id,
        )
        classification.source = "SemDTRayTracerColorClassifier"
        object_hypothesis.annotations.append(classification)


def _mean_bgr_for_object(
    color_image: np.ndarray, object_hypothesis: ObjectHypothesis
) -> np.ndarray | None:
    roi = object_hypothesis.roi.roi
    x = max(int(roi.pos.x), 0)
    y = max(int(roi.pos.y), 0)
    w = max(int(roi.width), 0)
    h = max(int(roi.height), 0)
    if w == 0 or h == 0:
        return None

    x2 = min(x + w, color_image.shape[1])
    y2 = min(y + h, color_image.shape[0])
    if x2 <= x or y2 <= y:
        return None

    patch = color_image[y:y2, x:x2]
    if patch.size == 0:
        return None

    if object_hypothesis.roi.mask is None:
        return patch.reshape(-1, 3).mean(axis=0)

    mask = object_hypothesis.roi.mask
    if mask.shape[0] != patch.shape[0] or mask.shape[1] != patch.shape[1]:
        return patch.reshape(-1, 3).mean(axis=0)

    valid = mask > 0
    if not np.any(valid):
        return patch.reshape(-1, 3).mean(axis=0)
    return patch[valid].mean(axis=0)


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semdt_raytracer"

    def implementation(self) -> Pipeline:
        raytracer_config = CrDescriptorFactory.create_descriptor("semdt_raytracer")
        color_classifier_descriptor = LambdaFunctionAnnotator.Descriptor()
        color_classifier_descriptor.parameters.func = classify_boxes_by_color

        plane_desc = PlaneAnnotator.Descriptor()
        plane_desc.parameters.distance_threshold = 0.01

        seq = Pipeline("SemDTRayTracerPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_desc),
                PointCloudClusterExtractor(),
                ClusterPoseBBAnnotator(),
                LambdaFunctionAnnotator(
                    name="RayTracerColorClassifier",
                    descriptor=color_classifier_descriptor,
                ),
                ExpectedStateRendererAnnotator(),
            ]
        )
        return seq
