"""
Image-based object cluster extraction.

This module provides functionality for extracting object clusters from color images using HSV color segmentation.
The main class :class:`ImageClusterExtractor` implements color-based segmentation and contour detection to identify
object clusters in RGB images.

Key features:

* HSV color space thresholding
* Contour detection and filtering
* 3D point cloud generation from depth data
* ROI and mask generation
* Query-based color parameter adjustment
* Visualization of detected clusters
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from timeit import default_timer

import cv2
import numpy as np
import open3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.exceptions import ImageContourMissing
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy.typing as npt


# %% Cluster extraction result
@dataclass
class ImageClusters:
    """Object hypotheses and visualization data extracted from an image."""

    object_hypotheses: list[ObjectHypothesis] = field(default_factory=list)
    """Objects whose point clouds meet the minimum size."""

    point_clouds: list[open3d.geometry.PointCloud] = field(default_factory=list)
    """Point clouds for all selected contours."""

    depth_mask: npt.NDArray | None = None
    """Color visualization of the last selected contour's depth mask."""


# %% Image cluster annotator


class ImageClusterExtractor(BaseAnnotator):
    """
    Extract object clusters from images using color segmentation.

    This annotator performs the following steps:

    * Converts RGB image to HSV color space
    * Applies HSV thresholding based on configured parameters
    * Detects and filters contours based on size
    * Generates point clouds from depth data for each contour
    * Creates ObjectHypothesis annotations with ROIs and masks
    * Provides visualization of detected clusters

    The HSV thresholds can be adjusted dynamically based on color queries.
    """

    class ViewMode:
        """
        Visualization modes for the annotator output.
        """

        masked_object: int = 1
        """
        Show masked RGB image of detected objects.
        """

        depth_mask: int = 2
        """
        Show depth mask of detected objects.
        """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for ImageClusterExtractor.

        Parameters:

        * HSV thresholding ranges
        * Contour filtering parameters
        * Point cloud generation settings
        * Color name to HSV range mappings
        * Outlier removal parameters
        """

        class Parameters:
            """
            Parameter class containing all configurable settings.
            """

            def __init__(self) -> None:
                """
                Initialize default parameter values.
                """
                self.hsv_min: tuple[int, int, int] = (150, 130, 85)
                self.hsv_max: tuple[int, int, int] = (200, 255, 255)
                self.erosion_iterations: int = 2

                # This parameter controls the filtering of the initial list of contours.
                # It is used to avoid very small contours when calculating 3d points etc.
                self.contour_min_size: int = 1000

                self.color_name_to_hsv_range: dict[
                    str, dict[str, tuple[int, int, int]]
                ] = dict()
                self.color_name_to_hsv_range["blue"] = {
                    "hsv_min": (150, 130, 85),
                    "hsv_max": (200, 255, 255),
                }
                # Red sits at the hue wheel's 0/255 seam, so its hue wraps around rather
                # than falling in one contiguous range: hsv_min's hue exceeding hsv_max's
                # signals that to _threshold_hsv, which then matches [0, hsv_max hue] OR
                # [hsv_min hue, 255] instead of a single contiguous range.
                self.color_name_to_hsv_range["red"] = {
                    "hsv_min": (245, 150, 95),
                    "hsv_max": (15, 255, 255),
                }

                self.outlier_removal: bool = True
                self.outlier_removal_nb_neighbors: int = 20
                self.outlier_removal_std_ratio: float = 2.0
                self.num_of_objects: int = 2

                # The minimal amount of 3D points of the object's pointcloud
                # This check is applied AFTER self.contour_min_size
                self.min_points_threshold: int = 62

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ImageClusterExtractor",
        descriptor: ImageClusterExtractor.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.color: npt.NDArray | None = None
        self.depth: npt.NDArray | None = None
        self.query = None
        self.camera_intrinsics = None

        self.display_mode = self.ViewMode.masked_object

    def adjust_hsv_threshold_to_query(self) -> None:
        """
        Adjust HSV thresholds based on color query.

        Checks for a color query in the CAS and updates the HSV thresholding parameters
        if a matching color is found in the color_name_to_hsv_range mapping.
        """
        if CASViews.QUERY not in self.get_cas().views:
            return
        self.query = self.get_cas().get(CASViews.QUERY)

        if not self.query:
            return

        if len(self.query.obj.color) == 0:
            return

        color = self.query.obj.color[0]
        if color not in self.descriptor.parameters.color_name_to_hsv_range:
            return

        self.descriptor.parameters.hsv_min = (
            self.descriptor.parameters.color_name_to_hsv_range[color]["hsv_min"]
        )
        self.descriptor.parameters.hsv_max = (
            self.descriptor.parameters.color_name_to_hsv_range[color]["hsv_max"]
        )

    def _threshold_hsv(
        self,
        hsv_min: tuple[int, int, int],
        hsv_max: tuple[int, int, int],
    ) -> npt.NDArray:
        """
        Threshold ``self.hsv`` to the given bounds, wrapping the hue channel around the
        0/255 seam when ``hsv_min``'s hue exceeds ``hsv_max``'s.

        True red sits exactly at that seam, so a single contiguous hue range only ever
        catches one side of it (confirmed live: a plain ``[215,255]`` range matched 124
        of 101760 pixels on a red object; wrapping recovered 1748 by also matching the
        pixels that wrapped to just above 0).

        :param hsv_min: Lower HSV bound.
        :param hsv_max: Upper HSV bound. A hue lower than ``hsv_min``'s signals
            wraparound.
        :return: Binary mask of pixels within the (possibly wrapped) bounds.
        """
        if hsv_min[0] <= hsv_max[0]:
            return cv2.inRange(self.hsv, hsv_min, hsv_max)

        low = cv2.inRange(
            self.hsv,
            (0, hsv_min[1], hsv_min[2]),
            (hsv_max[0], hsv_max[1], hsv_max[2]),
        )
        high = cv2.inRange(
            self.hsv,
            (hsv_min[0], hsv_min[1], hsv_min[2]),
            (255, hsv_max[1], hsv_max[2]),
        )
        return cv2.bitwise_or(low, high)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Detect object clusters and publish their annotations and visualization.

        :raises ImageContourMissing: If the color threshold produces no contours.
        """
        start_timer = default_timer()
        resized_color = self._prepare_images()
        self.hsv = cv2.cvtColor(resized_color, cv2.COLOR_BGR2HSV_FULL)
        self.adjust_hsv_threshold_to_query()
        hsv_mask = self._threshold_hsv(
            self.descriptor.parameters.hsv_min,
            self.descriptor.parameters.hsv_max,
        )
        contours = self._select_contours(hsv_mask)
        clusters = self._extract_clusters(resized_color, contours)
        visualization_image = cv2.bitwise_and(
            resized_color, resized_color, mask=hsv_mask
        )
        if (
            self.display_mode == self.ViewMode.depth_mask
            and clusters.depth_mask is not None
        ):
            visualization_image = clusters.depth_mask
        visualization_image = self._draw_hypotheses(
            visualization_image, clusters.object_hypotheses
        )
        self.get_annotator_output_struct().set_image(visualization_image)
        self.get_annotator_output_struct().set_geometries(clusters.point_clouds)
        self.feedback_message = (
            f"Processing took {(default_timer() - start_timer):.4f}s"
        )
        return Status.SUCCESS

    # %% Image preparation and contour selection
    def _prepare_images(self) -> npt.NDArray:
        """Load color and depth images and align the projection intrinsics."""
        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        cas = self.get_cas()

        resized_color = get_scaled_color_image_for_depth_image(cas, self.color)
        color_to_depth_ratio = cas.color_to_depth_ratio
        pointcloud_camera_model = (
            cas.require_camera_observation().effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            )
        )
        self.camera_intrinsics = Open3DCameraModelAdapter.to_intrinsic(
            pointcloud_camera_model
        )

        return resized_color

    def _select_contours(self, hsv_mask: npt.NDArray) -> list[npt.NDArray]:
        """Select the largest outer contours that exceed the configured minimum area.

        :raises ImageContourMissing: If the mask contains no contours.
        """
        contours, hierarchy = cv2.findContours(
            image=hsv_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            raise ImageContourMissing(context="image cluster extraction")

        contours = [
            contour
            for contour_index, contour in enumerate(contours)
            if hierarchy[0][contour_index][3] == -1
        ]
        contour_areas = np.asarray([cv2.contourArea(contour) for contour in contours])
        filtered_contour_areas = [
            area
            for area in contour_areas
            if area > self.descriptor.parameters.contour_min_size
        ]
        sorted_areas = sorted(filtered_contour_areas, reverse=True)
        largest_elements = sorted_areas[: self.descriptor.parameters.num_of_objects]
        filtered_contours = [
            contours[contour_index]
            for contour_index, area in enumerate(contour_areas)
            if area in largest_elements
        ]

        return filtered_contours

    # %% Point clouds and annotations
    def _create_point_cloud(
        self,
        resized_color: npt.NDArray,
        contour_mask: npt.NDArray,
        maximum_depth: float = 9.0,
    ) -> open3d.geometry.PointCloud:
        """Project the masked depth image and apply configured outlier removal."""
        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        masked_depth = np.where(contour_mask == 255, self.depth, 0)

        color_image = open3d.geometry.Image(color_rgb)
        depth_image = open3d.geometry.Image(masked_depth)
        rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            convert_rgb_to_intensity=False,
            depth_trunc=maximum_depth,
        )

        cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_intrinsics
        )

        if self.descriptor.parameters.outlier_removal:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=self.descriptor.parameters.outlier_removal_nb_neighbors,
                std_ratio=self.descriptor.parameters.outlier_removal_std_ratio,
            )
        return cloud

    def _extract_clusters(
        self,
        resized_color: npt.NDArray,
        contours: list[npt.NDArray],
        erosion_kernel_size: int = 5,
    ) -> ImageClusters:
        """Create contour point clouds and annotate those meeting the minimum size."""
        clusters = ImageClusters()
        for contour_index, contour in enumerate(contours):
            contour_mask = np.zeros_like(self.depth, dtype=np.uint8)
            cv2.drawContours(
                image=contour_mask,
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED,
            )

            # Erosion removes background near imperfectly aligned object boundaries.
            kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
            contour_mask = cv2.erode(
                contour_mask,
                kernel,
                iterations=self.descriptor.parameters.erosion_iterations,
            )
            clusters.depth_mask = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)

            cloud = self._create_point_cloud(resized_color, contour_mask)
            clusters.point_clouds.append(cloud)

            if len(cloud.points) < self.descriptor.parameters.min_points_threshold:
                continue

            object_hypothesis = ObjectHypothesis()
            object_hypothesis.id = str(contour_index)
            object_hypothesis.source = self.name
            object_hypothesis.points = cloud

            left, top, width, height = cv2.boundingRect(contour)
            object_hypothesis.roi.roi.pos.x = left
            object_hypothesis.roi.roi.pos.y = top
            object_hypothesis.roi.roi.width = width
            object_hypothesis.roi.roi.height = height

            clusters.object_hypotheses.append(object_hypothesis)
            self.get_cas().annotations.append(object_hypothesis)

        return clusters

    # %% Visualization
    def _draw_hypotheses(
        self,
        visualization_image: npt.NDArray,
        object_hypotheses: list[ObjectHypothesis],
    ) -> npt.NDArray:
        """Draw object identifiers, point counts, and bounding rectangles."""
        for object_hypothesis in object_hypotheses:
            region_of_interest = object_hypothesis.roi.roi
            upper_left = (region_of_interest.pos.x, region_of_interest.pos.y)
            upper_left_text = (region_of_interest.pos.x, region_of_interest.pos.y - 5)

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_image = cv2.putText(
                visualization_image,
                f"ROI-{object_hypothesis.id}({len(object_hypothesis.points.points)})",
                upper_left_text,
                font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )
            visualization_image = cv2.rectangle(
                visualization_image,
                upper_left,
                (
                    region_of_interest.pos.x + region_of_interest.width,
                    region_of_interest.pos.y + region_of_interest.height,
                ),
                (0, 0, 255),
                2,
            )

        return visualization_image

    def key_callback(self, key: int) -> None:
        """
        Handle keyboard input to change visualization mode.

        :param key: ASCII value of pressed key
        """
        if key == ord("1"):
            self.display_mode = self.ViewMode.masked_object
        if key == ord("2"):
            self.display_mode = self.ViewMode.depth_mask
