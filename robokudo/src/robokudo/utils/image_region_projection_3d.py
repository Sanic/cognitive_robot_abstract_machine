"""
Projection of registered image regions into colored 3D points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
import open3d as o3d
from krrood.ormatic.utils import classproperty
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.types.cv import Point2D, Rect
from robokudo.utils.cv_helper import (
    clamp_bounding_rect,
    crop_image,
    get_scaled_color_image_for_depth_image,
)
from robokudo.utils.exceptions import ProjectionMaskShapeMismatch
from robokudo.utils.o3d_helper import (
    get_cloud_from_rgb_depth_and_mask,
    scale_o3d_camera_intrinsics,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.cas import CAS

# %% image coordinates


class ImageCoordinates(StrEnum):
    """
    The image whose pixels define a region's position and extent.
    """

    COLOR = "color"
    """
    Pixels of the original color image.
    """

    DEPTH = "depth"
    """
    Pixels of the depth image.
    """


# %% region projection


@dataclass
class ImageRegionProjection3D:
    """
    Project image regions into colored points in the camera optical frame.

    Color and depth must be spatially registered. Region coordinates default to the
    original color image; depth-image coordinates can be selected explicitly.
    """

    rgb_image: npt.NDArray[np.uint8]
    """
    RGB image at depth resolution.
    """

    depth_image: npt.NDArray[np.uint16]
    """
    Registered depth measurements in millimeters.
    """

    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic
    """
    Camera calibration at depth resolution.
    """

    color_region: Rect
    """
    Full extent of the original color image.
    """

    depth_truncate: float = 9.0
    """
    Distance in meters beyond which measurements are excluded.
    """

    @classproperty
    def mask_foreground(cls) -> int:
        """
        Byte value marking selected image pixels.
        """
        return np.iinfo(np.uint8).max

    @classmethod
    def from_cas(
        cls, cas: CAS, depth_truncate: float | None = None
    ) -> ImageRegionProjection3D:
        """
        Prepare registered images and calibration for region projection.

        :param cas: Camera measurements with color-image intrinsics and a resolution ratio.
        :param depth_truncate: Optional distance limit in meters.
        :return: Projection of the current camera measurements.
        """
        color_image = cas.get(CASViews.COLOR_IMAGE)
        scaled_color = get_scaled_color_image_for_depth_image(cas, color_image)
        ratio = cas.get(CASViews.COLOR_TO_DEPTH_RATIO)
        height, width = color_image.shape[:2]
        projection = cls(
            rgb_image=cv2.cvtColor(scaled_color, cv2.COLOR_BGR2RGB),
            depth_image=cas.get(CASViews.DEPTH_IMAGE),
            camera_intrinsics=scale_o3d_camera_intrinsics(
                cas.get(CASViews.CAMERA_INTRINSIC), ratio.horizontal, ratio.vertical
            ),
            color_region=Rect(width=width, height=height),
        )
        if depth_truncate is not None:
            projection.depth_truncate = depth_truncate
        return projection

    def project_box(
        self,
        region: Rect,
        coordinates: ImageCoordinates = ImageCoordinates.COLOR,
        depth_truncate: float | None = None,
    ) -> o3d.geometry.PointCloud:
        """
        Project measurements inside a rectangle, clipped to the image boundaries.

        :param region: Rectangle in the selected image coordinates.
        :param coordinates: Image defining the rectangle's pixels.
        :param depth_truncate: Optional distance limit in meters.
        :return: Colored points within the rectangle.
        """
        mask = self._empty_mask(coordinates)
        self._crop_mask(mask, region)[:] = self.mask_foreground
        return self.project_mask(
            mask, coordinates=coordinates, depth_truncate=depth_truncate
        )

    def project_disc(
        self,
        center: Point2D,
        radius: int,
        coordinates: ImageCoordinates = ImageCoordinates.COLOR,
        depth_truncate: float | None = None,
    ) -> o3d.geometry.PointCloud:
        """
        Project measurements inside a filled circle.

        :param center: Circle center in the selected image coordinates.
        :param radius: Circle radius in pixels of the same image.
        :param coordinates: Image defining both the center and radius.
        :param depth_truncate: Optional distance limit in meters.
        :return: Colored points within the circle.
        """
        mask = self._empty_mask(coordinates)
        cv2.circle(
            mask,
            (center.x, center.y),
            radius,
            self.mask_foreground,
            thickness=cv2.FILLED,
        )
        return self.project_mask(
            mask, coordinates=coordinates, depth_truncate=depth_truncate
        )

    def project_mask(
        self,
        mask: npt.NDArray[np.uint8],
        coordinates: ImageCoordinates = ImageCoordinates.COLOR,
        region: Rect | None = None,
        depth_truncate: float | None = None,
    ) -> o3d.geometry.PointCloud:
        """
        Project measurements selected by a full-image or cropped byte mask.

        :param mask: Mask whose foreground equals :attr:`mask_foreground`.
        :param coordinates: Image defining the mask's pixels and optional region.
        :param region: Image region occupied by a cropped mask, clipped to image boundaries.
        :param depth_truncate: Optional distance limit in meters.
        :return: Colored points at foreground pixels with valid depth.
        :raises ProjectionMaskShapeMismatch: If the mask does not match its region.
        """
        image_mask = self._empty_mask(coordinates)
        target = image_mask if region is None else self._crop_mask(image_mask, region)
        if mask.shape != target.shape:
            raise ProjectionMaskShapeMismatch(mask.shape, target.shape)
        target[:] = mask
        depth_height, depth_width = self.depth_image.shape[:2]
        depth_mask = cv2.resize(
            image_mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST
        )
        return get_cloud_from_rgb_depth_and_mask(
            rgb_image=self.rgb_image,
            depth_image=self.depth_image,
            mask=depth_mask,
            camera_intrinsics=self.camera_intrinsics,
            depth_truncate=(
                self.depth_truncate if depth_truncate is None else depth_truncate
            ),
            mask_true_value=self.mask_foreground,
        )

    def _crop_mask(
        self, mask: npt.NDArray[np.uint8], region: Rect
    ) -> npt.NDArray[np.uint8]:
        """
        Select the intersection of an image mask and a rectangle.

        :param mask: Full-image mask.
        :param region: Rectangle in the mask's coordinates.
        :return: A view of the mask within the image boundaries.
        """
        height, width = mask.shape
        left, top, region_width, region_height = clamp_bounding_rect(
            region.as_tuple(), width, height
        )
        return crop_image(
            mask, (left, top), (max(region_width, 0), max(region_height, 0))
        )

    def _empty_mask(self, coordinates: ImageCoordinates) -> npt.NDArray[np.uint8]:
        """
        Allocate a background mask covering the selected image.

        :param coordinates: Image whose dimensions define the mask.
        :return: Byte mask with no selected pixels.
        """
        if coordinates == ImageCoordinates.DEPTH:
            return np.zeros(self.depth_image.shape[:2], dtype=np.uint8)
        return np.zeros(
            (self.color_region.height, self.color_region.width), dtype=np.uint8
        )
