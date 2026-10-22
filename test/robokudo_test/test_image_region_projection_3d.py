"""
Projection of image regions into colored depth points.
"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np
import open3d as o3d
import pytest

from robokudo.cas import CAS, CASViews
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.utils.image_region_projection_3d import (
    ImageCoordinates,
    ImageRegionProjection3D,
)
from robokudo.utils.o3d_helper import get_cloud_from_rgb_depth_and_mask
from robokudo.utils.exceptions import ProjectionMaskShapeMismatch

# %% region shapes


class ProjectionRegion(StrEnum):
    """
    Image regions whose projection must preserve selected pixels.
    """

    BOX = "box"
    """
    Filled rectangular region.
    """

    DISC = "disc"
    """
    Filled circular region.
    """

    MASK = "mask"
    """
    Foreground pixels of a full-image mask.
    """

    CROPPED_MASK = "cropped_mask"
    """
    Foreground pixels of a mask positioned within a rectangle.
    """


# %% camera measurements


@pytest.fixture(
    params=[
        ColorToDepthRatio(),
        ColorToDepthRatio(horizontal=0.5, vertical=0.5),
        ColorToDepthRatio(horizontal=0.5, vertical=1.0),
    ]
)
def projection_cas(request: pytest.FixtureRequest) -> CAS:
    """
    Provide registered images with equal, uniform or nonuniform resolution ratios.
    """
    ratio = request.param
    color = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    height, width = color.shape[:2]
    depth = np.full(
        (int(height * ratio.vertical), int(width * ratio.horizontal)),
        1000,
        dtype=np.uint16,
    )
    cas = CAS()
    cas.set(CASViews.COLOR_IMAGE, color)
    cas.set(CASViews.DEPTH_IMAGE, depth)
    cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ratio)
    cas.set(
        CASViews.CAMERA_INTRINSIC,
        o3d.camera.PinholeCameraIntrinsic(
            width, height, width, height, width / 2, height / 2
        ),
    )
    return cas


# %% region projection


@pytest.mark.parametrize("coordinates", list(ImageCoordinates))
@pytest.mark.parametrize("region_kind", list(ProjectionRegion))
def test_regions_select_matching_depth_points(
    projection_cas: CAS, coordinates: ImageCoordinates, region_kind: ProjectionRegion
) -> None:
    """
    Project each supported region in either image coordinate system.
    """
    projection = ImageRegionProjection3D.from_cas(projection_cas)
    color = projection_cas.get(CASViews.COLOR_IMAGE)
    depth = projection_cas.get(CASViews.DEPTH_IMAGE)
    shape = color.shape[:2] if coordinates == ImageCoordinates.COLOR else depth.shape
    mask = np.zeros(shape, dtype=np.uint8)
    region = Rect.from_corner_points(1, 1, shape[1] - 1, shape[0] - 1)
    left, top, right, bottom = region.get_corner_points()
    if region_kind == ProjectionRegion.DISC:
        center = Point2D(x=shape[1] // 2, y=shape[0] // 2)
        radius = 1
        cv2.circle(
            mask, (center.x, center.y), radius, projection.mask_foreground, cv2.FILLED
        )
        cloud = projection.project_disc(center, radius, coordinates=coordinates)
    else:
        mask[top:bottom, left:right] = projection.mask_foreground
        if region_kind == ProjectionRegion.BOX:
            cloud = projection.project_box(region, coordinates=coordinates)
        elif region_kind == ProjectionRegion.CROPPED_MASK:
            cloud = projection.project_mask(
                mask[top:bottom, left:right], region=region, coordinates=coordinates
            )
        else:
            cloud = projection.project_mask(mask, coordinates=coordinates)
    depth_height, depth_width = depth.shape
    expected_mask = cv2.resize(
        mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST
    )
    expected = get_cloud_from_rgb_depth_and_mask(
        cv2.cvtColor(
            cv2.resize(
                color, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST
            ),
            cv2.COLOR_BGR2RGB,
        ),
        depth,
        expected_mask,
        o3d.camera.PinholeCameraIntrinsic(
            depth_width,
            depth_height,
            depth_width,
            depth_height,
            depth_width / 2,
            depth_height / 2,
        ),
    )
    np.testing.assert_array_equal(cloud.points, expected.points)
    np.testing.assert_array_equal(cloud.colors, expected.colors)
    np.testing.assert_array_equal(projection_cas.get(CASViews.COLOR_IMAGE), color)
    np.testing.assert_array_equal(projection_cas.get(CASViews.DEPTH_IMAGE), depth)


def test_depth_limit_excludes_distant_measurements(projection_cas: CAS) -> None:
    """
    Apply the configured depth limit to every region projection.
    """
    projection = ImageRegionProjection3D.from_cas(projection_cas, depth_truncate=0.5)
    region = Rect.from_corner_points(
        0, 0, *projection_cas.get(CASViews.COLOR_IMAGE).shape[1::-1]
    )
    cloud = projection.project_box(region)
    np.testing.assert_array_equal(np.asarray(cloud.points), np.empty((0, 3)))


def test_box_outside_image_selects_no_points(projection_cas: CAS) -> None:
    """
    Keep negative coordinates outside the image from wrapping around its edges.
    """
    projection = ImageRegionProjection3D.from_cas(projection_cas)
    cloud = projection.project_box(Rect.from_corner_points(-4, -4, -1, -1))
    np.testing.assert_array_equal(np.asarray(cloud.points), np.empty((0, 3)))


def test_mask_shape_must_match_selected_image(projection_cas: CAS) -> None:
    """
    Reject masks whose dimensions do not match the declared coordinates.
    """
    projection = ImageRegionProjection3D.from_cas(projection_cas)
    with pytest.raises(ProjectionMaskShapeMismatch):
        projection.project_mask(np.zeros((1, 1), dtype=np.uint8))
