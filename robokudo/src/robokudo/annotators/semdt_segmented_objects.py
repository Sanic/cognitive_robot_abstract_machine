"""
Localize visible bodies from aligned ray-traced segmentation and depth.
"""

from __future__ import annotations

import numpy as np
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.cv import ImageROI, Point2D, Rect
from robokudo.types.scene import ObjectHypothesis

# %% Segmented object localization

MILLIMETERS_PER_METER = 1000.0
"""
Depth unit conversion for the ray-traced camera image.
"""

MIN_SEGMENT_PIXELS = 30
"""
Minimum visible area for a stable object pose and color estimate.
"""


class SegmentedObjectAnnotator(BaseAnnotator):
    """
    Create a posed object hypothesis for each visible rendered segment.
    """

    def __init__(self) -> None:
        """
        Initialize the segment localization annotator.
        """
        super().__init__(name=type(self).__name__)

    def update(self) -> Status:
        """
        Back-project each visible segment and add its image region and pose.
        """
        cas = self.get_cas()
        segmentation = cas.get(CASViews.OBJECT_IMAGE)
        depth = cas.get(CASViews.DEPTH_IMAGE)
        model = cas.require_camera_observation().effective_camera_model
        focal_x = model.focal_length_x
        focal_y = model.focal_length_y
        center_x = model.principal_point_x
        center_y = model.principal_point_y

        for body_index in np.unique(segmentation):
            if body_index < 0:
                continue
            rows, columns = np.nonzero((segmentation == body_index) & (depth > 0))
            if len(rows) < MIN_SEGMENT_PIXELS:
                continue

            nearest_row, farthest_row = int(rows.min()), int(rows.max()) + 1
            nearest_column, farthest_column = int(columns.min()), int(columns.max()) + 1
            mask = np.where(
                segmentation[nearest_row:farthest_row, nearest_column:farthest_column]
                == body_index,
                255,
                0,
            ).astype(np.uint8)
            z = depth[rows, columns] / MILLIMETERS_PER_METER
            x = (columns - center_x) * z / focal_x
            y = (rows - center_y) * z / focal_y
            pose = PoseAnnotation(
                translation=[float(np.median(axis)) for axis in (x, y, z)],
                rotation=[0.0, 0.0, 0.0, 1.0],
                source=self.name,
            )
            region = ImageROI(
                mask=mask,
                roi=Rect(
                    pos=Point2D(x=nearest_column, y=nearest_row),
                    width=farthest_column - nearest_column,
                    height=farthest_row - nearest_row,
                ),
            )
            cas.annotations.append(ObjectHypothesis(annotations=[pose], roi=region))
        return Status.SUCCESS
