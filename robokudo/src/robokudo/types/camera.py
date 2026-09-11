"""
ROS-independent camera metadata for perception frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@dataclass(frozen=True)
class CameraObservation:
    """
    Describe the camera state used to produce one perception frame.

    An annotator can access native and effective parameters from its CAS::

        observation = self.get_cas().camera_observation
        native_model = observation.camera.camera_model
        effective_model = observation.camera_model
        world_T_camera = observation.world_T_camera
    """

    camera: Camera
    """
    Semantic camera that produced the frame.
    """

    camera_model: PinholeCameraModel
    """
    Effective projection model of the delivered image.
    """

    world_T_camera: HomogeneousTransformationMatrix | None
    """
    Sampled camera pose in the world frame, if available.
    """

    timestamp_nanoseconds: int
    """
    Acquisition time as nanoseconds since the Unix epoch.
    """

    @property
    def resolution(self) -> CameraResolution:
        """
        Return the effective image resolution.
        """
        return self.camera_model.resolution

    @property
    def field_of_view(self) -> FieldOfView:
        """
        Return the effective image field of view.
        """
        return self.camera_model.field_of_view
