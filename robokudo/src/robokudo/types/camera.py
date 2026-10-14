"""
ROS-independent camera metadata for perception frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from robokudo.exceptions import CameraPoseMissing
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@dataclass(frozen=True)
class CameraObservation:
    """
    Describe the camera state used to produce one perception frame.

    :attr:`world_T_camera` alone transforms optical-frame points into world coordinates.
    The fixed optical attachment on :attr:`camera` also determines the physical-root
    pose needed for replay.

    An annotator can access native and effective parameters from its CAS::

        observation = self.get_cas().camera_observation
        native_model = observation.camera.camera_model
        effective_model = observation.effective_camera_model
        world_T_camera = observation.world_T_camera
    """

    camera: Camera
    """
    Semantic camera that produced the frame.
    """

    effective_camera_model: PinholeCameraModel
    """
    Effective projection model of the delivered image.
    """

    world_T_camera: HomogeneousTransformationMatrix | None
    """
    Sampled optical-frame pose in world coordinates, if available.

    This transform alone places optical-frame points in the world.
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
        return self.effective_camera_model.resolution

    @property
    def field_of_view(self) -> FieldOfView:
        """
        Return the effective image field of view.
        """
        return self.effective_camera_model.field_of_view

    def world_T_camera_or_raise(self) -> HomogeneousTransformationMatrix:
        """
        Return the sampled pose or raise when the producer had none.

        :return: Transform from the optical frame into the world frame.
        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        if self.world_T_camera is None:
            raise CameraPoseMissing()
        return self.world_T_camera

    @property
    def world_T_root(self) -> HomogeneousTransformationMatrix | None:
        """
        Recover the physical-root pose using the camera's fixed optical attachment.

        :return: Sampled physical-root pose, or no pose when none was sampled.
        """
        if self.world_T_camera is None:
            return None
        return self.world_T_camera @ self.camera.optical_attachment.inverse()

    @property
    def camera_T_world(self) -> HomogeneousTransformationMatrix:
        """
        Return the inverse sampled camera pose.

        :raises CameraPoseMissing: If the observation has no sampled pose.
        """
        return self.world_T_camera_or_raise().inverse()

    def with_world_T_camera(
        self, world_T_camera: HomogeneousTransformationMatrix
    ) -> CameraObservation:
        """
        Return a copy carrying the supplied sampled camera pose.

        :param world_T_camera: Transform from the optical frame into the world frame.
        :return: Observation containing the supplied pose.
        """
        return replace(self, world_T_camera=world_T_camera)
