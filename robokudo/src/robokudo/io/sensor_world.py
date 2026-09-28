"""
Project recorded camera references into a small reasoning world.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from robokudo.exceptions import InvalidCameraObservation
from robokudo.types.camera import CameraObservation
from robokudo.utils.ros_frames import normalize_ros_frame_id
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(frozen=True)
class SensorWorldProjector:
    """
    Create a world containing only a recorded camera and its pose frames.

    The sampled pose connects to its reference frame. A camera root with an equivalent
    ROS frame name attaches directly to that pose's child frame.
    """

    observation: CameraObservation
    """
    Recorded camera state that identifies the sensor context.
    """

    @staticmethod
    def _is_camera_frame_alias(pose_child: Body, camera_root: Body) -> bool:
        """
        Recognize ROS frame names that differ only by leading slashes.
        """
        return (
            pose_child.name.prefix == camera_root.name.prefix
            and pose_child.name.name != camera_root.name.name
            and normalize_ros_frame_id(pose_child.name.name)
            == normalize_ros_frame_id(camera_root.name.name)
        )

    def project(self) -> World:
        """
        Copy the camera and direct pose reference with their recorded identifiers.
        """
        source_camera = self.observation.camera
        source_root = source_camera.root
        source_pose = self.observation.world_T_camera
        source_pose_child = (
            source_pose.child_frame
            if source_pose is not None and source_pose.child_frame is not None
            else source_root
        )
        if source_pose is not None and source_pose.reference_frame is None:
            raise InvalidCameraObservation(
                reason="sampled camera pose has no reference frame"
            )
        if source_pose_child is not source_root and (
            not isinstance(source_pose_child, Body)
            or not self._is_camera_frame_alias(source_pose_child, source_root)
        ):
            raise InvalidCameraObservation(
                reason="sampled pose child frame differs from the camera root"
            )

        source_reference = (
            source_pose.reference_frame if source_pose is not None else None
        )
        root = Body(name=deepcopy(source_root.name), id=source_root.id)
        pose_child = (
            Body(name=deepcopy(source_pose_child.name), id=source_pose_child.id)
            if source_pose_child is not source_root
            else root
        )
        reference = (
            Body(name=deepcopy(source_reference.name), id=source_reference.id)
            if source_reference is not None and source_reference is not source_root
            else None
        )
        if reference is None and pose_child is not root:
            raise InvalidCameraObservation(
                reason="aliased camera pose has no separate reference frame"
            )
        axis = source_camera.forward_facing_axis.to_np().reshape(-1)
        camera = Camera(
            id=source_camera.id,
            name=deepcopy(source_camera.name),
            root=root,
            forward_facing_axis=Vector3(*axis[:3]),
            camera_model=deepcopy(source_camera.camera_model),
            camera_range=deepcopy(source_camera.camera_range),
            modalities=tuple(source_camera.modalities),
        )

        sensor_world = World()
        connection = None
        with sensor_world.modify_world():
            if reference is not None:
                sensor_world.add_body(reference)
            if pose_child is not root:
                sensor_world.add_body(pose_child)
            sensor_world.add_body(root)
            if reference is not None:
                connection = Connection6DoF.create_with_dofs(
                    parent=reference,
                    child=pose_child,
                    world=sensor_world,
                    name=PrefixedName(
                        name=f"{pose_child.name.name}_T_{reference.name.name}"
                    ),
                )
                sensor_world.add_connection(connection)
            if pose_child is not root:
                sensor_world.add_connection(
                    Connection6DoF.create_with_dofs(
                        parent=pose_child,
                        child=root,
                        world=sensor_world,
                        name=PrefixedName(
                            name=f"{root.name.name}_T_{pose_child.name.name}"
                        ),
                    )
                )
            sensor_world.add_semantic_annotation(camera)
        if connection is not None:
            with sensor_world.modify_world():
                connection.origin = HomogeneousTransformationMatrix(
                    data=source_pose,
                    reference_frame=reference,
                    child_frame=pose_child,
                )
        return sensor_world

    def install(self, reasoning_world: World) -> Camera:
        """
        Merge the sensor context and bind the camera to its merged root body.
        """
        reasoning_world.merge_world(self.project())
        camera = reasoning_world.get_semantic_annotation_by_id(
            self.observation.camera.id
        )
        root = reasoning_world.get_kinematic_structure_entity_by_id(
            self.observation.camera.root.id
        )
        if not isinstance(camera, Camera) or not isinstance(root, Body):
            raise InvalidCameraObservation(
                reason="sensor world did not contain a camera and root body"
            )
        with reasoning_world.modify_world():
            camera.root = root
            camera.forward_facing_axis.reference_frame = root
        return camera

    @staticmethod
    def apply_pose(world: World, observation: CameraObservation) -> None:
        """
        Apply one sampled camera pose to its direct sensor connection.
        """
        pose = observation.world_T_camera
        pose_child = (
            pose.child_frame
            if pose is not None and pose.child_frame is not None
            else observation.camera.root
        )
        if pose_child is not observation.camera.root and (
            not isinstance(pose_child, Body)
            or not SensorWorldProjector._is_camera_frame_alias(
                pose_child, observation.camera.root
            )
        ):
            raise InvalidCameraObservation(
                reason="sampled pose child frame differs from the camera root"
            )
        if pose is None:
            return
        if pose.reference_frame is None:
            raise InvalidCameraObservation(
                reason="sampled camera pose has no reference frame"
            )
        if pose.reference_frame is pose_child:
            return
        connection = pose_child.parent_connection
        if connection is None or connection.parent is not pose.reference_frame:
            raise InvalidCameraObservation(
                reason="camera pose is not connected to its reference frame"
            )
        with world.modify_world():
            connection.origin = pose
