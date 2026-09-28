"""
Resolve the Semantic Digital Twin world and camera used for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import robokudo.world as rk_world
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    RuntimeRobotWorldSource,
    SemDTRayTracerCameraConfig,
    WorldDescriptorSource,
)
from robokudo.exceptions import (
    CameraAnnotationAmbiguous,
    CameraAnnotationMissing,
    CameraPoseOverrideUnavailable,
    RobotAnnotationAmbiguous,
    RobotAnnotationMissing,
)
from robokudo.utils.module_loader import ModuleLoader
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF

# %% Resolved camera context


@dataclass
class RayTracingContext:
    """
    Pair the semantic world with the camera used to render it.
    """

    world: World
    """Semantic world rendered for the current camera source."""

    camera: Camera
    """
    Semantic camera that supplies pose, calibration, and range.
    """


# %% Context resolution


class SemDTCameraContextResolver:
    """
    Resolve and cache a ray-tracing context from a configured world source.
    """

    def __init__(
        self,
        camera_config: SemDTRayTracerCameraConfig,
        module_loader: ModuleLoader | None = None,
    ) -> None:
        """
        Initialize source resolution.

        :param camera_config: Camera configuration containing the world source.
        :param module_loader: Optional loader supplied by focused tests.
        """
        self.camera_config = camera_config
        """
        Camera configuration whose source is resolved.
        """
        self.module_loader = module_loader or ModuleLoader()
        """
        Loader used to import configured SemDT world descriptors.
        """
        self._context: RayTracingContext | None = None
        """
        Resolved world-camera pair after successful source resolution.
        """

    def resolve(self) -> RayTracingContext:
        """
        Resolve and cache a complete world-camera pair.

        Failed runtime resolution remains uncached so a later synchronization attempt
        can make the robot and its camera available.

        :return: Runtime world and camera used for rendered frames.
        """
        if self._context is not None:
            return self._context

        source = self.camera_config.source
        if isinstance(source, WorldDescriptorSource):
            world = self._load_descriptor_world(source)
            camera = self._select_descriptor_camera(world, source)
            self._apply_camera_pose(world, camera, source.camera_pose)
            rk_world.set_world(world)
        else:
            world = self._load_runtime_world()
            camera = self._select_runtime_robot_camera(world, source)

        rk_world.init_world_entity_tracker_from_world(world)
        self._context = RayTracingContext(world=world, camera=camera)
        return self._context

    @staticmethod
    def _apply_camera_pose(
        world: World,
        camera: Camera,
        camera_pose: HomogeneousTransformationMatrix | None,
    ) -> None:
        """
        Apply the configured initial camera pose when one is present.

        :param world: Runtime world that owns the camera connection.
        :param camera: Selected semantic camera.
        :param camera_pose: Initial camera-root pose, or ``None`` to preserve its pose.
        :raises CameraPoseOverrideUnavailable: If the camera attachment is immutable.
        """
        if camera_pose is None:
            return

        parent_connection = camera.root.parent_connection
        if not isinstance(parent_connection, Connection6DoF):
            raise CameraPoseOverrideUnavailable(
                camera_name=camera.name.name,
                connection_type=(
                    None
                    if parent_connection is None
                    else parent_connection.__class__.__name__
                ),
            )

        reference_frame = camera_pose.reference_frame or parent_connection.parent
        reference_T_camera = HomogeneousTransformationMatrix(
            data=camera_pose,
            reference_frame=reference_frame,
            child_frame=camera.root,
        )
        with world.modify_world():
            parent_connection.origin = reference_T_camera

    @staticmethod
    def _select_descriptor_camera(
        world: World,
        source: WorldDescriptorSource,
    ) -> Camera:
        """
        Select a camera declared by a standalone world descriptor.

        :param world: Descriptor world containing camera annotations.
        :param source: Descriptor source carrying the optional exact camera name.
        :return: Camera that provides the rendered frame.
        :raises CameraAnnotationMissing: If a configured camera does not exist.
        :raises CameraAnnotationAmbiguous: If automatic selection is ambiguous.
        """
        cameras = world.get_semantic_annotations_by_type(Camera)
        camera_name = source.camera_name
        if camera_name is not None:
            named_cameras = [camera for camera in cameras if camera.name == camera_name]
            if len(named_cameras) == 0:
                raise CameraAnnotationMissing(camera_name=str(camera_name))
            if len(named_cameras) > 1:
                raise CameraAnnotationAmbiguous(
                    camera_names=tuple(str(camera.name) for camera in named_cameras)
                )
            return named_cameras[0]

        if len(cameras) == 1:
            return cameras[0]
        if len(cameras) > 1:
            raise CameraAnnotationAmbiguous(
                camera_names=tuple(str(camera.name) for camera in cameras)
            )
        raise CameraAnnotationMissing()

    def _load_descriptor_world(self, source: WorldDescriptorSource) -> World:
        """
        Load the standalone world described by ``source``.

        :param source: Descriptor module coordinates.
        :return: Newly constructed descriptor world.
        """
        world_descriptor = self.module_loader.load_world_descriptor(
            ros_pkg_name=source.ros_package,
            module_name=source.descriptor_name,
        )
        return world_descriptor.world

    @staticmethod
    def _load_runtime_world() -> World:
        """
        Return the semantic world currently installed in RoboKudo.

        :return: Current runtime world, which may have been populated after config
            construction.
        """
        # TODO(WorldSynchronizer): Resolve this reference through the synchronization
        # API once it owns publication of the current semantic world.
        return rk_world.world_instance()

    @staticmethod
    def _select_runtime_robot_camera(
        world: World,
        source: RuntimeRobotWorldSource,
    ) -> Camera:
        """
        Select the runtime robot and return its default camera.

        :param world: Runtime world containing robot annotations.
        :param source: Runtime source carrying the optional exact robot name.
        :return: Default camera selected through the robot's Coraplex-compatible API.
        """
        robots = world.get_semantic_annotations_by_type(AbstractRobot)
        robot_name = source.robot_name
        if robot_name is not None:
            robots = [robot for robot in robots if robot.name == robot_name]
            if len(robots) == 0:
                raise RobotAnnotationMissing(robot_name=str(robot_name))

        if len(robots) == 1:
            return robots[0].get_default_camera()
        if len(robots) > 1:
            raise RobotAnnotationAmbiguous(
                robot_names=tuple(str(robot.name) for robot in robots)
            )
        raise RobotAnnotationMissing()
