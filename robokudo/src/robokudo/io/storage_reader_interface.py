"""
MongoDB storage reader interface for RoboKudo.

This module provides an interface for reading stored sensor data and annotations
from a MongoDB database. It supports:

* Reading stored RGB-D camera data
* Loading camera calibration information
* Restoring annotations and views
* Automatic cursor management
* Optional looping through stored data

The module is primarily used for:

* Replaying recorded data
* Testing and debugging pipelines
* Offline data analysis
* Visualization of stored data
"""

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, ClassVar
from uuid import UUID

from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World

from robokudo import world as rk_world
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
    MongoReplayMode,
)

from robokudo.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.exceptions import (
    CameraObservationMissing,
    InvalidCameraObservation,
    UnknownMode,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.cas_view_codecs import CameraObservationCodec, ViewPayload
from robokudo.io.sensor_world import SensorWorldProjector
from robokudo.io.storage import Storage, StorageDocumentField


@dataclass(frozen=True)
class RestoredWorld:
    """
    Capture the lookup context and duration of one snapshot restoration.
    """

    tracker: WorldEntityWithIDKwargsTracker
    """
    Entity lookup context for the restored world.
    """

    duration_seconds: float
    """
    Elapsed time spent decoding and applying the snapshot.
    """


class StorageReaderInterface(CameraInterface):
    """
    A camera interface for reading data from MongoDB storage.

    This interface reads sensor data and annotations that were previously stored using
    the StorageWriter annotator. It handles data deserialization and restoration of the
    Common Analysis Structure (CAS) views.
    """

    SENSOR_VIEW_NAMES: ClassVar[frozenset[CASViews]] = frozenset(
        {
            CASViews.COLOR_IMAGE,
            CASViews.DEPTH_IMAGE,
            CASViews.COLOR2DEPTH_RATIO,
            CASViews.CAMERA_OBSERVATION,
        }
    )
    """
    CAS views that represent recorded sensor inputs.
    """

    def __init__(self, camera_config: MongoCameraConfig) -> None:
        """
        Initialize the storage reader interface.

        Sets up MongoDB connection and creates a list reader for the specified database.

        :param camera_config: Configuration containing database settings
        """
        super().__init__(camera_config)

        self.storage: Storage = Storage(camera_config.db_name)
        """
        MongoDB storage interface.
        """
        self.reader: Storage.ListReader = self.storage.ListReader(camera_config.db_name)
        """
        List-based reader for MongoDB data.
        """
        self._recorded_camera_id: UUID | None = None
        """
        Camera identity selected from the first recorded world snapshot.
        """

    def _restore_sensor_world(self, cas_frame: dict[str, Any]) -> None:
        """
        Extract the recorded camera into the active reasoning world.
        """
        view_ids = cas_frame[StorageDocumentField.VIEW_IDS]
        if CASViews.CAMERA_OBSERVATION not in view_ids:
            raise CameraObservationMissing()
        camera_document = self.storage.load_view_document(
            cas_frame, CASViews.CAMERA_OBSERVATION
        )
        camera_payload = ViewPayload.from_document(camera_document)

        snapshot_started = perf_counter()
        recorded_world = World()
        restored = self._restore_world(
            cas_frame[StorageDocumentField.WORLD], recorded_world
        )
        snapshot_duration = perf_counter() - snapshot_started

        observation = CameraObservationCodec().decode_with_tracker(
            camera_payload, restored.tracker
        )
        projection_started = perf_counter()
        projector = SensorWorldProjector(observation)
        reasoning_world = rk_world.world_instance()
        recorded_entities = [observation.camera, observation.camera.root]
        if observation.world_T_camera is not None:
            reference = observation.world_T_camera.reference_frame
            if reference is not None:
                recorded_entities.append(reference)
            pose_child = observation.world_T_camera.child_frame
            if pose_child is not None and pose_child is not observation.camera.root:
                recorded_entities.append(pose_child)
        if any(
            reasoning_world.find_world_entity_with_id(entity.id) is not None
            for entity in recorded_entities
        ):
            raise InvalidCameraObservation(
                reason="recorded sensor entity already exists in the reasoning world"
            )
        if any(
            reasoning_world.get_bodies_by_name(entity.name)
            for entity in recorded_entities
            if entity is not observation.camera
        ) or any(
            camera.name == observation.camera.name
            for camera in reasoning_world.get_semantic_annotations_by_type(Camera)
        ):
            raise InvalidCameraObservation(
                reason="recorded sensor name already exists in the reasoning world"
            )
        projector.install(reasoning_world)
        rk_world.init_world_entity_tracker_from_world(reasoning_world)
        self._recorded_camera_id = observation.camera.id
        projection_duration = perf_counter() - projection_started
        self.rk_logger.debug(
            "Mongo sensor replay restored full world in %.3f ms and projected camera in %.3f ms",
            snapshot_duration * 1000.0,
            projection_duration * 1000.0,
        )

    @staticmethod
    def _restore_world(snapshot: str, target_world: World) -> RestoredWorld:
        """
        Apply a recorded snapshot into the supplied world.
        """
        started = perf_counter()
        tracker = WorldEntityWithIDKwargsTracker.from_world(target_world)
        WorldModelSnapshot.apply_to_json_snapshot_to_world(
            target_world, json.loads(snapshot), **tracker.create_kwargs()
        )
        return RestoredWorld(tracker=tracker, duration_seconds=perf_counter() - started)

    def _restore_full_world(self, cas_frame: dict[str, Any]) -> None:
        """
        Replace reasoning state with the complete recorded frame world.
        """
        rk_world.init_world_with_entity_tracker()
        reasoning_world = rk_world.world_instance()
        restored = self._restore_world(
            cas_frame[StorageDocumentField.WORLD], reasoning_world
        )
        rk_world.init_world_entity_tracker_from_world(reasoning_world)
        self.rk_logger.debug(
            "Mongo full replay restored world in %.3f ms",
            restored.duration_seconds * 1000.0,
        )

    def has_new_data(self) -> bool:
        """
        Check if more data is available to read.

        Handles looping behavior based on camera configuration and maintains cursor
        position in the data sequence.

        :return: True if more data is available, False otherwise
        """
        # Check if we have to reinitialize the cursor after we hit the end of the recorded data
        if self.camera_config.loop and not self.reader.cursor_has_frames():
            self.reader.reset_cursor()

        return self.reader.cursor_has_frames()

    def set_data(self, cas: CAS) -> None:
        """
        Update the Common Analysis Structure with data from storage.

        This method:
        * Retrieves the next frame from storage
        * Restores views and annotations
        * Updates camera intrinsics
        * Sets depth availability flag

        :param cas: Common Analysis Structure to update
        """
        cas_frame = self.reader.get_next_frame()
        if cas_frame is None:
            self.rk_logger.debug(f"Reader has no next frame cas_frame:={cas_frame}")
            return

        if self.camera_config.replay_mode == MongoReplayMode.SENSOR_CONTEXT:
            if self._recorded_camera_id is None:
                self._restore_sensor_world(cas_frame)
            included_views = set(self.SENSOR_VIEW_NAMES)
        elif self.camera_config.replay_mode == MongoReplayMode.FULL_WORLD:
            self._restore_full_world(cas_frame)
            included_views = None
        else:
            raise UnknownMode(
                mode=self.camera_config.replay_mode, context="Mongo replay"
            )
        cas_frame[StorageDocumentField.VIEWS] = {}
        self.storage.load_views_from_mongo_in_cas(
            cas_frame, included_view_names=included_views
        )

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in cas_frame[StorageDocumentField.VIEWS].items():
            if view_name == CASViews.CAMERA_OBSERVATION:
                cas.camera_observation = view_content
            else:
                cas.set(view_name, view_content)
        observation = cas.require_camera_observation()
        if self.camera_config.replay_mode == MongoReplayMode.SENSOR_CONTEXT:
            if observation.camera.id != self._recorded_camera_id:
                raise InvalidCameraObservation(
                    reason="recorded camera identity changed during playback"
                )
            SensorWorldProjector.apply_pose(rk_world.world_instance(), observation)

        # Restore annotations
        if self.camera_config.restore_annotations:
            self.storage.load_annotations_from_mongo_in_cas(cas_frame, cas)

        if cas.depth_image is None:
            # no depth image available
            AnnotatorPredefinedParameters.global_with_depth = False
