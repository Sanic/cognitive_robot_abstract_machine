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

from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)

from robokudo.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.cas import CAS
from robokudo.io.camera_interface import CameraInterface
from robokudo.io.camera_replay import RecordedCameraRegistry
from robokudo.io.storage import Storage


class StorageReaderInterface(CameraInterface):
    """
    A camera interface for reading data from MongoDB storage.

    This interface reads sensor data and annotations that were previously stored using
    the StorageWriter annotator. It handles data deserialization and restoration of the
    Common Analysis Structure (CAS) views.
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
        self.recorded_cameras = RecordedCameraRegistry()
        """
        Camera instances shared by frames from this recording.
        """

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

        cas_frame["views"] = {}
        self.storage.load_views_from_mongo_in_cas(
            cas_frame, camera_registry=self.recorded_cameras
        )

        # Bring flat CAS representation into the proper CAS class
        for view_name, view_content in cas_frame["views"].items():
            cas.set(view_name, view_content)
        cas.require_camera_observation()

        # Restore annotations
        if self.camera_config.restore_annotations:
            self.storage.load_annotations_from_mongo_in_cas(cas_frame, cas)

        if cas.depth_image is None:
            # no depth image available
            AnnotatorPredefinedParameters.global_with_depth = False
