from __future__ import annotations

# %% imports

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any

from krrood.ormatic.utils import classproperty
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry
from robokudo.io.camera_interface import CameraInterface, KinectCameraInterface
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.descriptors.camera_configs.config_olive import OliveCameraConfig
from robokudo.io.rectified_depth_camera_interface import RectifiedDepthCameraInterface
from robokudo.io.camera_without_depth_interface import OpenCVCameraWithoutDepthInterface
from robokudo.io.file_reader_interface import RGBDFileReaderInterface
from robokudo.io.semdt_raytracer_camera_interface import (
    SemDTRayTracerCameraInterface,
)
from robokudo.io.storage_reader_interface import StorageReaderInterface

# %% source identifiers


class CameraSource(StrEnum):
    """
    Camera sources supported by the descriptor factory.
    """

    STORAGE = "mongo"
    KINECT = "kinect"
    KINECT_WITHOUT_TRANSFORM = "kinect_wo_tf"
    REALSENSE = "realsense"
    TIAGO = "tiago"
    OPENCV = "opencv"
    ORBBEC = "orbbec"
    ORBBEC_WITHOUT_TRANSFORM = "orbbec_wo_tf"
    FILE = "file_reader"
    RAYTRACER = "semdt_raytracer"
    HSR = "hsr_ros2"


# %% descriptor construction


@dataclass
class CollectionReaderDescriptorFactory:
    """
    Factory class for creating CollectionReader descriptors.
    """

    @classproperty
    def _camera_interface_types(
        cls: type[CollectionReaderDescriptorFactory],
    ) -> dict[str, type[CameraInterface]]:
        """
        Interface constructors indexed by registered source name.
        """
        return {
            FoundationStereoCameraConfig.registry_name: RectifiedDepthCameraInterface,
            OliveCameraConfig.registry_name: RectifiedDepthCameraInterface,
            CameraSource.STORAGE: StorageReaderInterface,
            CameraSource.KINECT: KinectCameraInterface,
            CameraSource.KINECT_WITHOUT_TRANSFORM: KinectCameraInterface,
            CameraSource.REALSENSE: KinectCameraInterface,
            CameraSource.TIAGO: KinectCameraInterface,
            CameraSource.OPENCV: OpenCVCameraWithoutDepthInterface,
            CameraSource.ORBBEC: KinectCameraInterface,
            CameraSource.ORBBEC_WITHOUT_TRANSFORM: KinectCameraInterface,
            CameraSource.FILE: RGBDFileReaderInterface,
            CameraSource.RAYTRACER: SemDTRayTracerCameraInterface,
            CameraSource.HSR: KinectCameraInterface,
        }

    @staticmethod
    def create_descriptor(
        camera: str, **kwargs: Any
    ) -> CollectionReaderAnnotator.Descriptor:
        """
        Create a CollectionReader descriptor for the specified camera.

        :param camera: The name of the camera to create a descriptor for.
        :param kwargs: Additional keyword arguments to pass to the camera configuration.
        :returns: A CollectionReader descriptor for the specified camera.
        :raises ValueError: If the given camera config name is not registered in the
            camera config registry.
        :raises TypeError: If the keyword arguments are invalid for the camera config
            class.
        """
        camera_config = CameraConfigRegistry.create_config(camera, **kwargs)
        return CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=CollectionReaderDescriptorFactory._camera_interface_types[
                camera
            ](camera_config),
        )
