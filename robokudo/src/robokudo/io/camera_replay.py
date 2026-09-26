"""
Recorded sensor definitions and their runtime camera instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from typing_extensions import Any

from robokudo import world
from robokudo.exceptions import InvalidCameraObservation
from robokudo.io.cas_annotation_codecs import krrood_from_json, krrood_to_json
from semantic_digital_twin.datastructures.camera_model import (
    CameraModel,
    CameraModality,
    CameraRange,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

# %% Stored fields


class CameraSnapshotField(StrEnum):
    """
    Identify fields in a recorded camera definition.
    """

    IDENTITY = "identity"
    NAME = "name"
    ROOT_FRAME = "root_frame"
    FORWARD_AXIS = "forward_axis"
    NATIVE_MODEL = "native_model"
    MINIMUM_DISTANCE = "minimum_distance"
    MAXIMUM_DISTANCE = "maximum_distance"
    MODALITIES = "modalities"


class CameraPoseField(StrEnum):
    """
    Identify fields in a recorded camera pose.
    """

    MATRIX = "matrix"
    REFERENCE_FRAME = "reference_frame"


class RecordedNameField(StrEnum):
    """
    Identify fields in a stored semantic name.
    """

    NAME = "name"
    PREFIX = "prefix"


@dataclass(frozen=True)
class RecordedName:
    """
    Preserve both parts of a semantic name.
    """

    name: str
    """
    Local name of the recorded entity.
    """

    prefix: str | None
    """
    Optional namespace of the recorded entity.
    """

    @classmethod
    def from_name(cls, name: PrefixedName) -> RecordedName:
        """
        Capture a semantic entity name.
        """
        return cls(name=name.name, prefix=name.prefix)

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> RecordedName:
        """
        Read a stored semantic entity name.
        """
        return cls(
            name=document[RecordedNameField.NAME],
            prefix=document[RecordedNameField.PREFIX],
        )

    def to_document(self) -> dict[str, str | None]:
        """
        Write a semantic entity name without a world reference.
        """
        return {
            RecordedNameField.NAME: self.name,
            RecordedNameField.PREFIX: self.prefix,
        }

    def to_name(self) -> PrefixedName:
        """
        Create a semantic name for the replay world.
        """
        return PrefixedName(name=self.name, prefix=self.prefix)


@dataclass(frozen=True)
class RecordedCameraSnapshot:
    """
    Store the sensor properties shared by frames from one camera.
    """

    identity: str
    """
    Recording-local identifier used only by the reader registry.
    """

    name: RecordedName
    """
    Recorded camera annotation name.
    """

    root_frame: RecordedName
    """
    Recorded camera frame name.
    """

    forward_axis: tuple[float, float, float]
    """
    Viewing direction in the camera frame.
    """

    native_model: dict[str, Any]
    """
    Serialized native camera projection model.
    """

    minimum_distance: float
    """
    Nearest usable distance in meters.
    """

    maximum_distance: float
    """
    Farthest usable distance in meters.
    """

    modalities: tuple[str, ...]
    """
    Recorded image modalities.
    """

    @classmethod
    def from_camera(cls, camera: Camera) -> RecordedCameraSnapshot:
        """
        Capture a camera definition without its world identifiers.
        """
        axis = camera.forward_facing_axis.to_np().reshape(-1)
        return cls(
            identity=str(camera.id),
            name=RecordedName.from_name(camera.name),
            root_frame=RecordedName.from_name(camera.root.name),
            forward_axis=tuple(float(value) for value in axis[:3]),
            native_model=krrood_to_json(camera.camera_model),
            minimum_distance=camera.camera_range.minimum_distance,
            maximum_distance=camera.camera_range.maximum_distance,
            modalities=tuple(str(modality) for modality in camera.modalities),
        )

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> RecordedCameraSnapshot:
        """
        Read a recorded camera definition.
        """
        return cls(
            identity=document[CameraSnapshotField.IDENTITY],
            name=RecordedName.from_document(document[CameraSnapshotField.NAME]),
            root_frame=RecordedName.from_document(
                document[CameraSnapshotField.ROOT_FRAME]
            ),
            forward_axis=tuple(document[CameraSnapshotField.FORWARD_AXIS]),
            native_model=document[CameraSnapshotField.NATIVE_MODEL],
            minimum_distance=document[CameraSnapshotField.MINIMUM_DISTANCE],
            maximum_distance=document[CameraSnapshotField.MAXIMUM_DISTANCE],
            modalities=tuple(document[CameraSnapshotField.MODALITIES]),
        )

    def to_document(self) -> dict[str, Any]:
        """
        Write the sensor definition as a Mongo-compatible document.
        """
        return {
            CameraSnapshotField.IDENTITY: self.identity,
            CameraSnapshotField.NAME: self.name.to_document(),
            CameraSnapshotField.ROOT_FRAME: self.root_frame.to_document(),
            CameraSnapshotField.FORWARD_AXIS: list(self.forward_axis),
            CameraSnapshotField.NATIVE_MODEL: self.native_model,
            CameraSnapshotField.MINIMUM_DISTANCE: self.minimum_distance,
            CameraSnapshotField.MAXIMUM_DISTANCE: self.maximum_distance,
            CameraSnapshotField.MODALITIES: list(self.modalities),
        }


@dataclass(frozen=True)
class RecordedCameraPose:
    """
    Store a sampled numeric camera pose and its reference frame.
    """

    matrix: list[list[float]]
    """
    Numeric homogeneous transformation matrix.
    """

    reference_frame: RecordedName | None
    """
    Named frame in which the matrix was sampled.
    """

    @classmethod
    def from_transform(
        cls, transform: HomogeneousTransformationMatrix
    ) -> RecordedCameraPose:
        """
        Capture a pose without serializing frame identifiers.
        """
        reference = transform.reference_frame
        return cls(
            matrix=transform.to_np().tolist(),
            reference_frame=(
                RecordedName.from_name(reference.name)
                if reference is not None
                else None
            ),
        )

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> RecordedCameraPose:
        """
        Read a sampled camera pose.
        """
        reference = document[CameraPoseField.REFERENCE_FRAME]
        return cls(
            matrix=document[CameraPoseField.MATRIX],
            reference_frame=(
                RecordedName.from_document(reference) if reference is not None else None
            ),
        )

    def to_document(self) -> dict[str, Any]:
        """
        Write only numeric pose data and the reference frame name.
        """
        return {
            CameraPoseField.MATRIX: self.matrix,
            CameraPoseField.REFERENCE_FRAME: (
                self.reference_frame.to_document()
                if self.reference_frame is not None
                else None
            ),
        }

    def validated_matrix(self) -> np.ndarray:
        """
        Return a finite homogeneous matrix or reject malformed data.
        """
        try:
            matrix = np.asarray(self.matrix, dtype=float)
        except (TypeError, ValueError) as error:
            raise InvalidCameraObservation(
                reason="pose matrix is not numeric"
            ) from error
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise InvalidCameraObservation(
                reason="pose matrix must be finite and 4 by 4"
            )
        if not np.array_equal(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise InvalidCameraObservation(
                reason="pose matrix has an invalid final row"
            )
        return matrix


# %% Reader state


@dataclass
class RecordedCameraRegistry:
    """
    Keep reconstructed cameras stable during one recording playback.
    """

    cameras: dict[str, tuple[RecordedCameraSnapshot, Camera]] = field(
        default_factory=dict
    )
    """
    Camera definitions and instances indexed by recording-local identity.
    """

    def resolve(
        self,
        snapshot: RecordedCameraSnapshot,
        reference_name: RecordedName | None = None,
    ) -> Camera:
        """
        Return the camera for a stored identity, validating repeated definitions.
        """
        existing = self.cameras.get(snapshot.identity)
        if existing is not None:
            prior_snapshot, camera = existing
            if prior_snapshot != snapshot:
                raise InvalidCameraObservation(
                    reason=f"camera definition changed for '{snapshot.identity}'"
                )
            return camera

        replay_world = world.world_instance()
        root_name = snapshot.root_frame.to_name()
        if replay_world.get_bodies_by_name(root_name):
            raise InvalidCameraObservation(
                reason=f"camera frame '{root_name}' already exists in the runtime world"
            )
        camera_name = snapshot.name.to_name()
        if any(
            camera.name == camera_name
            for camera in replay_world.get_semantic_annotations_by_type(Camera)
        ):
            raise InvalidCameraObservation(
                reason=f"camera '{camera_name}' already exists in the runtime world"
            )
        native_model = krrood_from_json(snapshot.native_model)
        if not isinstance(native_model, CameraModel):
            raise InvalidCameraObservation(reason="native camera model is invalid")
        if (
            len(snapshot.forward_axis) != 3
            or not np.isfinite(snapshot.forward_axis).all()
        ):
            raise InvalidCameraObservation(reason="forward axis is invalid")

        existing_root = replay_world.root
        reference = None
        created_reference = False
        if reference_name is not None and reference_name != snapshot.root_frame:
            bodies = replay_world.get_bodies_by_name(reference_name.to_name())
            if len(bodies) > 1:
                raise InvalidCameraObservation(
                    reason=f"reference frame '{reference_name.to_name()}' is ambiguous"
                )
            reference = bodies[0] if bodies else Body(name=reference_name.to_name())
            created_reference = not bodies
        root = Body(name=root_name)
        camera = Camera(
            name=camera_name,
            root=root,
            forward_facing_axis=Vector3(*snapshot.forward_axis),
            camera_model=native_model,
            camera_range=CameraRange(
                minimum_distance=snapshot.minimum_distance,
                maximum_distance=snapshot.maximum_distance,
            ),
            modalities=tuple(CameraModality(value) for value in snapshot.modalities),
        )
        with replay_world.modify_world():
            if created_reference:
                replay_world.add_body(reference)
                if existing_root is not None:
                    self._connect(replay_world, existing_root, reference)
            replay_world.add_body(root)
            parent = reference if reference is not None else existing_root
            if parent is not None:
                self._connect(replay_world, parent, root)
            replay_world.add_semantic_annotation(camera)
        world.init_world_entity_tracker_from_world(replay_world)
        self.cameras[snapshot.identity] = (snapshot, camera)
        return camera

    @staticmethod
    def _connect(replay_world: Any, parent: Body, child: Body) -> None:
        """
        Attach a reconstructed sensor frame to the reasoning world.
        """
        connection = Connection6DoF.create_with_dofs(
            parent=parent,
            child=child,
            world=replay_world,
            name=PrefixedName(name=f"{child.name.name}_T_{parent.name.name}"),
        )
        replay_world.add_connection(connection)

    def resolve_pose(
        self, pose: RecordedCameraPose, camera: Camera
    ) -> HomogeneousTransformationMatrix:
        """
        Bind a sampled matrix to replay-world reference and camera frames.
        """
        matrix = pose.validated_matrix()
        replay_world = world.world_instance()
        reference = None
        if pose.reference_frame is not None:
            reference_name = pose.reference_frame.to_name()
            bodies = replay_world.get_bodies_by_name(reference_name)
            if len(bodies) > 1:
                raise InvalidCameraObservation(
                    reason=f"reference frame '{reference_name}' is ambiguous"
                )
            if not bodies:
                raise InvalidCameraObservation(
                    reason=f"reference frame '{reference_name}' is missing"
                )
            reference = bodies[0]
        transform = HomogeneousTransformationMatrix(
            data=matrix, reference_frame=reference, child_frame=camera.root
        )
        if reference is not None and reference is not camera.root:
            connection = camera.root.parent_connection
            if connection is None or connection.parent is not reference:
                raise InvalidCameraObservation(
                    reason="camera pose reference frame changed during playback"
                )
            with replay_world.modify_world():
                connection.origin = transform
        return transform
