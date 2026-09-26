"""
Verify that recorded camera observations replay into a fresh world.
"""

from copy import deepcopy

import numpy as np
import pytest

from robokudo import world as rk_world
from robokudo.cas import CASViews
from robokudo.exceptions import InvalidCameraObservation
from robokudo.io.camera_replay import (
    CameraPoseField,
    CameraSnapshotField,
    RecordedCameraRegistry,
)
from robokudo.io.cas_view_codecs import CameraObservationField, ViewPayload
from robokudo.io.storage import Storage
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    CameraRange,
    FieldOfViewCameraModel,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world_description.world_entity import Body

# %% Recorded observations


def _observation(
    camera_name: str = "recorded_camera",
    with_pose: bool = True,
) -> CameraObservation:
    """
    Build one independent recorded sensor observation.
    """
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=525.0,
        focal_length_y=525.0,
        principal_point_x=319.5,
        principal_point_y=239.5,
    )
    root = Body(name=PrefixedName(name=f"{camera_name}_frame"))
    reference = Body(name=PrefixedName(name="recorded_map"))
    camera = Camera(
        name=PrefixedName(name=camera_name),
        root=root,
        forward_facing_axis=Vector3.Z(),
        camera_model=FieldOfViewCameraModel(
            view=FieldOfView(), image_resolution=model.resolution
        ),
        camera_range=CameraRange(minimum_distance=0.2, maximum_distance=12.0),
        modalities=(CameraModality.COLOR, CameraModality.DEPTH),
    )
    return CameraObservation(
        camera=camera,
        effective_camera_model=model,
        world_T_camera=(
            HomogeneousTransformationMatrix(reference_frame=reference, child_frame=root)
            if with_pose
            else None
        ),
        timestamp_nanoseconds=123,
    )


def test_recorded_camera_replays_in_fresh_world():
    """
    Reconstruct camera metadata without resolving recorded entity identifiers.
    """
    rk_world.init_world_with_entity_tracker()
    inferred_body = Body(name=PrefixedName(name="recorded_inferred_object"))
    with rk_world.world_instance().modify_world():
        rk_world.world_instance().add_body(inferred_body)
    observation = _observation()
    document = Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, observation)

    rk_world.init_world_with_entity_tracker()
    _, replayed = Storage.decode_view_document(document)

    assert replayed.camera is not observation.camera
    assert replayed.camera.id != observation.camera.id
    assert replayed.camera.name == observation.camera.name
    assert replayed.camera.root.name == observation.camera.root.name
    assert replayed.camera.camera_model == observation.camera.camera_model
    assert replayed.camera.camera_range == observation.camera.camera_range
    assert replayed.camera.modalities == observation.camera.modalities
    np.testing.assert_array_equal(
        replayed.camera.forward_facing_axis.to_np(),
        observation.camera.forward_facing_axis.to_np(),
    )
    assert replayed.effective_camera_model == observation.effective_camera_model
    assert replayed.timestamp_nanoseconds == observation.timestamp_nanoseconds
    assert (
        replayed.world_T_camera.reference_frame.name
        == observation.world_T_camera.reference_frame.name
    )
    assert replayed.world_T_camera.child_frame is replayed.camera.root
    np.testing.assert_array_equal(
        replayed.world_T_camera.to_np(), observation.world_T_camera.to_np()
    )
    assert set(rk_world.world_instance().state.keys()).isdisjoint(
        {observation.camera.id, observation.camera.root.id}
    )
    assert rk_world.world_instance().get_bodies_by_name(inferred_body.name) == []


def test_recorded_camera_identity_is_stable_across_frames():
    """
    Reuse one reconstructed camera while frame metadata changes.
    """
    rk_world.init_world_with_entity_tracker()
    observation = _observation()
    document = Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, observation)
    registry = RecordedCameraRegistry()

    _, first = Storage.decode_view_document(document, registry)
    next_document = deepcopy(document)
    next_payload = ViewPayload.from_document(next_document).payload
    next_payload[CameraObservationField.TIMESTAMP] += 1
    next_payload[CameraObservationField.WORLD_T_CAMERA][CameraPoseField.MATRIX][0][
        3
    ] = 0.5
    _, second = Storage.decode_view_document(next_document, registry)

    assert second.camera is first.camera
    assert second.world_T_camera.child_frame is first.camera.root
    assert second.timestamp_nanoseconds == first.timestamp_nanoseconds + 1
    assert second.world_T_camera.to_np()[0, 3] == 0.5


def test_two_recorded_cameras_remain_distinct():
    """
    Resolve separate recording identities to separate camera annotations.
    """
    rk_world.init_world_with_entity_tracker()
    registry = RecordedCameraRegistry()
    first = _observation(camera_name="first_camera")
    second = _observation(camera_name="second_camera")

    _, first_replayed = Storage.decode_view_document(
        Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, first), registry
    )
    _, second_replayed = Storage.decode_view_document(
        Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, second), registry
    )

    assert first_replayed.camera is not second_replayed.camera
    assert first_replayed.camera.root is not second_replayed.camera.root


def test_recorded_camera_without_pose_replays_with_unrelated_runtime_entity():
    """
    Preserve a missing pose without importing recorded world entities.
    """
    rk_world.init_world_with_entity_tracker()
    unrelated = Body(name=PrefixedName(name="unrelated"))
    replay_world = rk_world.world_instance()
    with replay_world.modify_world():
        replay_world.add_body(unrelated)
    observation = _observation(with_pose=False)

    _, replayed = Storage.decode_view_document(
        Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, observation)
    )

    assert replayed.world_T_camera is None
    assert replayed.camera.root.parent_connection.parent is unrelated
    assert replay_world.get_body_by_name(unrelated.name) is unrelated


def test_changed_recorded_camera_definition_is_rejected():
    """
    Reject conflicting definitions for one recording identity.
    """
    rk_world.init_world_with_entity_tracker()
    observation = _observation()
    document = Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, observation)
    registry = RecordedCameraRegistry()
    Storage.decode_view_document(document, registry)
    changed_document = Storage.encode_view_document(
        CASViews.CAMERA_OBSERVATION, observation
    )
    ViewPayload.from_document(changed_document).payload[CameraObservationField.CAMERA][
        CameraSnapshotField.MINIMUM_DISTANCE
    ] = 0.5

    with pytest.raises(InvalidCameraObservation):
        Storage.decode_view_document(changed_document, registry)


def test_existing_runtime_camera_frame_is_rejected():
    """
    Avoid guessing whether an existing frame belongs to a recorded camera.
    """
    rk_world.init_world_with_entity_tracker()
    observation = _observation()
    runtime_frame = Body(name=observation.camera.root.name)
    with rk_world.world_instance().modify_world():
        rk_world.world_instance().add_body(runtime_frame)

    with pytest.raises(InvalidCameraObservation):
        Storage.decode_view_document(
            Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, observation)
        )


@pytest.mark.parametrize("matrix", [[[1.0]], [[float("nan")] * 4] * 4])
def test_malformed_recorded_camera_pose_is_rejected(matrix):
    """
    Reject a malformed numeric pose using a RoboKudo exception.
    """
    rk_world.init_world_with_entity_tracker()
    document = Storage.encode_view_document(CASViews.CAMERA_OBSERVATION, _observation())
    ViewPayload.from_document(document).payload[CameraObservationField.WORLD_T_CAMERA][
        CameraPoseField.MATRIX
    ] = matrix

    with pytest.raises(InvalidCameraObservation):
        Storage.decode_view_document(document)
