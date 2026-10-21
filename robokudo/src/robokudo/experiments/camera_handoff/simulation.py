# %% imports

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.experiments.camera_handoff.exceptions import InvalidExperiment
from robokudo.experiments.camera_handoff.perception import CameraObservation
from robokudo.io.semdt_raytracer_camera_interface import SemDTRayTracerCameraInterface
from robokudo.pipeline import Pipeline
from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Scale

if TYPE_CHECKING:
    from robokudo.cas import CAS
    from semantic_digital_twin.world import World

# %% shared-scene cameras


class SceneNames(StrEnum):
    """Scene bodies and the pipelines observing them."""

    TABLE = "table"
    TARGET = "target_cube"
    DISTRACTOR = "blue_cube"
    HEAD = "head_camera"
    HAND = "hand_camera"


class BlackboardKey(StrEnum):
    """Runtime storage required by the standard pipeline lifecycle."""

    OUTPUT_BUFFER = "annotator_output_pipeline_map_buffer"


@dataclass(eq=False)
class SharedSceneCamera(SemDTRayTracerCameraInterface):
    """Render a persistent scene and omit simulator-only views from perception."""

    camera_config: SemDTRayTracerCameraConfig
    """Camera pose, resolution, and rendering settings."""
    world: World
    """Scene shared by both simulated cameras."""
    include_depth: bool = False
    """Whether metric depth is made available to the pipeline."""

    def __post_init__(self) -> None:
        """Initialize the inherited renderer with this camera's configuration."""
        if self.camera_config.resolution <= 0:
            raise InvalidExperiment("Camera resolution must be positive.")
        super().__init__(self.camera_config)

    def _load_runtime_world(self) -> World:
        """Return the persistent scene instead of loading a world descriptor."""
        if self.world.is_empty():
            raise InvalidExperiment("The camera scene is empty.")
        return self.world

    def set_data(self, cas: CAS) -> None:
        """Expose sensor observations without object IDs or scene ground truth."""
        super().set_data(cas)
        for view in (
            CASViews.OBJECT_IMAGE,
            CASViews.OBJECT_COLOR_MAP,
            CASViews.GROUND_TRUTH_WORLD_REFERENCE,
        ):
            cas.views.pop(view, None)
        if not self.include_depth:
            cas.views.pop(CASViews.DEPTH_IMAGE, None)


@dataclass
class CameraPipeline:
    """An independently captured RoboKudo CollectionReader pipeline."""

    camera: SharedSceneCamera
    """Sensor adapter used by this pipeline's CollectionReader."""
    pipeline: Pipeline = field(init=False)
    """Pipeline owning its own CAS and frame counter."""

    def __post_init__(self) -> None:
        """Connect the simulated camera to a dedicated CollectionReader."""
        descriptor = CollectionReaderAnnotator.Descriptor(
            self.camera.camera_config, self.camera
        )
        self.pipeline = Pipeline(self.camera.camera_config.camera_frame)
        self.pipeline.add_child(CollectionReaderAnnotator(descriptor))
        blackboard = Blackboard()
        if not blackboard.exists(BlackboardKey.OUTPUT_BUFFER):
            blackboard.set(BlackboardKey.OUTPUT_BUFFER, AnnotatorOutputPerPipelineMap())
        self.pipeline.setup()

    def capture(self) -> CameraObservation:
        """Capture a fresh image and copy calibration out of the pipeline CAS."""
        self.pipeline.tick_once()
        if self.pipeline.status is not Status.SUCCESS:
            raise InvalidExperiment("A camera pipeline failed to capture an image.")
        cas = self.pipeline.cas
        return CameraObservation(
            cas.get(CASViews.COLOR_IMAGE).copy(),
            cas.get(CASViews.CAMERA_INTRINSIC).intrinsic_matrix.copy(),
            cas.camera_to_world_transform.to_np().copy(),
        )


@dataclass
class CameraPair:
    """Independent cameras observing a single stationary scene."""

    head: CameraPipeline
    """Fixed high-resolution RGB-D camera."""
    hand: CameraPipeline
    """Moving low-resolution monocular camera."""


# %% tabletop scene


@dataclass
class TabletopScene:
    """An axis-aligned cube and a differently colored distractor on a table."""

    cube_edge: float = 0.1
    """Known target edge length in meters."""
    table_width: float = 0.8
    """Table extent along both horizontal axes."""
    table_thickness: float = 0.04
    """Table thickness below the support plane at world z zero."""
    distractor_offset: float = 0.25
    """Horizontal separation of target and distractor."""
    table_color: Color = field(default_factory=lambda: Color(0.6, 0.6, 0.6, 1))
    """Surface color of the table."""
    distractor_color: Color = field(default_factory=lambda: Color(0.2, 0.3, 0.8, 1))
    """Surface color of the distractor cube."""

    def create_world(self, target_color: np.ndarray) -> World:
        """Build one persistent scene containing the target and distractor."""
        descriptor = BaseWorldDescriptor()
        root = descriptor.world.root
        descriptor.build_objects(
            root,
            [
                ObjectSpec(
                    name=SceneNames.TABLE,
                    box_scale=Scale(
                        self.table_width, self.table_width, self.table_thickness
                    ),
                    color=self.table_color,
                    pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=-self.table_thickness / 2, reference_frame=root
                    ),
                ),
                ObjectSpec(
                    name=SceneNames.TARGET,
                    box_scale=Scale(self.cube_edge, self.cube_edge, self.cube_edge),
                    color=Color(
                        *[
                            float(channel) / np.iinfo(np.uint8).max
                            for channel in target_color[::-1]
                        ],
                        1,
                    ),
                    pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=self.cube_edge / 2, reference_frame=root
                    ),
                ),
                ObjectSpec(
                    name=SceneNames.DISTRACTOR,
                    box_scale=Scale(self.cube_edge, self.cube_edge, self.cube_edge),
                    color=self.distractor_color,
                    pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=self.distractor_offset,
                        z=self.cube_edge / 2,
                        reference_frame=root,
                    ),
                ),
            ],
        )
        return descriptor.world
