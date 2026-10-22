# %% imports

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, pi

import numpy as np
import trimesh
from PIL import Image
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.experiments.camera_handoff.experiment import (
    ApproachEvaluator,
    CameraHandoffExperiment,
    ComparisonResult,
)
from robokudo.experiments.camera_handoff.feature_tracking import FeatureTargetTracker
from robokudo.experiments.camera_handoff.recording import ApproachFiles, ImageRecorder
from robokudo.experiments.camera_handoff.simulation import (
    CameraPair,
    SceneNames,
    SharedSceneCamera,
)
from robokudo.experiments.camera_handoff.textured_models import TexturedModel
from robokudo.experiments.camera_handoff.tabletop import TabletopInitializer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.shape_collection import ShapeCollection

if TYPE_CHECKING:
    from pathlib import Path

    from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
    from semantic_digital_twin.datastructures.field_of_view import FieldOfView
    from semantic_digital_twin.spatial_computations.raytracer import RayTracer


# %% texture rendering


@dataclass(eq=False)
class TexturedSceneCamera(SharedSceneCamera):
    """
    Require textured RGB output instead of accepting flat-color fallback.
    """

    def _try_render_trimesh_rgb(
        self,
        ray_tracer: RayTracer,
        camera_to_world: HomogeneousTransformationMatrix,
        resolution: CameraResolution,
        field_of_view: FieldOfView,
    ) -> np.ndarray:
        """
        Render surface materials without an OpenGL display or flat-color fallback.
        """
        origins, directions, pixels = ray_tracer.create_camera_rays(
            camera_to_world, resolution, field_of_view
        )
        rgb = np.zeros((resolution.width, resolution.height, 3), dtype=np.uint8)
        nearest = np.full(len(origins), np.inf)
        for node in ray_tracer.scene.graph.nodes_geometry:
            world_T_geometry, geometry_name = ray_tracer.scene.graph[node]
            geometry = ray_tracer.scene.geometry[geometry_name]
            geometry_T_world = np.linalg.inv(world_T_geometry)
            local_origins = (
                origins @ geometry_T_world[:3, :3].T + geometry_T_world[:3, 3]
            )
            local_directions = directions @ geometry_T_world[:3, :3].T
            points, ray_indices, faces = geometry.ray.intersects_location(
                local_origins, local_directions, multiple_hits=False
            )
            world_points = points @ world_T_geometry[:3, :3].T + world_T_geometry[:3, 3]
            distances = np.linalg.norm(world_points - origins[ray_indices], axis=1)
            valid = (
                (distances < nearest[ray_indices])
                & (distances >= self.camera_config.min_distance)
                & (distances <= self.camera_config.max_distance)
            )
            if not np.any(valid):
                continue
            points, ray_indices, faces, distances = (
                points[valid],
                ray_indices[valid],
                faces[valid],
                distances[valid],
            )
            if geometry.visual.kind == "texture":
                barycentric = trimesh.triangles.points_to_barycentric(
                    geometry.triangles[faces], points
                )
                coordinates = np.einsum(
                    "ij,ijk->ik", barycentric, geometry.visual.uv[geometry.faces[faces]]
                )
                colors = geometry.visual.material.to_color(coordinates)[:, :3]
            else:
                colors = geometry.visual.face_colors[faces, :3]
            image_pixels = pixels[ray_indices]
            rgb[image_pixels[:, 0], image_pixels[:, 1]] = colors
            nearest[ray_indices] = distances
        return rgb[:, :, ::-1].copy()


# %% textured approach


@dataclass
class TexturedHandoffExperiment(CameraHandoffExperiment):
    """
    Use a textured model with head initialization and hand-image tracking.
    """

    model: TexturedModel = field(default_factory=TexturedModel)
    """
    Textured object replacing the cube target.
    """

    feature_tracker: FeatureTargetTracker = field(default_factory=FeatureTargetTracker)
    """
    Hand-image outline and temporal feature tracking.
    """

    head_initializer: TabletopInitializer = field(default_factory=TabletopInitializer)
    """
    Tabletop segmentation and cluster bounding-box pose initialization.
    """

    head_distance: float = 0.65
    """
    Head-camera distance from the table center along negative world y.
    """

    head_elevation: float = 0.40
    """
    Head-camera height above the support plane.
    """

    def create_cameras(self) -> CameraPair:
        """
        Replace :meth:`CameraHandoffExperiment.create_cameras`' cube with a model.
        """
        cameras = super().create_cameras()
        shape = self.model.load()
        bounds = self.model.measure_bounds(shape)
        world = cameras.head.camera.world
        body = world.get_body_by_name(SceneNames.TARGET)
        with world.modify_world():
            body.visual = ShapeCollection([shape])
            body.collision = ShapeCollection([shape])
            body.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-float(bounds[0, 2]), reference_frame=world.root
                )
            )
        center_height = float((bounds[1, 2] - bounds[0, 2]) / 2)
        for camera in (cameras.head, cameras.hand):
            config = camera.camera.camera_config
            config.rgb_mode = "trimesh"
            adapter = TexturedSceneCamera(config, world, camera.camera.include_depth)
            camera.camera = adapter
            camera.pipeline.children[0].descriptor.parameters.camera_interface = adapter
        head_config = cameras.head.camera.camera_config
        head_config.camera_y = -self.head_distance
        head_config.camera_z = self.head_elevation
        head_config.camera_roll = -pi / 2 - atan2(
            self.head_elevation - center_height, self.head_distance
        )
        cameras.hand.camera.camera_config.camera_z = center_height
        self.head_initializer.configure(cameras.head)
        return cameras

    def run(self, output_directory: Path) -> ComparisonResult:
        """
        Record :meth:`CameraHandoffExperiment.run` with model-feature tracking.
        """
        cameras = self.create_cameras()
        head = cameras.head.capture()
        output_directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(head.color[:, :, ::-1]).save(
            output_directory / self.model.asset.preview_filename
        )
        selected = self.head_initializer.estimate(cameras.head.pipeline.cas, head)
        target = self.feature_tracker.acquire(
            head,
            cameras.head.pipeline.cas.get(CASViews.DEPTH_IMAGE).astype(float) / 1000,
            selected.estimate.dimensions,
            self.position_bias,
            segmented_target=selected,
        )
        recorder = ImageRecorder(output_directory)
        recorder.write_initial_estimate(cameras.head.pipeline.cas, head, selected)
        bounds = self.model.measure_bounds(self.model.load())
        world = cameras.hand.camera.world
        truth_center = (
            world.compute_forward_kinematics_np(
                world.root, world.get_body_by_name(SceneNames.TARGET)
            )[:3, 3]
            + (bounds[0] + bounds[1]) / 2
        )
        evaluator = ApproachEvaluator(truth_center)
        results = []
        for approach in (ApproachFiles.HEAD_ONLY, ApproachFiles.FEEDBACK):
            self.feature_tracker.previous = None
            results.append(
                self.approach(
                    cameras.hand,
                    target,
                    recorder,
                    approach,
                    evaluator,
                    self.feature_tracker,
                )
            )
        comparison = ComparisonResult(*results)
        recorder.write_comparison(comparison, head)
        return comparison
