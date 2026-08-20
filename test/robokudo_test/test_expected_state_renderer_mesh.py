from pathlib import Path

import numpy as np
import open3d as o3d

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.expected_state_renderer import ExpectedStateRendererAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.analysis_engines.semdt_raytracer import AnalysisEngine
from robokudo.descriptors.worlds.world_semdt_raytracer_needle import (
    WorldDescriptor as NeedleWorldDescriptor,
)
from robokudo.pipeline import Pipeline
from robokudo.utils.pose_refinement import is_refinement_result_better
from robokudo.utils.semdt_ground_truth import body_support_extent_along_normal
from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Mesh


def test_resolve_ground_truth_object_model_accepts_mesh_body():
    mesh_path = _needle_mesh_path()
    cas = _cas_with_mesh_body(mesh_path)
    annotator = _expected_state_renderer_with_cas(cas)

    model = annotator._resolve_ground_truth_object_model()

    assert model is not None
    assert model.shape_type == "mesh"
    assert model.mesh_path == mesh_path
    assert model.mesh_origin is not None
    assert model.mesh_scale is not None


def test_build_expected_world_preserves_mesh_shape():
    mesh_path = _needle_mesh_path()
    cas = _cas_with_mesh_body(mesh_path)
    annotator = _expected_state_renderer_with_cas(cas)
    model = annotator._resolve_ground_truth_object_model()

    expected_world = annotator._build_expected_world(
        object_center_world=np.array([-1.0, 1.0, 0.8], dtype=np.float64),
        gt_object_model=model,
    )

    bodies = expected_world.get_bodies_by_name("expected_needle")
    assert len(bodies) == 1
    mesh_shapes = [shape for shape in bodies[0].visual if isinstance(shape, Mesh)]
    assert len(mesh_shapes) == 1
    assert Path(mesh_shapes[0].filename) == mesh_path


def test_create_o3d_mesh_for_gt_object_model_uses_mesh_geometry():
    mesh_path = _needle_mesh_path()
    cas = _cas_with_mesh_body(mesh_path)
    annotator = _expected_state_renderer_with_cas(cas)
    model = annotator._resolve_ground_truth_object_model()

    assert model.shape_type == "mesh"
    mesh = annotator._create_o3d_mesh_for_gt_object_model(model)

    assert isinstance(mesh, o3d.geometry.TriangleMesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0


def test_needle_world_descriptor_uses_needle_mesh_collision():
    world_descriptor = NeedleWorldDescriptor()

    bodies = world_descriptor.world.get_bodies_by_name("needle")

    assert len(bodies) == 1
    assert len(bodies[0].collision) == 1
    assert isinstance(bodies[0].collision[0], Mesh)
    assert Path(bodies[0].collision[0].filename) == _needle_mesh_path()


def test_mesh_support_extent_uses_mesh_vertices():
    world_descriptor = NeedleWorldDescriptor()
    body = world_descriptor.world.get_bodies_by_name("needle")[0]
    mesh_shape = body.collision[0]

    support_extent = body_support_extent_along_normal(
        body=body,
        normal_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    assert isinstance(mesh_shape, Mesh)
    assert support_extent is not None
    assert np.isclose(support_extent, -mesh_shape.local_frame_bounding_box.min_z)


def test_pose_prior_prevents_small_silhouette_gain_from_winning_large_drift():
    assert (
        is_refinement_result_better(
            candidate_outline_score=0.45,
            candidate_pixel_error=0.5,
            incumbent_outline_score=0.42,
            incumbent_pixel_error=0.5,
            centroid_scale_px=25.0,
            centroid_weight=0.25,
            candidate_prior_distance_m=0.06,
            incumbent_prior_distance_m=0.0,
            prior_scale_m=0.06,
            prior_weight=0.15,
        )
        is False
    )


def test_semdt_raytracer_analysis_engine_targets_needle_mesh_world():
    pipeline = AnalysisEngine().implementation()

    collection_reader = next(
        child
        for child in pipeline.children
        if isinstance(child, CollectionReaderAnnotator)
    )
    expected_state_renderer = next(
        child
        for child in pipeline.children
        if isinstance(child, ExpectedStateRendererAnnotator)
    )
    cluster_extractor = next(
        child
        for child in pipeline.children
        if isinstance(child, PointCloudClusterExtractor)
    )
    expected_state_parameters = expected_state_renderer.descriptor.parameters
    cluster_parameters = cluster_extractor.descriptor.parameters

    assert (
        collection_reader.descriptor.parameters.camera_config.world_descriptor_name
        == "world_semdt_raytracer_needle"
    )
    assert expected_state_parameters.target_classname == "needle"
    assert expected_state_parameters.ground_truth_body_name == "needle"
    assert expected_state_parameters.random_offset_translation_m == 0.12
    assert expected_state_parameters.candidate_pose_sample_count == 90
    assert expected_state_parameters.candidate_pose_sampling_radius_m == 0.12
    assert expected_state_parameters.refinement_convergence_pixel_error == 1.0
    assert (
        expected_state_parameters.refinement_min_outline_score_for_convergence == 0.30
    )
    assert expected_state_parameters.refinement_score_delta_pixel_error_gate == 1.0
    assert expected_state_parameters.pose_prior_translation_scale_m == 0.06
    assert expected_state_parameters.pose_prior_weight == 0.8
    assert cluster_parameters.dbscan_min_cluster_count == 8
    assert cluster_parameters.min_cluster_count == 20
    assert cluster_parameters.min_on_plane_point_count == 20
    assert cluster_parameters.eps == 0.03


def _needle_mesh_path() -> Path:
    return Path(__file__).resolve().parents[2] / "robokudo" / "Needle.stl"


def _cas_with_mesh_body(mesh_path: Path) -> CAS:
    world_descriptor = BaseWorldDescriptor()
    root = world_descriptor.world.root
    world_descriptor.build_objects(
        root,
        [
            ObjectSpec(
                name="needle",
                mesh_path=mesh_path,
                color=Color(0.85, 0.85, 0.08, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.0,
                    y=1.0,
                    z=0.8,
                    yaw=0.2,
                    reference_frame=root,
                ),
            )
        ],
    )
    cas = CAS()
    cas.set_ref(CASViews.GROUND_TRUTH_WORLD_REFERENCE, world_descriptor.world)
    return cas


def _expected_state_renderer_with_cas(cas: CAS) -> ExpectedStateRendererAnnotator:
    descriptor = ExpectedStateRendererAnnotator.Descriptor()
    descriptor.parameters.ground_truth_body_name = "needle"
    descriptor.parameters.expected_object_name = "expected_needle"
    annotator = ExpectedStateRendererAnnotator(descriptor=descriptor)
    pipeline = Pipeline("ExpectedStateRendererMeshTestPipeline")
    pipeline.cas = cas
    pipeline.add_child(annotator)
    return annotator
