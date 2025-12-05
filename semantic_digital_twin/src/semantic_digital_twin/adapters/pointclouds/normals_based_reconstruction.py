from abc import ABC
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from semantic_digital_twin.adapters.pointclouds.processor import PointCloudProcessor


@dataclass
class NormalsBasedProcessor(PointCloudProcessor, ABC):
    """
    Base class for point cloud processors that require normals.
    """

    radius: float = field(default=0.05)
    """
    Radius used for normal estimation
    """

    max_nearest_neighbors: int = field(default=30)
    """
    Max neighbors used for normal estimation
    """

    orient_normals_number: int = field(default=50)
    """
    Number of neighbors used for normal orientation
    """

    def __post_init__(self):
        self.point_cloud_data.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius,
                max_nn=self.max_nearest_neighbors,
            )
        )
        self.point_cloud_data.orient_normals_consistent_tangent_plane(
            k=self.orient_normals_number
        )


@dataclass
class PoissonReconstructionProcessor(NormalsBasedProcessor):
    """
    Constructs a mesh from a point cloud using Poisson surface reconstruction.
    """

    depth: int = field(default=9)
    """
    Poisson octree depth (higher = more detail & more noise)
    """

    scale: float = field(default=1.1)
    """
    Poisson scale parameter (bounding sphere scaling)
    """

    trim_quantile: float = field(default=0.02)
    """
    Quantile for density-based trimming (0.0–0.2 is typical)
    """

    linear_fit: bool = field(default=False)
    """
    If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.
    """

    def construct_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.point_cloud_data,
            depth=self.depth,
            scale=self.scale,
            linear_fit=self.linear_fit,
        )

        densities_np = np.asarray(densities)
        trim_quantile = self.trim_quantile
        if trim_quantile > 0.0:
            threshold = np.quantile(densities_np, trim_quantile)
            mask = densities_np < threshold
            mesh.remove_vertices_by_mask(mask)

        bbox = self.point_cloud_data.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        return mesh


@dataclass
class BallPivotingProcessor(NormalsBasedProcessor):
    """
    Constructs a mesh from a point cloud using the Ball Pivoting Algorithm.
    """

    ball_pivoting_average_nearest_neighbor_factor: float = 2.0
    """
    Multiplier for average NN distance to define BPA radii.
    """

    def construct_mesh(self) -> o3d.geometry.TriangleMesh:
        distances = self.point_cloud_data.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances))
        r1 = self.ball_pivoting_average_nearest_neighbor_factor * avg_dist
        r2 = 2.0 * r1
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            self.point_cloud_data,
            o3d.utility.DoubleVector([r1, r2]),
        )
        return mesh
