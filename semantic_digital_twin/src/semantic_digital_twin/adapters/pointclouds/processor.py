from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
from typing_extensions import Self


@dataclass
class OutlierRemoval:
    """
    Parameters for statistical outlier removal
    """

    number_of_neighbors: int
    """
    Number of neighbors for statistical outlier removal
    """

    std_ratio: float
    """
    Std_ratio for statistical outlier removal
    """

    def remove_outliers(
        self, point_cloud_data: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        cl, ind = point_cloud_data.remove_statistical_outlier(
            nb_neighbors=self.number_of_neighbors,
            std_ratio=self.std_ratio,
        )
        return point_cloud_data.select_by_index(ind)


@dataclass
class PointCloudProcessor(ABC):
    """
    Base class for point cloud processors that construct meshes from point clouds.
    """

    point_cloud_data: o3d.geometry.PointCloud
    """
    Input point cloud data.
    """

    outlier_removal: Optional[OutlierRemoval] = None
    """
    Parameters for statistical outlier removal. If None, no outlier removal is applied.
    """

    @abstractmethod
    def construct_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Constructs a mesh from the point cloud.
        """

    @classmethod
    def from_pts_file(
        cls,
        pts_file_path: str,
        outlier_removal: Optional[OutlierRemoval] = None,
        **kwargs,
    ) -> Self:
        """
        Creates a PointCloudProcessor from a .pts file.
        :param pts_file_path: Path to the .pts file.
        :param outlier_removal: Parameters for statistical outlier removal. If None, no outlier removal is applied.

        :return: PointCloudProcessor instance.
        """
        pts = np.loadtxt(pts_file_path)  # expects x y z per line
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] < 3:
            raise ValueError(
                "Input .pts file must have at least 3 columns (x y z) per line."
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        return cls(point_cloud_data=pcd, outlier_removal=outlier_removal)

    def export_as_obj_file(self, output_path: str, remove_duplication: bool = True):
        """
        Exports the constructed mesh as an OBJ file.
        """
        mesh = self.construct_mesh()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()

        if remove_duplication:
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()

        mesh.orient_triangles()
        mesh.compute_vertex_normals()
        success = o3d.io.write_triangle_mesh(output_path, mesh)

        if not success:
            raise RuntimeError(f"Failed to write OBJ to {output_path}")
