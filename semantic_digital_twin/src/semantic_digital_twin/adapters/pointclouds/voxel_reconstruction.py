from dataclasses import dataclass, field
from typing import Optional, Self, List, Tuple

import numpy as np
import open3d as o3d
from numpy._typing import NDArray

from semantic_digital_twin.adapters.pointclouds.processor import (
    PointCloudProcessor,
    OutlierRemoval,
)


@dataclass
class MorphologicalClosing:
    """
    Parameters to apply morphological closing in voxel space before building cube mesh
    """

    radius: int = field(default=1)
    """
    Neighborhood radius (in voxels) for morphological closing
    """

    iterations: int = field(default=1)
    """
    Number of morphological closing iterations
    """

    def apply(self, occupancy_grid: NDArray) -> NDArray:
        from scipy.ndimage import binary_closing

        radius = self.radius
        iterations = self.iterations
        size = 2 * radius + 1
        structure = np.ones((size, size, size), dtype=bool)
        occupancy_grid = binary_closing(
            occupancy_grid, structure=structure, iterations=iterations
        )
        return occupancy_grid


@dataclass
class VoxelProcessor(PointCloudProcessor):
    """
    Constructs a mesh from a point cloud using voxelization and marching cubes.
    """

    voxel_size: float = field(default=0.02)
    """
    Voxel size for voxelization method
    """

    closing_algorithm: Optional[MorphologicalClosing] = None
    """
    Parameters for morphological closing. If None, no morphological closing is applied.
    """

    @classmethod
    def from_pts_file(
        cls,
        pts_file_path: str,
        outlier_removal: Optional[OutlierRemoval] = None,
        closing_algorithm: Optional[MorphologicalClosing] = None,
        **kwargs,
    ) -> Self:
        """
        Creates a PointCloudProcessor from a .pts file.
        :param pts_file_path: Path to the .pts file.
        :param outlier_removal: Parameters for statistical outlier removal. If None, no outlier removal is applied.
        :param closing_algorithm: Parameters for morphological closing. If None, no morphological closing is applied.

        :return: PointCloudProcessor instance.
        """
        processor = super().from_pts_file(pts_file_path)
        processor.closing_algorithm = closing_algorithm
        return processor

    def construct_mesh(self):
        """
        Constructs a mesh from the point cloud using voxelization and marching cubes.
        1. Voxelizes the point cloud.
        2. Constructs an occupancy grid.
        3. (Optional) Applies morphological closing to the occupancy grid.
        4. Applies the marching cubes algorithm to extract the mesh.
        5. Returns the resulting mesh.
        """

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            self.point_cloud_data,
            voxel_size=self.voxel_size,
        )
        voxels = voxel_grid.get_voxels()
        voxel_size = float(voxel_grid.voxel_size)

        if len(voxels) == 0:
            raise RuntimeError("Voxel grid is empty; try decreasing voxel_size.")

        occupancy_grid = self._construct_occupancy_grid(voxels)

        if self.closing_algorithm is not None:
            occupancy_grid = self.closing_algorithm.apply(occupancy_grid)

        vertices, faces, normals = self._apply_marching_cubes(
            voxel_size, occupancy_grid
        )

        # compensate for min_idx offset and add origin
        indices = np.array([v.grid_index for v in voxels], dtype=int)
        min_idx = indices.min(axis=0)
        origin = np.asarray(voxel_grid.origin, dtype=float)
        vertices += min_idx.astype(float) * voxel_size
        vertices += origin

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        return mesh

    @staticmethod
    def _construct_occupancy_grid(voxels: List[o3d.geometry.Voxel]) -> NDArray:
        """
        Constructs a 3D occupancy grid from the list of voxels.

        :param voxels: List of voxels to construct the occupancy grid from.

        :return: Occupancy grid as a boolean array.
        """
        indices = np.array([v.grid_index for v in voxels], dtype=int)
        min_idx = indices.min(axis=0)
        max_idx = indices.max(axis=0)
        shape = (max_idx - min_idx + 1).astype(int)

        occupancy_grid = np.zeros(shape, dtype=bool)
        shifted_coordinates = indices - min_idx
        occupancy_grid[
            shifted_coordinates[:, 0],
            shifted_coordinates[:, 1],
            shifted_coordinates[:, 2],
        ] = True

        return occupancy_grid

    @staticmethod
    def _apply_marching_cubes(
        voxel_size: float, occupancy_grid: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Applies the marching cubes algorithm to the occupancy grid to extract the mesh.

        :param voxel_size: Voxel size of the occupancy grid.
        :param occupancy_grid: Occupancy grid as a boolean array.

        """
        from skimage import measure

        # marching cubes expects a scalar field; bool -> uint8
        volume = occupancy_grid.astype(np.uint8)

        # spacing in world units per axis
        spacing = (voxel_size, voxel_size, voxel_size)

        verts, faces, normals, values = measure.marching_cubes(
            volume,
            level=0.5,
            spacing=spacing,
        )

        return verts, faces, normals
