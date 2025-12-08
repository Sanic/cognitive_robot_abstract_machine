import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from typing_extensions import Self


class PointCloudLoadError(Exception):
    """Raised when a point cloud file cannot be loaded or parsed."""


class ReconstructionError(Exception):
    """Raised when a mesh reconstruction fails."""


@dataclass
class OutlierRemoval:
    """
    Parameters for statistical outlier removal
    """

    number_of_neighbors: int = field(default=20)
    """
    Number of neighbors for statistical outlier removal
    """

    std_ratio: float = field(default=5.0)
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
class ResidualComputationConfig:

    max_residual: Optional[float] = None
    """
    Maximum residual value for color mapping. If None, auto‑computed.
    """

    colormap: str = "viridis"
    """
    Colormap to use for residual visualization.
    """

    threshold: Optional[float] = None
    """
    If set, hide mesh triangles with mean residual above this threshold.
    """

    hide_above_threshold: bool = False
    """
    Whether to hide triangles above the threshold.
    """

    subsample_voxel: float = field(default=0.0)
    """
    Voxel downsample the point cloud before computing residuals. If 0.0, no downsampling is applied.
    """

    report_area_over_threshold: bool = True
    """
    Whether to print the area of triangles over the threshold.
    """


@dataclass
class ViridisScalarColorizer:
    """Maps scalar values in [0,1] to RGB using a fixed LUT."""

    _look_up_table: np.ndarray = field(init=False)

    def __post_init__(self):
        base = np.array(
            [
                [0.267, 0.004, 0.329],
                [0.282, 0.140, 0.457],
                [0.254, 0.265, 0.531],
                [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.748, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
                [0.993, 0.906, 0.144],
                [0.993, 0.773, 0.188],
                [0.990, 0.636, 0.285],
                [0.984, 0.503, 0.384],
                [0.969, 0.382, 0.494],
                [0.940, 0.278, 0.607],
            ],
            dtype=float,
        )
        x = np.linspace(0.0, 1.0, base.shape[0])
        xi = np.linspace(0.0, 1.0, 256)
        look_up_table = np.zeros((256, 3), dtype=float)
        for c in range(3):
            look_up_table[:, c] = np.interp(xi, x, base[:, c])
        self._look_up_table = look_up_table

    def map(
        self, values: np.ndarray, max_vertices: Optional[float] = None
    ) -> np.ndarray:
        if values.size == 0:
            return np.zeros((0, 3), dtype=float)
        if max_vertices is None or not np.isfinite(max_vertices) or max_vertices <= 0.0:
            # Robust default: 95th percentile to reduce outlier influence
            max_vertices = (
                float(np.percentile(values[np.isfinite(values)], 95.0))
                if np.any(np.isfinite(values))
                else 1.0
            )
            if max_vertices <= 0.0:
                max_vertices = 1.0
        vals = np.asarray(values, dtype=float)
        vals[~np.isfinite(vals)] = max_vertices
        t = np.clip(vals / max_vertices, 0.0, 1.0)
        idx = np.minimum(
            (t * (len(self._look_up_table) - 1)).astype(int),
            len(self._look_up_table) - 1,
        )
        return self._look_up_table[idx]


@dataclass
class PointCloudProcessor(ABC):
    """
    Base class for point cloud processors that construct meshes from point clouds.
    """

    point_cloud_name: str = "PointCloud"

    point_cloud_data: o3d.geometry.PointCloud = field(kw_only=True)
    """
    Input point cloud data.
    """

    outlier_removal: Optional[OutlierRemoval] = None
    """
    Parameters for statistical outlier removal. If None, no outlier removal is applied.
    """

    residual_computation_config: ResidualComputationConfig = field(
        default_factory=ResidualComputationConfig
    )
    """
    Configuration for residual computation and visualization.
    """

    def __post_init__(self):
        if self.outlier_removal:
            self.point_cloud_data = self.outlier_removal.remove_outliers(
                self.point_cloud_data
            )

    def compute_mesh(
        self, remove_duplication: bool = True
    ) -> o3d.geometry.TriangleMesh:
        mesh = self._compute_mesh()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()

        if remove_duplication:
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()

        return mesh

    @abstractmethod
    def _compute_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Constructs a mesh from the point cloud.
        """

    @classmethod
    def from_pts_file(
        cls,
        pts_file_path: Path,
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

    @classmethod
    def from_nth_in_directory(
        cls,
        directory: Path,
        n: int,
        pattern: str = "*.pts",
        sort_by_name: bool = True,
        reverse: bool = False,
        outlier_removal: Optional[OutlierRemoval] = None,
        **kwargs,
    ) -> Self:
        """
        Load the n-th ``.pts`` file from a directory.

        This helper lists files matching ``pattern`` in ``directory``, sorts them
        deterministically, picks the n-th item (1-based), and loads it using ``from_pts_file``.

        :param directory: Directory to search for files.
        :param n: 1-based index of the file to load.
        :param pattern: Glob pattern to match files (default: ``*.pts``).
        :param sort_by_name: If True, sort files by name; otherwise, sort by
        modification time (default: True).
        :param reverse: If True, reverse the sort order (default: False).
        :param outlier_removal: Parameters for statistical outlier removal. If None, no outlier removal is applied.
        """
        if not directory.exists() or not directory.is_dir():
            raise PointCloudLoadError(
                f"Directory not found or not a directory: {directory}"
            )

        files = list(directory.glob(pattern))
        if not files:
            raise PointCloudLoadError(
                f"No files matching pattern '{pattern}' found in: {directory}"
            )

        if sort_by_name:
            files.sort(key=lambda p: p.name.lower(), reverse=reverse)
        else:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=reverse)

        if n < 1 or n > len(files):
            raise PointCloudLoadError(
                f"Requested index {n} is out of range for {len(files)} file(s) in {directory}"
            )

        target = files[n - 1]
        return cls.from_pts_file(target, outlier_removal=outlier_removal)

    def export_as_obj_file(self, output_path: str, remove_duplication: bool = True):
        """
        Exports the constructed mesh as an OBJ file.
        """
        mesh = self.compute_mesh(remove_duplication=remove_duplication)

        mesh.orient_triangles()
        mesh.compute_vertex_normals()
        success = o3d.io.write_triangle_mesh(output_path, mesh)

        if not success:
            raise RuntimeError(f"Failed to write OBJ to {output_path}")

    def compute_residual_mesh_and_name(self):
        name = f"{self.point_cloud_name} Residuals"
        mesh = self.compute_mesh()

        subsample_voxel = self.residual_computation_config.subsample_voxel
        point_cloud_data = self.point_cloud_data
        if subsample_voxel > 0.0:
            point_cloud_data = point_cloud_data.voxel_down_sample(
                voxel_size=float(subsample_voxel)
            )

        if len(point_cloud_data.points) == 0:
            raise ReconstructionError("Point cloud is empty.")

        residuals = self._compute_residuals(mesh, point_cloud_data)

        residual_mesh = self._colored_mesh_from_residuals(mesh, residuals)

        threshold = self.residual_computation_config.threshold
        if threshold is not None:
            self._hide_mesh_above_threshold(residual_mesh, mesh, residuals, threshold)
            name = f"{name} (≤ {threshold:.3g})"

        if self.residual_computation_config.report_area_over_threshold:
            self._report_area_over_threshold(name, residual_mesh, residuals, threshold)

        return name, residual_mesh

    @staticmethod
    def _compute_residuals(mesh, point_cloud_data):
        kdtree = o3d.geometry.KDTreeFlann(point_cloud_data)
        vertices = np.asarray(mesh.vertices)
        residuals = np.empty((len(vertices),), dtype=float)
        for i, v in enumerate(vertices):
            _, _, d2 = kdtree.search_knn_vector_3d(v, 1)
            residuals[i] = math.sqrt(d2[0]) if len(d2) else float("inf")
        return residuals

    def _colored_mesh_from_residuals(self, mesh, residuals):
        max_vertices = (
            self.residual_computation_config.max_residual
            if self.residual_computation_config.max_residual is not None
            else None
        )
        colors = ViridisScalarColorizer().map(residuals, max_vertices=max_vertices)
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        residual_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.copy()),
            o3d.utility.Vector3iVector(tris.copy()),
        )
        residual_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return residual_mesh

    @staticmethod
    def _hide_mesh_above_threshold(residual_mesh, mesh, residuals, threshold):
        tri_idx = np.asarray(mesh.triangles)
        tri_mean = residuals[tri_idx].mean(axis=1)
        mask_remove = tri_mean > float(threshold)
        residual_mesh.remove_triangles_by_mask(mask_remove.tolist())
        residual_mesh.remove_unreferenced_vertices()

    @staticmethod
    def _report_area_over_threshold(name, mesh, residuals, threshold):
        # Skip report on empty meshes
        if len(mesh.vertices) == 0:
            print(f"[{name}] Mesh is empty.")
            return

        stats = {
            "mean": float(np.mean(residuals)),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
        }

        name = f"{name}_residuals"

        print(
            f"[{name}] Mesh→Point residuals: rmse={stats['rmse']:.6f}, "
            f"mean={stats['mean']:.6f}, max={stats['max']:.6f}"
        )
        tri_idx = np.asarray(mesh.triangles)
        tri_mean = residuals[tri_idx].mean(axis=1)
        if threshold is not None:
            over = int((tri_mean > float(threshold)).sum())
            total = int(len(tri_idx))
            pct = (100.0 * over / total) if total > 0 else 0.0
            print(f"  - triangles over threshold: {over}/{total} ({pct:.1f}%)")
