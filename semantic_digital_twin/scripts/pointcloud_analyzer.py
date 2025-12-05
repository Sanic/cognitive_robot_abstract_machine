#!/usr/bin/env python3
"""
Point Cloud Analyzer
====================

This script provides a small, object‑oriented tool to:

- Load a .pts point cloud file.
- Visualize the point cloud in Open3D.
- Reconstruct one or more meshes using different methods (Poisson, Ball Pivoting).
- Visualize results with an Open3D visualizer that allows toggling individual geometries.

Example .pts file path (as mentioned in the issue):
    /home/pmania/Downloads/archive/PartAnnotation/03001627-chair/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts

Run:
    python -m pycram.scripts.pointcloud_analyzer --file /path/to/file.pts --show-pcd --poisson --ball-pivot

Notes
-----
- The visualizer tries to use the modern Open3D O3DVisualizer with a sidebar for toggling geometries.
  If unavailable, it falls back to other available drawing utilities.
- No tests are added, as requested. Use this script with parameters to experiment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import open3d as o3d  # type: ignore
except Exception as exc:  # pragma: no cover - depends on environment
    raise RuntimeError(
        "Open3D is required to run this script. Please install 'open3d'."
    ) from exc


# ========== Exceptions ==========


class PointCloudLoadError(Exception):
    """Raised when a point cloud file cannot be loaded or parsed."""


class ReconstructionError(Exception):
    """Raised when a mesh reconstruction fails."""


# ========== Configuration Data Classes ==========


@dataclass
class PointCloudConfig:
    """Configuration for loading and preparing a point cloud.

    The configuration controls voxel downsampling and normal estimation.
    """

    voxel_size: float = 0.0
    estimate_normals: bool = True
    normal_radius: float = 0.05
    normal_max_nn: int = 30


@dataclass
class PoissonConfig:
    """Configuration for Poisson surface reconstruction."""

    enabled: bool = False
    depth: int = 10
    scale: float = 1.1
    linear_fit: bool = False
    density_trim_quantile: float = 0.02
    simplify_target_triangles: Optional[int] = None


@dataclass
class BallPivotConfig:
    """Configuration for Ball Pivoting surface reconstruction."""

    enabled: bool = False
    radius_factor: float = 2.5
    radii_multipliers: Tuple[float, float, float] = (1.0, 2.0, 4.0)
    simplify_target_triangles: Optional[int] = None


@dataclass
class PyVistaConfig:
    """Configuration for PyVista surface reconstruction.

    The reconstruction uses PyVista's ``reconstruct_surface`` on the point cloud.
    """

    enabled: bool = False
    simplify_target_triangles: Optional[int] = None


@dataclass
class MeshResidualVizConfig:
    """Controls coloring meshes by distance to the input point cloud.

    When enabled, per‑vertex nearest‑neighbor distances to the point cloud are
    computed and mapped to colors for a heatmap visualization.
    """

    enabled: bool = False
    max_residual: Optional[float] = None
    colormap: str = "viridis"
    threshold: Optional[float] = None
    hide_above_threshold: bool = False
    subsample_voxel: Optional[float] = None
    report_area_over_threshold: bool = True

@dataclass
class VisualizationConfig:
    """Configuration for visualization preferences."""

    title: str = "Point Cloud Analyzer"
    width: int = 1280
    height: int = 800
    show_point_cloud: bool = True
    show_meshes: bool = True
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    point_size: float = 3.0
    mesh_residuals: MeshResidualVizConfig = field(default_factory=MeshResidualVizConfig)


@dataclass
class AnalyzerConfig:
    """Top-level configuration for a complete analysis run."""

    file: Path = field(default=Path("/home/pmania/Downloads/archive/PartAnnotation/03001627-chair/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts"))
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    ball_pivot: BallPivotConfig = field(default_factory=BallPivotConfig)
    pyvista: PyVistaConfig = field(default_factory=PyVistaConfig)
    visualize: VisualizationConfig = field(default_factory=VisualizationConfig)


# ========== Core Domain Classes ==========


class PointCloudLoader:
    """Loads and prepares point clouds from .pts files.

    The loader expects a .pts file with whitespace‑separated values: X Y Z [R G B ...].
    Additional columns beyond XYZ are ignored, except RGB if present.
    """

    def __init__(self, config: PointCloudConfig) -> None:
        self.config = config

    def load(self, file_path: Path) -> o3d.geometry.PointCloud:
        """Load a point cloud from ``file_path``.

        Supports .pts with XYZ and optional RGB columns. Lines starting with '#' or '//' are ignored.
        """
        if not file_path.exists():
            raise PointCloudLoadError(f"File not found: {file_path}")

        try:
            points: List[List[float]] = []
            colors: List[List[float]] = []
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    tokens = line.split()

                    # Heuristic: skip a single integer header line sometimes present in .pts
                    if len(tokens) == 1:
                        try:
                            int(tokens[0])
                            continue
                        except ValueError:
                            pass

                    if len(tokens) < 3:
                        continue

                    try:
                        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
                    except ValueError:
                        # Skip malformed lines
                        continue

                    points.append([x, y, z])

                    # Optional RGB handling (various conventions exist)
                    rgb: Optional[Tuple[float, float, float]] = None
                    if len(tokens) >= 6:
                        try:
                            r, g, b = float(tokens[3]), float(tokens[4]), float(tokens[5])
                            # Normalize if values look like 0..255
                            if max(r, g, b) > 1.5:
                                rgb = (r / 255.0, g / 255.0, b / 255.0)
                            else:
                                rgb = (r, g, b)
                        except ValueError:
                            rgb = None
                    if rgb is not None:
                        colors.append([rgb[0], rgb[1], rgb[2]])

            if not points:
                raise PointCloudLoadError(
                    f"No valid points parsed from file: {file_path}"
                )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors and len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # Downsample if configured
            if self.config.voxel_size and self.config.voxel_size > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=self.config.voxel_size)

            if self.config.estimate_normals:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.config.normal_radius, max_nn=self.config.normal_max_nn
                    )
                )
                pcd.normalize_normals()

            return pcd
        except OSError as exc:
            raise PointCloudLoadError(
                f"Failed reading point cloud file: {file_path}"
            ) from exc

    def load_nth_from_directory(
        self,
        directory: Path,
        n: int,
        pattern: str = "*.pts",
        sort_by_name: bool = True,
        reverse: bool = False,
    ) -> o3d.geometry.PointCloud:
        """Load the n-th ``.pts`` file from a directory.

        This helper lists files matching ``pattern`` in ``directory``, sorts them
        deterministically, picks the n-th item (1-based), and loads it using
        :py:meth:`load`.

        Raises
        ------
        PointCloudLoadError
            If the directory is invalid, no files match, or ``n`` is out of range.
        """
        if not directory.exists() or not directory.is_dir():
            raise PointCloudLoadError(f"Directory not found or not a directory: {directory}")

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
        return self.load(target)


class MeshReconstructor:
    """Interface for mesh reconstruction algorithms."""

    def name(self) -> str:
        """Return a human‑friendly name for the reconstructor."""

        raise NotImplementedError

    def reconstruct(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Reconstruct a mesh from the given point cloud."""

        raise NotImplementedError

    @staticmethod
    def _postprocess(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Perform basic cleanup on a mesh.

        Operations include removing degenerate and duplicated elements and computing normals.
        """
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh


class PoissonReconstructor(MeshReconstructor):
    """Poisson surface reconstruction strategy."""

    def __init__(self, config: PoissonConfig) -> None:
        self.config = config

    def name(self) -> str:
        return "Poisson"

    def reconstruct(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        if not self.config.enabled:
            raise ReconstructionError("Poisson reconstruction is disabled in config.")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.config.depth,
            scale=self.config.scale,
            linear_fit=self.config.linear_fit,
        )

        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional speedup
            raise ReconstructionError("NumPy is required for Poisson postprocessing.") from exc

        densities_np = np.asarray(densities)
        if densities_np.size:
            # Trim low-density vertices (likely artifacts)
            q = min(max(self.config.density_trim_quantile, 0.0), 0.5)
            # q = 0.2
            threshold = np.quantile(densities_np, q)
            vertices_to_keep = densities_np >= threshold
            mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

        # Crop to the point cloud bounding box (expanded a bit)
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.1, bbox.get_center())
        mesh = mesh.crop(bbox)

        mesh = self._postprocess(mesh)

        if self.config.simplify_target_triangles and self.config.simplify_target_triangles > 0:
            target = int(self.config.simplify_target_triangles)
            mesh = mesh.simplify_quadric_decimation(target)
            mesh.compute_vertex_normals()

        return mesh


class BallPivotReconstructor(MeshReconstructor):
    """Ball Pivoting surface reconstruction strategy."""

    def __init__(self, config: BallPivotConfig) -> None:
        self.config = config

    def name(self) -> str:
        return "BallPivot"

    def reconstruct(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        if not self.config.enabled:
            raise ReconstructionError("Ball Pivot reconstruction is disabled in config.")

        if not pcd.has_normals():
            # Ball pivoting requires normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            pcd.normalize_normals()

        mean_nn = _estimate_mean_nearest_neighbor_distance(pcd)
        if mean_nn <= 0.0:
            raise ReconstructionError("Could not estimate a valid nearest neighbor distance.")

        base_radius = max(mean_nn * self.config.radius_factor, 1e-6)
        radii = o3d.utility.DoubleVector([base_radius * m for m in self.config.radii_multipliers])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        mesh = self._postprocess(mesh)

        if self.config.simplify_target_triangles and self.config.simplify_target_triangles > 0:
            target = int(self.config.simplify_target_triangles)
            mesh = mesh.simplify_quadric_decimation(target)
            mesh.compute_vertex_normals()

        return mesh


class PyVistaReconstructor(MeshReconstructor):
    """PyVista surface reconstruction strategy.

    Applies PyVista's ``reconstruct_surface`` filter to the input point cloud.
    """

    def __init__(self, config: PyVistaConfig) -> None:
        self.config = config

    def name(self) -> str:
        return "PyVista"

    def reconstruct(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        if not self.config.enabled:
            raise ReconstructionError("PyVista reconstruction is disabled in config.")

        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - optional speedup
            raise ReconstructionError("NumPy is required for PyVista reconstruction.") from exc

        try:
            import pyvista as pv  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ReconstructionError(
                "PyVista is required for this reconstruction. Please install 'pyvista'."
            ) from exc

        # Convert Open3D point cloud to PyVista PolyData
        points_np = np.asarray(pcd.points)
        if points_np.size == 0:
            raise ReconstructionError("Point cloud is empty.")

        pv_points = pv.PolyData(points_np)

        # Reconstruct surface using PyVista filter (VTK SurfaceReconstruction)
        try:
            pv_mesh = pv_points.reconstruct_surface()
        except Exception as exc:
            raise ReconstructionError(f"PyVista reconstruct_surface failed: {exc}") from exc

        # Ensure triangulated mesh for Open3D compatibility
        try:
            if not pv_mesh.is_all_triangles:
                pv_mesh = pv_mesh.triangulate()
        except Exception:
            pv_mesh = pv_mesh.triangulate()

        # Convert PyVista PolyData to Open3D TriangleMesh
        verts = np.asarray(pv_mesh.points, dtype=float)
        faces_arr = pv_mesh.faces.reshape((-1, 4))[:, 1:4]
        faces = np.asarray(faces_arr, dtype=np.int64)

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces),
        )

        mesh = self._postprocess(mesh)

        if self.config.simplify_target_triangles and self.config.simplify_target_triangles > 0:
            target = int(self.config.simplify_target_triangles)
            mesh = mesh.simplify_quadric_decimation(target)
            mesh.compute_vertex_normals()

        return mesh


def _estimate_mean_nearest_neighbor_distance(
    pcd: o3d.geometry.PointCloud, sample_size: int = 2000
) -> float:
    """Estimate the mean nearest neighbor distance of a point cloud.

    Uses a uniform random subset for efficiency on very large point clouds.
    """
    import random

    n = len(pcd.points)
    if n < 2:
        return 0.0

    indices: List[int] = list(range(n))
    if n > sample_size:
        indices = random.sample(indices, sample_size)

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    distances: List[float] = []
    for idx in indices:
        _, _, dists = kdtree.search_knn_vector_3d(pcd.points[idx], 2)
        if len(dists) >= 2 and dists[1] > 0.0:
            distances.append(math.sqrt(dists[1]))

    if not distances:
        return 0.0
    return float(sum(distances) / len(distances))

def compute_reconstruction_metrics(
    pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
) -> Dict[str, float]:
    """Compute distance metrics between original points and the reconstructed mesh.

    This approximates the point-to-surface distance by sampling the mesh densely.
    """
    try:
        import numpy as np
    except ImportError:
        return {}

    # Sample the mesh to create a dense representation of the surface
    # We sample enough points to ensure the surface is well represented
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=len(pcd.points) * 2)

    # Compute distance from every original point to the closest point on the mesh surface
    dists = pcd.compute_point_cloud_distance(mesh_pcd)
    dists = np.asarray(dists)

    if dists.size == 0:
        return {"rmse": 0.0, "hausdorff": 0.0}

    rmse = np.sqrt(np.mean(dists**2))
    hausdorff = np.max(dists)
    mean_dist = np.mean(dists)

    return {
        "rmse": float(rmse),
        "hausdorff": float(hausdorff),
        "mean": float(mean_dist),
    }


# ========== Residuals (Mesh → PointCloud) ==========


class MeshResidualComputer:
    """Computes per‑vertex distances from a mesh to a point cloud.

    The distance for each vertex is the nearest‑neighbor distance in the point cloud,
    optionally after voxel subsampling the point cloud for performance.
    """

    def compute(
        self,
        mesh: o3d.geometry.TriangleMesh,
        pcd: o3d.geometry.PointCloud,
        subsample_voxel: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if len(mesh.vertices) == 0:
            return np.zeros((0,), dtype=float), {"rmse": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

        cloud = pcd
        if subsample_voxel is not None and subsample_voxel > 0.0:
            cloud = pcd.voxel_down_sample(voxel_size=float(subsample_voxel))

        if len(cloud.points) == 0:
            # No reference points – mark all as inf to clearly indicate problem
            verts = np.asarray(mesh.vertices)
            residuals = np.full((len(verts),), float("inf"), dtype=float)
            stats = {
                "mean": float(np.mean(residuals)),
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "max": float(np.max(residuals)),
                "median": float(np.median(residuals)),
            }
            return residuals, stats

        kdtree = o3d.geometry.KDTreeFlann(cloud)
        vertices = np.asarray(mesh.vertices)
        residuals = np.empty((len(vertices),), dtype=float)
        for i, v in enumerate(vertices):
            _, _, d2 = kdtree.search_knn_vector_3d(v, 1)
            residuals[i] = math.sqrt(d2[0]) if len(d2) else float("inf")

        stats = {
            "mean": float(np.mean(residuals)),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
        }
        return residuals, stats


class ScalarColorizer:
    """Maps scalar values in [0, vmax] to RGB colors using a fixed colormap.

    Default colormap is a small built‑in Viridis‑like lookup table to avoid extra
    dependencies. Values above vmax are clipped.
    """

    def __init__(self, colormap: str = "viridis") -> None:
        self.colormap = colormap
        self._lut = self._build_lut(colormap)

    def _build_lut(self, name: str) -> np.ndarray:
        # Compact viridis‑like 256‑entry LUT
        if name.lower() == "viridis":
            # A reduced table constructed from viridis samples; values in [0,1]
            # For brevity include 64 entries and interpolate to 256
            base = np.array([
                [0.267, 0.004, 0.329], [0.282, 0.140, 0.457], [0.254, 0.265, 0.531], [0.207, 0.372, 0.553],
                [0.164, 0.471, 0.558], [0.128, 0.567, 0.551], [0.135, 0.659, 0.518], [0.267, 0.748, 0.441],
                [0.478, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144], [0.993, 0.773, 0.188],
                [0.990, 0.636, 0.285], [0.984, 0.503, 0.384], [0.969, 0.382, 0.494], [0.940, 0.278, 0.607],
            ], dtype=float)
            # Interpolate to 256 entries
            x = np.linspace(0.0, 1.0, base.shape[0])
            xi = np.linspace(0.0, 1.0, 256)
            lut = np.zeros((256, 3), dtype=float)
            for c in range(3):
                lut[:, c] = np.interp(xi, x, base[:, c])
            return np.clip(lut, 0.0, 1.0)
        # Fallback to grayscale
        g = np.linspace(0.0, 1.0, 256)
        return np.stack([g, g, g], axis=1)

    def map(self, values: np.ndarray, vmax: Optional[float] = None) -> np.ndarray:
        if values.size == 0:
            return np.zeros((0, 3), dtype=float)
        if vmax is None or not np.isfinite(vmax) or vmax <= 0.0:
            # Robust default: 95th percentile to reduce outlier influence
            vmax = float(np.percentile(values[np.isfinite(values)], 95.0)) if np.any(np.isfinite(values)) else 1.0
            if vmax <= 0.0:
                vmax = 1.0
        vals = np.asarray(values, dtype=float)
        vals[~np.isfinite(vals)] = vmax
        t = np.clip(vals / vmax, 0.0, 1.0)
        idx = np.minimum((t * (len(self._lut) - 1)).astype(int), len(self._lut) - 1)
        return self._lut[idx]


# ========== Visualization ==========


class Visualizer:
    """Displays geometries with toggles using Open3D visualizers.

    Prefers the modern O3DVisualizer with a geometry list panel. Falls back to other
    drawing methods if the GUI system is not available.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def show(self, geometries: Dict[str, o3d.geometry.Geometry]) -> None:
        """Open a window to visualize the provided geometries.

        Geometry names will appear in the UI and can be toggled on and off.
        """
        # Try O3DVisualizer with GUI (best UI for toggling)
        if hasattr(o3d.visualization, "gui") and hasattr(o3d.visualization, "O3DVisualizer"):
            try:
                self._show_with_o3d_visualizer(geometries)
                return
            except Exception:
                pass

        # Try the convenience draw() API with UI
        if hasattr(o3d.visualization, "draw"):
            items = []
            for name, geom in geometries.items():
                items.append({"name": name, "geometry": geom})
            try:
                o3d.visualization.draw(
                    geometries=items,
                    title=self.config.title,
                    show_ui=True,
                    width=self.config.width,
                    height=self.config.height,
                    point_size=self.config.point_size,
                )
                return
            except Exception:
                pass

        # Legacy fallback
        o3d.visualization.draw_geometries(list(geometries.values()))

    def _show_with_o3d_visualizer(self, geometries: Dict[str, o3d.geometry.Geometry]) -> None:
        gui = o3d.visualization.gui
        app = gui.Application.instance
        app.initialize()

        window = o3d.visualization.O3DVisualizer(self.config.title, self.config.width, self.config.height)
        window.show_settings = True

        for name, geom in geometries.items():
            try:
                # Set a visible point size for point clouds when possible
                if isinstance(geom, o3d.geometry.PointCloud):
                    mr = o3d.visualization.rendering.MaterialRecord()
                    mr.shader = "defaultUnlit"
                    mr.point_size = float(self.config.point_size)
                    window.add_geometry(name, geom, mr)
                else:
                    window.add_geometry(name, geom)
            except Exception:
                # Fallback to default add if material path is not supported
                window.add_geometry(name, geom)

        # Initial visibility based on simple rules
        for name in list(geometries.keys()):
            visible = True
            if not self.config.show_point_cloud and name.lower().startswith("point"):
                visible = False
            if not self.config.show_meshes and ("mesh" in name.lower() or name.lower().startswith("poisson") or name.lower().startswith("ball")):
                visible = False
            try:
                window.set_geometry_visibility(name, visible)
            except Exception:
                pass

        try:
            window.set_background(self.config.background_color)
        except Exception:
            pass

        window.reset_camera_to_default()
        app.add_window(window)
        app.run()


# ========== Orchestration ==========


def analyze(config: AnalyzerConfig) -> None:
    """Run the analysis pipeline and visualize results.

    This function loads the point cloud, reconstructs requested meshes, and displays them.
    """
    loader = PointCloudLoader(config.pointcloud)
    # Allow passing a directory: load the first matching .pts using the helper
    if isinstance(config.file, Path) and config.file.is_dir():
        pcd = loader.load_nth_from_directory(config.file, n=1)
    else:
        pcd = loader.load(config.file)

    # Always register the point cloud in the visualizer so it can be toggled on/off
    # in the UI, regardless of the initial visibility preference. Initial visibility
    # is controlled inside the Visualizer based on VisualizationConfig.show_point_cloud.
    geometries: Dict[str, o3d.geometry.Geometry] = {"PointCloud": pcd}

    reconstructors: List[MeshReconstructor] = []
    if config.poisson.enabled:
        reconstructors.append(PoissonReconstructor(config.poisson))
    if config.ball_pivot.enabled:
        reconstructors.append(BallPivotReconstructor(config.ball_pivot))
    if config.pyvista.enabled:
        reconstructors.append(PyVistaReconstructor(config.pyvista))

    for reconstructor in reconstructors:
        try:
            mesh = reconstructor.reconstruct(pcd)
        except Exception as exc:
            raise ReconstructionError(f"{reconstructor.name()} reconstruction failed: {exc}") from exc

        metrics = compute_reconstruction_metrics(pcd, mesh)
        print(f"[{reconstructor.name()}] Reconstruction Metrics:")
        for k, v in metrics.items():
            print(f"  - {k:<10}: {v:.6f}")

        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        geometries[f"{reconstructor.name()} Mesh"] = mesh

        # Optional: add mesh→point residual heatmap as a separate geometry
        if config.visualize.mesh_residuals.enabled:
            residuals, rstats = MeshResidualComputer().compute(
                mesh, pcd, config.visualize.mesh_residuals.subsample_voxel
            )
            vmax = (
                config.visualize.mesh_residuals.max_residual
                if config.visualize.mesh_residuals.max_residual is not None
                else None
            )
            colors = ScalarColorizer(colormap=config.visualize.mesh_residuals.colormap).map(residuals, vmax=vmax)

            # Create a colored copy of the mesh
            verts = np.asarray(mesh.vertices)
            tris = np.asarray(mesh.triangles)
            mesh_res = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts.copy()),
                o3d.utility.Vector3iVector(tris.copy()),
            )
            mesh_res.vertex_colors = o3d.utility.Vector3dVector(colors)

            name = f"{reconstructor.name()} Mesh Residuals"

            thr = config.visualize.mesh_residuals.threshold
            if thr is not None and config.visualize.mesh_residuals.hide_above_threshold:
                tri_idx = np.asarray(mesh.triangles)
                tri_mean = residuals[tri_idx].mean(axis=1)
                mask_remove = tri_mean > float(thr)
                mesh_res.remove_triangles_by_mask(mask_remove.tolist())
                mesh_res.remove_unreferenced_vertices()
                name = f"{name} (≤ {thr:.3g})"

            geometries[name] = mesh_res
            print(
                f"[{reconstructor.name()}] Mesh→Point residuals: rmse={rstats['rmse']:.6f}, "
                f"mean={rstats['mean']:.6f}, max={rstats['max']:.6f}"
            )
            if (
                config.visualize.mesh_residuals.report_area_over_threshold
                and thr is not None
                and len(tris) > 0
            ):
                tri_idx = np.asarray(mesh.triangles)
                tri_mean = residuals[tri_idx].mean(axis=1)
                over = int((tri_mean > float(thr)).sum())
                total = int(len(tri_idx))
                pct = (100.0 * over / total) if total > 0 else 0.0
                print(f"  - triangles over threshold: {over}/{total} ({pct:.1f}%)")

    Visualizer(config.visualize).show(geometries)


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a .pts point cloud, reconstruct meshes (Poisson, Ball Pivot, PyVista), and visualize with toggles.\n"
            "Example path: /home/pmania/Downloads/archive/PartAnnotation/03001627-chair/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts"
        )
    )

    # You can pass either a .pts file or a directory. If a directory is provided,
    # the first .pts file (sorted by name) will be loaded.
    default_path = "/home/pmania/Downloads/archive/PartAnnotation/03001627-chair/points/"

    parser.add_argument(
        "--file",
        type=Path,
        default=default_path,
        help="Path to a .pts file or a directory containing .pts files (first will be loaded)",
    )

    # Point cloud options
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel downsampling size (0 to disable)")
    parser.add_argument("--no-estimate-normals", action="store_true", help="Disable normal estimation for the point cloud")
    parser.add_argument("--normal-radius", type=float, default=0.05, help="Radius for normal estimation")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Max neighbors for normal estimation")

    # Visualization toggles
    parser.add_argument("--title", type=str, default="Point Cloud Analyzer", help="Window title")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=800, help="Window height")
    parser.add_argument(
        "--show-pcd",
        action="store_true",
        default=True,
        help="Show point cloud initially (default: on)",
    )
    parser.add_argument("--hide-meshes", action="store_true", help="Hide meshes initially")
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size for rendering point clouds",
    )

    # Poisson
    parser.add_argument("--poisson", action="store_true", help="Enable Poisson reconstruction")
    parser.add_argument("--poisson-depth", type=int, default=10, help="Poisson octree depth")
    parser.add_argument("--poisson-scale", type=float, default=1.1, help="Poisson scale")
    parser.add_argument("--poisson-linear-fit", action="store_true", help="Enable Poisson linear fit")
    parser.add_argument(
        "--poisson-trim-q",
        type=float,
        default=0.02,
        help="Quantile for trimming low-density vertices (0..0.5)",
    )
    parser.add_argument(
        "--poisson-simplify",
        type=int,
        default=None,
        help="Target triangle count for quadric decimation (optional)",
    )

    # Ball Pivot
    parser.add_argument("--ball-pivot", action="store_true", help="Enable Ball Pivot reconstruction")
    parser.add_argument("--bp-radius-factor", type=float, default=2.5, help="Nearest-neighbor factor for base radius")
    parser.add_argument(
        "--bp-radii-multipliers",
        type=float,
        nargs=3,
        default=(1.0, 2.0, 4.0),
        help="Multipliers to build radii list for Ball Pivot",
    )
    parser.add_argument(
        "--bp-simplify",
        type=int,
        default=None,
        help="Target triangle count for quadric decimation (optional)",
    )

    # PyVista
    parser.add_argument("--pyvista", action="store_true", help="Enable PyVista reconstruct_surface reconstruction")
    parser.add_argument(
        "--pv-simplify",
        type=int,
        default=None,
        help="Target triangle count for quadric decimation (optional)",
    )

    # Mesh residual visualization (mesh → point cloud)
    parser.add_argument(
        "--show-mesh-residuals",
        action="store_true",
        help="Color reconstructed meshes by distance to the input point cloud",
    )
    parser.add_argument(
        "--mesh-residual-max",
        type=float,
        default=None,
        help="Clip residual colormap at this max value (auto if omitted)",
    )
    parser.add_argument(
        "--mesh-residual-threshold",
        type=float,
        default=None,
        help="Threshold (in distance units) for optional filtering/reporting",
    )
    parser.add_argument(
        "--mesh-residual-hide-above-threshold",
        action="store_true",
        help="Hide triangles whose average vertex residual exceeds threshold",
    )
    parser.add_argument(
        "--mesh-residual-colormap",
        type=str,
        default="viridis",
        help="Colormap name for residuals (viridis|grayscale)",
    )
    parser.add_argument(
        "--mesh-residual-subsample-voxel",
        type=float,
        default=None,
        help="Voxel size to subsample the point cloud for KDTree (speeds up large clouds)",
    )
    parser.add_argument(
        "--no-mesh-residual-area-report",
        action="store_true",
        help="Do not print fraction of triangles above threshold",
    )

    return parser


def _args_to_config(args) -> AnalyzerConfig:
    pc_cfg = PointCloudConfig(
        voxel_size=args.voxel_size,
        estimate_normals=not args.no_estimate_normals,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
    )

    vis_cfg = VisualizationConfig(
        title=args.title,
        width=args.width,
        height=args.height,
        show_point_cloud=args.show_pcd,
        show_meshes=not args.hide_meshes,
        point_size=args.point_size,
    )

    # Wire mesh residual visualization options
    vis_cfg.mesh_residuals.enabled = bool(args.show_mesh_residuals)
    vis_cfg.mesh_residuals.max_residual = args.mesh_residual_max
    vis_cfg.mesh_residuals.colormap = str(args.mesh_residual_colormap)
    vis_cfg.mesh_residuals.threshold = args.mesh_residual_threshold
    vis_cfg.mesh_residuals.hide_above_threshold = bool(args.mesh_residual_hide_above_threshold)
    vis_cfg.mesh_residuals.subsample_voxel = args.mesh_residual_subsample_voxel
    vis_cfg.mesh_residuals.report_area_over_threshold = not bool(args.no_mesh_residual_area_report)

    poi_cfg = PoissonConfig(
        enabled=args.poisson,
        depth=args.poisson_depth,
        scale=args.poisson_scale,
        linear_fit=args.poisson_linear_fit,
        density_trim_quantile=args.poisson_trim_q,
        simplify_target_triangles=args.poisson_simplify,
    )

    bp_cfg = BallPivotConfig(
        enabled=args.ball_pivot,
        radius_factor=args.bp_radius_factor,
        radii_multipliers=tuple(args.bp_radii_multipliers),
        simplify_target_triangles=args.bp_simplify,
    )

    pv_cfg = PyVistaConfig(
        enabled=args.pyvista,
        simplify_target_triangles=args.pv_simplify,
    )

    cfg = AnalyzerConfig(
        file=args.file,
        pointcloud=pc_cfg,
        poisson=poi_cfg,
        ball_pivot=bp_cfg,
        pyvista=pv_cfg,
        visualize=vis_cfg,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = _args_to_config(args)
    analyze(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
