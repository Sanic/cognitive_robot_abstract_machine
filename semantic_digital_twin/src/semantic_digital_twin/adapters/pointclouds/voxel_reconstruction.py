from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
class VoxelProcessorBase(PointCloudProcessor, ABC):
    """
    Base class for voxel-based mesh construction from point clouds.
    Subclasses define how to turn voxels into a scalar field for marching cubes.
    """

    point_cloud_name: str = "VoxelBase"

    voxel_size: float = field(default=0.02)
    """
    Voxel size for voxelization.
    """

    closing_algorithm: Optional[MorphologicalClosing] = None
    """
    Optional morphology step (semantics depend on subclass).
    """

    @classmethod
    def from_pts_file(
        cls,
        pts_file_path: Path,
        outlier_removal: Optional[OutlierRemoval] = None,
        closing_algorithm: Optional[MorphologicalClosing] = None,
        **kwargs,
    ) -> "VoxelProcessorBase":
        """
        Same as PointCloudProcessor.from_pts_file, but plumbs through closing_algorithm.
        """
        processor = super().from_pts_file(
            pts_file_path,
            outlier_removal=outlier_removal,
            **kwargs,
        )
        processor.closing_algorithm = closing_algorithm
        return processor

    @classmethod
    def from_nth_in_directory(
        cls,
        directory: Path,
        n: int,
        pattern: str = "*.pts",
        sort_by_name: bool = True,
        reverse: bool = False,
        outlier_removal: Optional[OutlierRemoval] = None,
        closing_algorithm: Optional[MorphologicalClosing] = None,
        **kwargs,
    ) -> "VoxelProcessorBase":
        """
        Convenience constructor mirroring your existing VoxelProcessor.from_nth_in_directory.
        """
        processor = super().from_nth_in_directory(
            directory=directory,
            n=n,
            pattern=pattern,
            sort_by_name=sort_by_name,
            reverse=reverse,
            outlier_removal=outlier_removal,
            **kwargs,
        )
        processor.closing_algorithm = closing_algorithm
        return processor

    def _compute_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        1. Voxelize the point cloud.
        2. Build a scalar field from the voxels (subclass).
        3. Optional morphology on the scalar field (subclass semantics).
        4. Marching cubes on the scalar field.
        5. Map to world coordinates and return a TriangleMesh.
        """
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            self.point_cloud_data,
            voxel_size=self.voxel_size,
        )
        voxels = voxel_grid.get_voxels()
        voxel_size = float(voxel_grid.voxel_size)

        if len(voxels) == 0:
            raise RuntimeError("Voxel grid is empty; try decreasing voxel_size.")

        # Common index bookkeeping
        indices = np.array([v.grid_index for v in voxels], dtype=int)
        min_idx = indices.min(axis=0)
        origin = np.asarray(voxel_grid.origin, dtype=float)

        # Subclass builds scalar field on [i,j,k] grid.
        scalar_field = self._build_scalar_field(voxels, indices, min_idx)

        # Optional morphology, interpreted by subclass.
        if self.closing_algorithm is not None:
            scalar_field = self._apply_morphology(scalar_field)

        # Subclass chooses iso-level.
        iso_level = self._marching_cubes_iso_level(scalar_field)

        vertices, faces, normals = self._apply_marching_cubes(
            voxel_size=voxel_size,
            scalar_field=scalar_field,
            level=iso_level,
        )

        # Map vertices back to world coordinates.
        # marching_cubes returns vertices in index-space of scalar_field.
        vertices += min_idx.astype(float) * voxel_size
        vertices += origin

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        return mesh

    # -------- Hooks for subclasses --------

    @abstractmethod
    def _build_scalar_field(
        self,
        voxels: List[o3d.geometry.Voxel],
        indices: NDArray,
        min_idx: NDArray,
    ) -> NDArray:
        """
        Construct a 3D scalar field (float or uint8) from the voxel list.

        The returned array must be indexable as scalar_field[i, j, k].
        """

    def _apply_morphology(self, scalar_field: NDArray) -> NDArray:
        """
        Default morphology: no-op.
        Subclasses can override and call self.closing_algorithm.apply(...)
        on a mask.
        """
        return scalar_field

    @abstractmethod
    def _marching_cubes_iso_level(self, scalar_field: NDArray) -> float:
        """
        Decide which iso-level to use for marching cubes.
        """

    @staticmethod
    def _apply_marching_cubes(
        voxel_size: float,
        scalar_field: NDArray,
        level: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Run marching cubes on the scalar field.
        """
        from skimage import measure

        spacing = (voxel_size, voxel_size, voxel_size)

        verts, faces, normals, values = measure.marching_cubes(
            scalar_field,
            level=level,
            spacing=spacing,
        )

        return verts, faces, normals


@dataclass
class OccupancyVoxelProcessor(VoxelProcessorBase):
    """
    Original behavior: boolean occupancy grid -> uint8 volume -> marching cubes at level=0.5.
    """

    point_cloud_name: str = "VoxelOccupancy"

    def _build_scalar_field(
        self,
        voxels: List[o3d.geometry.Voxel],
        indices: NDArray,
        min_idx: NDArray,
    ) -> NDArray:
        shifted = indices - min_idx
        max_idx = indices.max(axis=0)
        shape = (max_idx - min_idx + 1).astype(int)

        occupancy = np.zeros(shape, dtype=bool)
        occupancy[
            shifted[:, 0],
            shifted[:, 1],
            shifted[:, 2],
        ] = True

        # scalar_field is uint8 0/1 as expected by old pipeline
        return occupancy.astype(np.uint8)

    def _apply_morphology(self, scalar_field: NDArray) -> NDArray:
        """
        Apply morphological closing on the binary mask, then return uint8 again.
        """
        if self.closing_algorithm is None:
            return scalar_field

        mask = scalar_field > 0
        mask = self.closing_algorithm.apply(mask)
        return mask.astype(np.uint8)

    def _marching_cubes_iso_level(self, scalar_field: NDArray) -> float:
        # 0/1 volume → 0.5 threshold like before
        return 0.5


from scipy.ndimage import gaussian_filter, distance_transform_edt


@dataclass
class ScalarFieldVoxelProcessor(VoxelProcessorBase):
    """
    Density / blurred / optional SDF scalar field -> marching cubes.

    This should give smoother, more meaningful surfaces than raw occupancy,
    while staying fully voxel-based.
    """

    point_cloud_name: str = "VoxelScalarField"

    blur_sigma: float = 1.0
    """
    Standard deviation for Gaussian blur in voxel units.
    """

    use_sdf: bool = False
    """
    If True, convert binary support into an approximate signed distance field
    and run MC at level ~= 0.0.
    """

    sdf_truncation_voxels: int = 4
    """
    Truncate SDF values beyond this distance (in voxels).
    """

    iso_level: Optional[float] = None
    """
    Iso-level for MC:
      - If use_sdf and iso_level is None -> 0.0 (zero level-set).
      - If not using SDF and iso_level is None -> 0.2 (heuristic in [0,1] density range).
    """

    support_threshold: float = 0.05
    """
    Threshold in [0,1] on the (normalized) density to define "occupied support".
    Using a small positive threshold instead of >0 makes the support more robust
    to noise and under-sampling, and gives blur a chance to "grow" weakly
    sampled regions (like the front of the seat).
    """

    def _build_scalar_field(
        self,
        voxels: List[o3d.geometry.Voxel],
        indices: NDArray,
        min_idx: NDArray,
    ) -> NDArray:
        shifted = indices - min_idx
        max_idx = indices.max(axis=0)
        shape = (max_idx - min_idx + 1).astype(int)

        # 1) density grid: count how many points per voxel
        density = np.zeros(shape, dtype=float)
        for x, y, z in shifted:
            density[x, y, z] += 1.0

        if density.max() > 0.0:
            density /= density.max()  # normalize to [0, 1]

        # 2) morphology on support mask (if requested)
        #    use a small positive support_threshold so weakly sampled regions
        #    are not entirely thrown away.
        if self.closing_algorithm is not None:
            support = density > self.support_threshold
            support = self.closing_algorithm.apply(support)
            density *= support  # zero out removed regions

        # 3) Gaussian blur (Laplacian-like smoothing on the field)
        if self.blur_sigma > 0.0:
            density = gaussian_filter(density, sigma=self.blur_sigma)
            # optional renormalization
            if density.max() > 0.0:
                density /= density.max()

        # 4) optional SDF conversion
        if self.use_sdf:
            return self._density_to_sdf(density)

        # Directly use (possibly blurred) density as scalar field
        return density

    def _density_to_sdf(self, density: NDArray) -> NDArray:
        """
        Approximate signed distance field (SDF) from density support:
        positive inside, negative outside.
        """
        support = density > self.support_threshold

        if not np.any(support):
            # nothing occupied; return zeros
            return np.zeros_like(density, dtype=float)

        inside = distance_transform_edt(support)
        outside = distance_transform_edt(~support)

        # sign convention: positive inside
        sdf = inside - outside

        # truncate distances
        t = self.sdf_truncation_voxels
        if t is not None and t > 0:
            sdf = np.clip(sdf, -t, t)

        return sdf.astype(float)

    def _apply_morphology(self, scalar_field: NDArray) -> NDArray:
        """
        Morphology is already applied on the density support mask before blur/SDF.
        Nothing to do here by default.
        """
        return scalar_field

    def _marching_cubes_iso_level(self, scalar_field: NDArray) -> float:

        if self.iso_level is not None:
            return self.iso_level

        if self.use_sdf:
            # zero level-set of SDF
            return 0.0

        # density in [0,1]; 0.2–0.4 is a reasonable starting band
        return 0.2
