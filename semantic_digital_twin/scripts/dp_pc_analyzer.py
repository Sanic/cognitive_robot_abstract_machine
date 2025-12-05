import numpy as np
import open3d as o3d
from functools import lru_cache
from dataclasses import dataclass
import random


# ------------------------------------------------------------------------
# Debug config helpers
# ------------------------------------------------------------------------

@dataclass
class DebugConfig:
    enabled: bool = False          # master switch
    max_depth: int = 5             # how deep in recursion to print
    print_threshold_vol: int = 0   # only log boxes with volume >= this
    regions_of_interest: list = None  # list of (x0,x1,y0,y1,z0,z1) boxes


def box_overlaps_region(box, roi):
    """
    Check if 3D index-space box overlaps region of interest.
    box, roi: (x0,x1,y0,y1,z0,z1)
    """
    x0, x1, y0, y1, z0, z1 = box
    rx0, rx1, ry0, ry1, rz0, rz1 = roi
    # AABB overlap
    return not (
        x1 <= rx0 or rx1 <= x0 or
        y1 <= ry0 or ry1 <= y0 or
        z1 <= rz0 or rz1 <= z0
    )


# ------------------------------------------------------------------------
# 1. Synthetic table as an Open3D point cloud
# ------------------------------------------------------------------------

def create_synthetic_table_point_cloud(
    tabletop_size=(0.8, 0.5),   # (length_x, width_z)
    tabletop_thickness=0.05,
    tabletop_height=0.7,
    leg_thickness=0.05,
    leg_height=0.7,
    point_spacing=0.01,
):
    """
    Create a simple rectangular table:
        - tabletop: solid block
        - 4 legs: solid blocks at corners

    All in world coordinates, returned as o3d.geometry.PointCloud.
    """
    lx, lz = tabletop_size
    t = tabletop_thickness
    h = tabletop_height
    lt = leg_thickness

    # Tabletop: centered around origin in x/z, at height h .. h + t
    x_top = np.arange(-lx / 2, lx / 2, point_spacing)
    z_top = np.arange(-lz / 2, lz / 2, point_spacing)
    y_top = np.arange(h, h + t, point_spacing)

    X_top, Y_top, Z_top = np.meshgrid(x_top, y_top, z_top, indexing="ij")
    tabletop_points = np.stack([X_top.ravel(), Y_top.ravel(), Z_top.ravel()], axis=1)

    # Legs: 4 solid vertical blocks at corners, from y=0..leg_height
    # Leg centers at (±(lx/2 - lt/2), ±(lz/2 - lt/2))
    leg_centers = [
        (-lx / 2 + lt / 2, -lz / 2 + lt / 2),
        ( lx / 2 - lt / 2, -lz / 2 + lt / 2),
        (-lx / 2 + lt / 2,  lz / 2 - lt / 2),
        ( lx / 2 - lt / 2,  lz / 2 - lt / 2),
    ]

    all_leg_points = []
    for cx, cz in leg_centers:
        x_leg = np.arange(cx - lt / 2, cx + lt / 2, point_spacing)
        z_leg = np.arange(cz - lt / 2, cz + lt / 2, point_spacing)
        y_leg = np.arange(0.0, leg_height, point_spacing)
        X_leg, Y_leg, Z_leg = np.meshgrid(x_leg, y_leg, z_leg, indexing="ij")
        leg_pts = np.stack([X_leg.ravel(), Y_leg.ravel(), Z_leg.ravel()], axis=1)
        all_leg_points.append(leg_pts)

    all_points = np.vstack([tabletop_points] + all_leg_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    return pcd


# ------------------------------------------------------------------------
# 2. Convert Open3D VoxelGrid -> dense occupancy grid
# ------------------------------------------------------------------------

def build_occupancy_from_voxel_grid(voxel_grid):
    """
    Convert an Open3D VoxelGrid into a dense 3D occupancy array.

    Returns:
        occ: boolean 3D numpy array, shape (Nx, Ny, Nz)
        voxel_size: float
        grid_origin_world: np.array(3,) - world coords of voxel (0,0,0) min corner
    """
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        raise ValueError("VoxelGrid is empty")

    voxel_size = voxel_grid.voxel_size

    # Each voxel has .grid_index (i, j, k)
    indices = np.array([v.grid_index for v in voxels], dtype=np.int32)  # (M, 3)

    mins = indices.min(axis=0)  # [ix_min, iy_min, iz_min]
    maxs = indices.max(axis=0)  # [ix_max, iy_max, iz_max]

    compressed = indices - mins
    dims = compressed.max(axis=0) + 1
    Nx, Ny, Nz = dims.tolist()

    occ = np.zeros((Nx, Ny, Nz), dtype=bool)
    occ[compressed[:, 0], compressed[:, 1], compressed[:, 2]] = True

    # World position of voxel (0,0,0) min corner.
    grid_origin_world = np.array(voxel_grid.origin) + voxel_size * mins

    return occ, voxel_size, grid_origin_world


# ------------------------------------------------------------------------
# 3. Prefix sums + DP guillotine decomposition
# ------------------------------------------------------------------------

def build_prefix_sums(occ):
    """
    3D prefix sums (integral volume) to query occupied voxel counts in O(1).
    """
    occ_int = occ.astype(np.int32)
    Nx, Ny, Nz = occ_int.shape
    ps = np.zeros((Nx + 1, Ny + 1, Nz + 1), dtype=np.int32)
    ps[1:, 1:, 1:] = occ_int.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return ps


def get_occ_count(ps, x0, x1, y0, y1, z0, z1):
    """
    Query #occupied voxels in [x0,x1) × [y0,y1) × [z0,z1) using prefix sums ps.
    """
    return (
        ps[x1, y1, z1]
        - ps[x0, y1, z1]
        - ps[x1, y0, z1]
        - ps[x1, y1, z0]
        + ps[x0, y0, z1]
        + ps[x0, y1, z0]
        + ps[x1, y0, z0]
        - ps[x0, y0, z0]
    )


def decompose_occupancy_grid_to_boxes(
    occ,
    voxel_size,
    grid_origin_world,
    epsilon=0.0,
    lambda_empty=0.1,
    min_box_dim=1,
    debug: DebugConfig = None,
):
    """
    DP-based guillotine decomposition on dense occupancy grid.

    Args:
        occ: bool 3D array, shape (Nx, Ny, Nz)
        voxel_size: float
        grid_origin_world: np.array(3,) - world coords of box index (0,0,0) min corner
        epsilon: max allowed empty fraction in SINGLE_BOX (softened: always
                 allow SINGLE_BOX if no splitting is possible)
        lambda_empty: penalty per empty voxel inside SINGLE_BOX
        min_box_dim: minimum dimension (in voxels) where splitting is allowed
        debug: DebugConfig or None

    Returns:
        boxes_o3d: list of open3d.geometry.AxisAlignedBoundingBox
        box_indices: list of index-space boxes (x0,x1,y0,y1,z0,z1)
    """
    Nx, Ny, Nz = occ.shape
    ps = build_prefix_sums(occ)

    occ_indices = np.argwhere(occ)
    if occ_indices.size == 0:
        return [], []

    x_min, y_min, z_min = occ_indices.min(axis=0)
    x_max, y_max, z_max = occ_indices.max(axis=0) + 1  # half-open

    dp_choice = {}
    INF = 1e18

    # Note: depth is part of the cache key; that's fine for debugging
    @lru_cache(maxsize=None)
    def F(x0, x1, y0, y1, z0, z1, depth=0):
        key = (x0, x1, y0, y1, z0, z1)
        box = key

        occ_count = get_occ_count(ps, x0, x1, y0, y1, z0, z1)
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        empty_count = volume - occ_count

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        can_split = (dx > min_box_dim) or (dy > min_box_dim) or (dz > min_box_dim)

        # Debug: print basic stats
        if debug and debug.enabled:
            if depth <= debug.max_depth and volume >= debug.print_threshold_vol:
                # check ROI
                roi_hit = False
                if debug.regions_of_interest:
                    for roi in debug.regions_of_interest:
                        if box_overlaps_region(box, roi):
                            roi_hit = True
                            break
                indent = "  " * depth
                print(
                    f"{indent}Box {box}: depth={depth}, "
                    f"occ={occ_count}, empty={empty_count}, vol={volume}, roi={roi_hit}"
                )

        if occ_count == 0:
            dp_choice[key] = ("EMPTY", None)
            if debug and debug.enabled:
                if depth <= debug.max_depth and volume >= debug.print_threshold_vol:
                    indent = "  " * depth
                    print(f"{indent}-> EMPTY (cost=0)")
            return 0.0

        best_cost = INF
        best_choice = None

        empty_fraction = empty_count / float(volume)

        # Option A: represent as single box
        # Always allow SINGLE_BOX if no further splitting is allowed
        if empty_fraction <= epsilon or not can_split:
            cost_box = 1.0 + lambda_empty * empty_count
            best_cost = cost_box
            best_choice = ("SINGLE_BOX", None)

        # Option B: try guillotine splits
        if can_split:
            # Split along X
            if dx > 1:
                for s in range(x0 + 1, x1):
                    c = F(x0, s, y0, y1, z0, z1, depth + 1) + F(s, x1, y0, y1, z0, z1, depth + 1)
                    if c < best_cost:
                        best_cost = c
                        best_choice = ("SPLIT_X", s)

            # Split along Y
            if dy > 1:
                for s in range(y0 + 1, y1):
                    c = F(x0, x1, y0, s, z0, z1, depth + 1) + F(x0, x1, s, y1, z0, z1, depth + 1)
                    if c < best_cost:
                        best_cost = c
                        best_choice = ("SPLIT_Y", s)

            # Split along Z
            if dz > 1:
                for s in range(z0 + 1, z1):
                    c = F(x0, x1, y0, y1, z0, s, depth + 1) + F(x0, x1, y0, y1, s, z1, depth + 1)
                    if c < best_cost:
                        best_cost = c
                        best_choice = ("SPLIT_Z", s)

        dp_choice[key] = best_choice

        # Debug: log final decision
        if debug and debug.enabled:
            if depth <= debug.max_depth and volume >= debug.print_threshold_vol:
                indent = "  " * depth
                ctype, param = best_choice
                if ctype == "SINGLE_BOX":
                    print(f"{indent}-> SINGLE_BOX (cost={best_cost:.3f})")
                elif ctype in ("SPLIT_X", "SPLIT_Y", "SPLIT_Z"):
                    axis = ctype[-1]
                    print(f"{indent}-> SPLIT_{axis} at {param} (cost={best_cost:.3f})")
                elif ctype == "EMPTY":
                    print(f"{indent}-> EMPTY (cost=0.0)")
                else:
                    print(f"{indent}-> UNKNOWN CHOICE {best_choice} (cost={best_cost:.3f})")

        return best_cost

    def collect_boxes(x0, x1, y0, y1, z0, z1, out_list):
        key = (x0, x1, y0, y1, z0, z1)
        ctype, param = dp_choice[key]

        if ctype == "EMPTY":
            return

        if ctype == "SINGLE_BOX":
            out_list.append((x0, x1, y0, y1, z0, z1))
            return

        if ctype == "SPLIT_X":
            s = param
            collect_boxes(x0, s, y0, y1, z0, z1, out_list)
            collect_boxes(s, x1, y0, y1, z0, z1, out_list)

        elif ctype == "SPLIT_Y":
            s = param
            collect_boxes(x0, x1, y0, s, z0, z1, out_list)
            collect_boxes(x0, x1, s, y1, z0, z1, out_list)

        elif ctype == "SPLIT_Z":
            s = param
            collect_boxes(x0, x1, y0, y1, z0, s, out_list)
            collect_boxes(x0, x1, y0, y1, s, z1, out_list)

    # Run DP on bounding box
    F(x_min, x_max, y_min, y_max, z_min, z_max, depth=0)

    # Backtrack to get boxes in index space
    box_indices = []
    collect_boxes(x_min, x_max, y_min, y_max, z_min, z_max, box_indices)

    # Convert to Open3D AABBs
    boxes_o3d = []
    for ix0, ix1, iy0, iy1, iz0, iz1 in box_indices:
        min_idx = np.array([ix0, iy0, iz0], dtype=float)
        max_idx = np.array([ix1, iy1, iz1], dtype=float)

        min_corner = grid_origin_world + voxel_size * min_idx
        max_corner = grid_origin_world + voxel_size * max_idx

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
        boxes_o3d.append(aabb)

    return boxes_o3d, box_indices


def decompose_point_cloud_to_boxes(
    pcd,
    voxel_size=0.02,
    epsilon=0.0,
    lambda_empty=0.05,
    min_box_dim=1,
    debug: DebugConfig = None,
):
    """
    Full pipeline: Open3D PointCloud -> VoxelGrid -> DP decomposition -> AABBs.
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    occ, vs, origin = build_occupancy_from_voxel_grid(voxel_grid)

    boxes, idx_boxes = decompose_occupancy_grid_to_boxes(
        occ,
        voxel_size=vs,
        grid_origin_world=origin,
        epsilon=epsilon,
        lambda_empty=lambda_empty,
        min_box_dim=min_box_dim,
        debug=debug,
    )
    return boxes, idx_boxes, occ, vs, origin


# ------------------------------------------------------------------------
# 4. Visual debug helper
# ------------------------------------------------------------------------

def visualize_debug_boxes(occ, voxel_size, grid_origin_world, box_indices, pcd=None):
    """
    Given list of index-space boxes (x0,x1,y0,y1,z0,z1),
    visualize them as AABBs in Open3D.
    Optionally with the original point cloud.
    """
    geoms = []
    if pcd is not None:
        geoms.append(pcd)

    for key in box_indices:
        ix0, ix1, iy0, iy1, iz0, iz1 = key
        min_idx = np.array([ix0, iy0, iz0], dtype=float)
        max_idx = np.array([ix1, iy1, iz1], dtype=float)
        min_corner = grid_origin_world + voxel_size * min_idx
        max_corner = grid_origin_world + voxel_size * max_idx

        aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
        aabb.color = [random.random(), random.random(), random.random()]
        geoms.append(aabb)

    o3d.visualization.draw_geometries(geoms)


# ------------------------------------------------------------------------
# 5. Optional NumPy-only unit test for the synthetic table (no Open3D)
# ------------------------------------------------------------------------

def unit_test_synthetic_table():
    """
    Simple unit test on a 20x20x20 voxel "table" grid using the same DP logic
    (but without Open3D).
    """
    Nx = Ny = Nz = 20
    occ = np.zeros((Nx, Ny, Nz), dtype=bool)

    # Table top
    occ[4:16, 14:15, 4:16] = True

    # Four legs
    occ[4:5,   5:15, 4:5]    = True
    occ[15:16, 5:15, 4:5]    = True
    occ[4:5,   5:15, 15:16]  = True
    occ[15:16, 5:15, 15:16]  = True

    ps = build_prefix_sums(occ)
    x_min, y_min, z_min = np.argwhere(occ).min(axis=0)
    x_max, y_max, z_max = np.argwhere(occ).max(axis=0) + 1

    # Minimal wrapper to reuse the main DP
    boxes_o3d, idx_boxes = decompose_occupancy_grid_to_boxes(
        occ,
        voxel_size=1.0,
        grid_origin_world=np.zeros(3),
        epsilon=0.0,
        lambda_empty=0.1,
        min_box_dim=1,
        debug=None,
    )

    print("Unit test synthetic table: got", len(idx_boxes), "boxes")
    print("Boxes:", idx_boxes)
    # For a perfect table, we expect 5 boxes
    if len(idx_boxes) != 5:
        print("WARNING: expected 5 boxes for ideal table; check parameters/logic")

    return idx_boxes


# ------------------------------------------------------------------------
# 6. Demo: synthetic table -> DP -> debug -> visualize
# ------------------------------------------------------------------------

if __name__ == "__main__":
    # Optional: run the unit test on the voxel-level synthetic table
    print("Running unit test on synthetic voxel table...")
    unit_test_synthetic_table()
    print()

    # Create synthetic table point cloud
    pcd = create_synthetic_table_point_cloud(
        tabletop_size=(0.8, 0.5),
        tabletop_thickness=0.05,
        tabletop_height=0.7,
        leg_thickness=0.05,
        leg_height=0.7,
        point_spacing=0.01,
    )

    print("Synthetic table point cloud:", np.asarray(pcd.points).shape[0], "points")

    # Debug configuration
    debug_cfg = DebugConfig(
        enabled=True,          # turn to False to silence DP logging
        max_depth=4,           # how deep in recursion to log
        print_threshold_vol=5, # don't log tiny 1-voxel boxes
        regions_of_interest=None
        # you can set this to e.g. [(4,16,14,15,4,16)] to highlight tabletop region
    )

    # Run full pipeline with Open3D voxelization + DP
    boxes, idx_boxes, occ, vs, origin = decompose_point_cloud_to_boxes(
        pcd,
        voxel_size=0.02,
        epsilon=0.01,
        lambda_empty=0.05,
        min_box_dim=1,
        debug=debug_cfg,
    )

    print("\nDP decomposition produced", len(boxes), "boxes")
    print("Index-space boxes (x0,x1,y0,y1,z0,z1):")
    for i, key in enumerate(idx_boxes):
        print(f"  Box {i}: {key}")

    print("\nWorld-space boxes (min, max):")
    for i, b in enumerate(boxes):
        print(f"  Box {i}: min={b.get_min_bound()}, max={b.get_max_bound()}")

    # Visualize all resulting boxes + point cloud
    visualize_debug_boxes(occ, vs, origin, idx_boxes, pcd)