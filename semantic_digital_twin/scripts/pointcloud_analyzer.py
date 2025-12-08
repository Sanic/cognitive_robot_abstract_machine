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

from pathlib import Path

from semantic_digital_twin.adapters.pointclouds.normals_based_reconstruction import (
    PoissonReconstructionProcessor,
    BallPivotingProcessor,
    PyVistaProcessor,
)
from semantic_digital_twin.adapters.pointclouds.visualizer import (
    PointCloudReconstructionVisualizer,
)
from semantic_digital_twin.adapters.pointclouds.voxel_reconstruction import (
    VoxelProcessor,
    MorphologicalClosing,
)


def main() -> None:
    patrick_path = Path(
        "/home/pmania/Downloads/archive/PartAnnotation/03001627-chair/points/"
    )
    luca_path = Path("/home/itsme/Downloads/archive/PartAnnotation/03001627/points")

    kitchen_path = Path("/home/pmania/Downloads/archive/PPP/pts/")

    # default_path = patrick_path
    default_path = kitchen_path

    processors = []

    poisson_processor = PoissonReconstructionProcessor.from_nth_in_directory(
        default_path, 1
    )
    processors.append(poisson_processor)

    # ball_pivoting_processor = BallPivotingProcessor.from_nth_in_directory(
    #     default_path, 1
    # )
    # processors.append(ball_pivoting_processor)

    # pyvista_processor = PyVistaProcessor.from_nth_in_directory(default_path, 1)
    # processors.append(pyvista_processor)

    voxel_processor = VoxelProcessor.from_nth_in_directory(
        default_path, 1, closing_algorithm=MorphologicalClosing()
    )
    processors.append(voxel_processor)

    PointCloudReconstructionVisualizer(
        poisson_processor.point_cloud_data, processors
    ).show()

    ...


if __name__ == "__main__":  # pragma: no cover
    main()
