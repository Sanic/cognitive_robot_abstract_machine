import os.path
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
    MorphologicalClosing,
    VoxelProcessor,
)

file_path = Path(
    "/home/itsme/Downloads/archive/PartAnnotation/03001627/points/1a6f615e8b1b5ae4dbbc9440457e303e.pts"
)


output_directory = "/home/itsme/Downloads/reconstructed_meshes"

parent = os.path.dirname(output_directory)

# Require the parent directory to already exist
if not os.path.isdir(parent):
    raise FileNotFoundError(f"Parent directory does not exist: {parent}")

# Now create ONLY the final directory, if missing
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

voxel_processor = VoxelProcessor.from_pts_file(
    file_path,
    closing_algorithm=MorphologicalClosing(),
)
mesh = voxel_processor.compute_mesh()
voxel_processor.export_as_obj_file(os.path.join(output_directory, "voxelized_mesh.obj"))

poisson_processor = PoissonReconstructionProcessor.from_pts_file(file_path)
poisson_mesh = poisson_processor.compute_mesh()
poisson_processor.export_as_obj_file(os.path.join(output_directory, "poisson_mesh.obj"))

ball_pivoting_processor = BallPivotingProcessor.from_pts_file(file_path)
ball_pivoting_mesh = ball_pivoting_processor.compute_mesh()
ball_pivoting_processor.export_as_obj_file(
    os.path.join(output_directory, "ball_pivoting_mesh.obj")
)

pyvista_processor = PyVistaProcessor.from_pts_file(file_path)
pyvista_mesh = pyvista_processor.compute_mesh()
pyvista_processor.export_as_obj_file(os.path.join(output_directory, "pyvista_mesh.obj"))


PointCloudReconstructionVisualizer
