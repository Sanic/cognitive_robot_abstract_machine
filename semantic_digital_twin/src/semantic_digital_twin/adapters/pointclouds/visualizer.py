from dataclasses import dataclass, field
from typing import List, Tuple

from semantic_digital_twin.adapters.pointclouds.processor import PointCloudProcessor
import open3d as o3d


@dataclass
class PointCloudReconstructionVisualizerParameters:
    title: str = "Point Cloud Analyzer"
    width: int = 1650
    height: int = 1400
    show_point_cloud: bool = True
    show_meshes: bool = True
    show_settings: bool = True
    point_size: float = 3.0
    visualize_residuals: bool = True
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class PointCloudReconstructionVisualizer:

    point_cloud_data: o3d.geometry.PointCloud

    point_cloud_processors: List[PointCloudProcessor] = field(default_factory=list)

    visualizer_config: PointCloudReconstructionVisualizerParameters = field(
        default_factory=PointCloudReconstructionVisualizerParameters
    )

    def show(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        window = o3d.visualization.O3DVisualizer(
            self.visualizer_config.title,
            self.visualizer_config.width,
            self.visualizer_config.height,
        )
        window.show_settings = self.visualizer_config.show_settings

        material_record = o3d.visualization.rendering.MaterialRecord()
        material_record.shader = "defaultUnlit"
        material_record.point_size = float(self.visualizer_config.point_size)
        window.add_geometry("PointCloud", self.point_cloud_data, material_record)

        names = []

        for processor in self.point_cloud_processors:
            mesh = processor.compute_mesh()
            name = processor.point_cloud_name
            names.append(name)
            window.add_geometry(name, mesh, is_visible=False)

            if self.visualizer_config.visualize_residuals:
                name, mesh = processor.compute_residual_mesh_and_name()
                names.append(name)
                window.add_geometry(name, mesh, is_visible=False)

        window.reset_camera_to_default()
        app.add_window(window)
        app.run()
