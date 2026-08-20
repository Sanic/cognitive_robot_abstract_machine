from pathlib import Path

from robokudo.world_descriptor import (
    BaseWorldDescriptor,
    ObjectSpec,
    PredefinedObject,
    RegionSpec,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class WorldDescriptor(BaseWorldDescriptor):
    """
    A tabletop SemDT RayTracer world with a mesh-backed needle target.
    """

    def __init__(self) -> None:
        super().__init__()
        root = self.world.root

        table_top_z = 0.78
        table_thickness = 0.06
        needle_mesh_path = self._needle_mesh_path()

        object_specs = [
            ObjectSpec(
                name="table",
                box_scale=Scale(1.20, 0.80, table_thickness),
                color=Color(0.65, 0.58, 0.48, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10,
                    y=1.25,
                    z=table_top_z - (table_thickness / 2.0),
                    reference_frame=root,
                ),
            )
        ]

        region_specs = [
            RegionSpec(
                name="table_surface_region",
                box_scale=Scale(1.00, 0.60, 0.03),
                color=Color(0.10, 0.70, 0.30, 0.20),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10,
                    y=1.25,
                    z=table_top_z + 0.015,
                    reference_frame=root,
                ),
            )
        ]

        self.build_objects(root, object_specs)

        needle_color = Color(0.94, 0.78, 0.08, 1.0)
        with self.world.modify_world():
            needle_visual_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(needle_mesh_path),
                color=needle_color,
            )
            needle_collision_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(needle_mesh_path),
                color=needle_color,
            )
            needle_body = Body(
                name=PrefixedName(name="needle"),
                visual=ShapeCollection([needle_visual_mesh]),
                collision=ShapeCollection([needle_collision_mesh]),
            )
            needle_connection = Connection6DoF.create_with_dofs(
                parent=root,
                child=needle_body,
                world=self.world,
            )
            self.world.add_connection(needle_connection)
            self.world.add_semantic_annotation(PredefinedObject(body=needle_body))

        with self.world.modify_world():
            needle_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-1.03,
                y=1.25,
                z=(table_top_z - needle_collision_mesh.local_frame_bounding_box.min_z),
                yaw=0.35,
                reference_frame=root,
            )

        self.build_regions(root, region_specs)

    @staticmethod
    def _needle_mesh_path() -> Path:
        """
        Return the local STL path for the needle mesh.
        """
        return Path(__file__).resolve().parents[4] / "Needle.stl"
