import numpy as np
import threading
import rclpy

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Table,
    Sofa,
    TrashCan,
    Fridge,
    CounterTop,
    Wall,
    Cabinet,
    Cupboard,
    ShelfLayer,
    Hinge,
    Door,
    Handle,
    DiningTable,
    Leg,
    Drawer,
    Desk,
    Lid,
    Sink,
    Dishwasher,
    Cooktop,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
    DegreeOfFreedom,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
    PrismaticConnection,
)
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room, Floor
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class KitchenEnvironment:
    """
    Manages the Kitchen Environment world with walls, furniture, and room layouts.
    """

    def get_world(self) -> World:
        """
        Constructs and returns a new World instance, setting up its environment,
        including walls, furniture, and rooms.

        :return: A new world instance with the initialized environment.
        """
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)

        self._build_environment_walls(world)
        self._build_environment_furniture(world)
        self._build_environment_rooms(world)

        return world

    def _build_environment_walls(self, world: World):
        """
        Builds and configures the environment walls for a given world. This involves creating
        various walls with predefined dimensions, transformation matrices, and connections.

        :param world: An instance representing the environment world where walls are to be
        configured and added.

        :return: The modified world instance with configured walls and connections.
        """
        root = world.root
        root_transformation = HomogeneousTransformationMatrix.from_xyz_rpy()

        with world.modify_world():
            south_wall1 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall1"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-2.01
                ),
                scale=Scale(x=0.05, y=1.00, z=3.00),
            )

            south_wall2 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall2"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-1.45, yaw=np.pi / 2
                ),
                scale=Scale(x=0.05, y=0.29, z=3.00),
            )

            south_wall3 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall3"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29, y=-0.9925
                ),
                scale=Scale(x=0.05, y=1.085, z=1.00),
            )

            south_wall4 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall4"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-0.45, yaw=np.pi / 2
                ),
                scale=Scale(x=0.05, y=0.29, z=1.00),
            )

            south_wall5 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall5"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=0.45, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 0.29, 1.00),
            )

            south_wall6 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall6"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=1.80
                ),
                scale=Scale(0.05, 2.75, 1.00),
            )

            south_wall7 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall7"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=5.16
                ),
                scale=Scale(0.05, 2.27, 1.00),
            )

            east_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("east_wall"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.462, y=-2.535, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 4.924, 3.00),
            )

            middle_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("middle_wall"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.20975, y=5.00
                ),
                scale=Scale(0.05, 2.67, 1.00),
            )

            west_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("west_wall"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.9345, y=6.32, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 4.449, 3.00),
            )

            north_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("north_wall"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.949, y=1.51
                ),
                scale=Scale(0.05, 8.04, 3.00),
            )

        north_west_wall = Cylinder(width=1.53, height=3.00)
        shape_geometry = ShapeCollection([north_west_wall])
        north_west_wall_body = Body(
            name=PrefixedName("north_west_wall_body"),
            collision=shape_geometry,
            visual=shape_geometry,
        )

        root_C_north_west_wall = FixedConnection(
            parent=root,
            child=north_west_wall_body,
            parent_T_connection_expression=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=4.924, y=6.295, z=1.50
            ),
        )

        with world.modify_world():
            world.add_connection(root_C_north_west_wall)
            return world

    def _build_environment_furniture(self, world: World):
        """
        Adds furniture items and room layouts (kitchen, living room, bedroom, office) to the scene graph.
        Connects furniture bodies and room structures hierarchically under the main root.
        Returns the updated World object with furniture integrated.
        """
        root = world.root
        root_transformation = HomogeneousTransformationMatrix.from_xyz_rpy()

        with world.modify_world():
            # --- DETAILED TRASH CAN ---
            trash_can = TrashCan.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("trash_can"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=0.416, y=5.5, z=0.2),
                scale=Scale(x=0.30, y=0.30, z=0.40),
                wall_thickness=0.02
            )
            for shape in trash_can.root.visual.shapes: shape.color = Color.GRAY()

            # Hinge
            trash_lid_hinge = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("trash_lid_hinge"),
                active_axis=Vector3.Y(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=-np.pi / 2), upper=DerivativeMap[float](position=0.0))
            )
            # Re-parent hinge to trash can
            trash_can.add_hinge(trash_lid_hinge)
            trash_lid_hinge.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.15, z=0.2)

            # Lid
            trash_lid = Lid.create_with_new_body_in_world(
                world=world, name=PrefixedName("trash_lid"),
                scale=Scale(x=0.30, y=0.30, z=0.02)
            )
            for shape in trash_lid.root.visual.shapes: shape.color = Color.BLACK()
            # Re-parent lid to hinge
            trash_lid.root.parent_connection.parent = trash_lid_hinge.root
            trash_lid.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.15, z=0.01)

            # --- DETAILED REFRIGERATOR (Standing on floor) ---
            fridge_length, fridge_width, fridge_height = 0.60, 0.658, 1.49

            # Create fridge
            refrigerator = Fridge.create_with_new_body_in_world(
                name=PrefixedName("refrigerator"),
                world=world,
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=0.537, y=-2.181,
                                                                                                     z=fridge_height / 2,
                                                                                                     yaw=-np.pi / 2),
                scale=Scale(x=fridge_length, y=fridge_width, z=fridge_height),
                wall_thickness=0.02
            )
            for shape in refrigerator.root.visual.shapes: shape.color = Color.GRAY()

            # Hinge for Door
            fridge_door_hinge = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("fridge_door_hinge"),
                active_axis=Vector3.Z(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0), upper=DerivativeMap[float](position=np.pi / 2))
            )
            door_height = (fridge_height - 0.08) * 0.75
            # Re-parent hinge to refrigerator
            refrigerator.add_hinge(fridge_door_hinge)
            fridge_door_hinge.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-fridge_length / 2, y=-fridge_width / 2, z=fridge_height / 2 - door_height / 2)

            # 1. Door (75% height)
            fridge_door = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("fridge_door"),
                scale=Scale(x=0.02, y=fridge_width, z=door_height)
            )
            for shape in fridge_door.root.visual.shapes: shape.color = Color.WHITE()
            # Re-parent door to hinge
            fridge_door.add_hinge(fridge_door_hinge)
            fridge_door.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(y=fridge_width / 2)
            refrigerator.add_door(fridge_door)

            # 2. Lower Drawer (25% height)
            drawer_height = (fridge_height - 0.08) * 0.25
            fridge_drawer = Drawer.create_with_new_body_in_world(
                world=world, name=PrefixedName("fridge_drawer"),
                scale=Scale(x=0.5, y=fridge_width - 0.04, z=drawer_height - 0.01),
                active_axis=Vector3.NEGATIVE_X(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0),
                                                        upper=DerivativeMap[float](position=0.5)))
            for shape in fridge_drawer.root.visual.shapes: shape.color = Color.WHITE()
            # Re-parent drawer to refrigerator
            refrigerator.add_drawer(fridge_drawer)
            fridge_drawer.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-fridge_length / 2 + 0.25, z=-fridge_height / 2 + 0.08 + drawer_height / 2)

            # 3. Handles
            # 3.1 Door Handle
            handle_bar_length = 0.5
            handle_thickness = 0.02
            handle_depth = 0.04
            
            fridge_door_handle = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("fridge_door_handle"),
                scale=Scale(x=handle_depth, y=handle_bar_length, z=handle_thickness),
                thickness=handle_thickness
            )
            for shape in fridge_door_handle.root.visual.shapes: shape.color = Color.GRAY()
            # Re-parent handle to door
            fridge_door.add_handle(fridge_door_handle)
            # Rotate by roll=np.pi/2 to make it vertical
            fridge_door_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.02, y=fridge_width / 2 - 0.03, roll=np.pi / 2)

            # 3.2 Drawer Handle
            fridge_drawer_handle = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("fridge_drawer_handle"),
                scale=Scale(x=0.04, y=0.5, z=0.02),
                thickness=0.02
            )
            for shape in fridge_drawer_handle.root.visual.shapes: shape.color = Color.GRAY()
            # Re-parent handle to drawer
            fridge_drawer.add_handle(fridge_drawer_handle)
            fridge_drawer_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.26, z=drawer_height / 2 - 0.03)

            # --- KITCHEN COUNTER  ---
            counter_top_length, counter_top_depth, counter_top_height = 2.044, 0.658, 0.6
            counter_top_root_transformation = root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=1.887, y=-2.181, z=counter_top_height / 2,
                                                                                           yaw=-np.pi / 2)

            # Place the plate on top of the modules (z = ct_h + plate_thickness/2)
            counter_top = CounterTop.create_with_new_body_in_world(
                world=world, name=PrefixedName("counter_top"),
                world_root_T_self=counter_top_root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(z=counter_top_height / 2 + 0.02),
                scale=Scale(counter_top_depth, counter_top_length, 0.04))
            for shape in counter_top.root.visual.shapes: shape.color = Color.BEIGE()

            # 0. Sink
            sink = Sink.create_with_new_body_in_world(
                world=world, name=PrefixedName("sink"),
                scale=Scale(0.4, 0.6, 0.005)
            )
            for shape in sink.root.visual.shapes: shape.color = Color.BLACK()
            sink_connection = sink.root.parent_connection
            world.remove_connection(sink_connection)
            sink_connection.parent = counter_top.root
            sink_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.7, z=0.025)
            world.add_connection(sink_connection)

            module_1_width, module_2_width = 0.60, 0.55
            module_3_width = counter_top_length - module_1_width - module_2_width

            # 1. Module 1: Cabinet
            module_1_y_position = -counter_top_length / 2 + module_1_width / 2
            module_1_cabinet = Cabinet.create_with_new_body_in_world(
                world=world, name=PrefixedName("module_1_cabinet"),
                world_root_T_self=counter_top_root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(y=module_1_y_position),
                scale=Scale(counter_top_depth, module_1_width, counter_top_height), wall_thickness=0.02)
            for shape in module_1_cabinet.root.visual.shapes: shape.color = Color.GRAY()

            module_1_hinge = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("module_1_hinge"),
                active_axis=Vector3.Z(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0), upper=DerivativeMap[float](position=np.pi / 2))
            )
            module_1_cabinet.add_hinge(module_1_hinge)
            module_1_hinge.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-counter_top_depth / 2, y=-module_1_width / 2, z=0)

            module_1_door = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("module_1_door"),
                scale=Scale(x=0.02, y=module_1_width, z=counter_top_height)
            )
            for shape in module_1_door.root.visual.shapes: shape.color = Color.WHITE()
            module_1_door.add_hinge(module_1_hinge)
            module_1_door.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(y=module_1_width / 2)
            module_1_cabinet.add_door(module_1_door)

            # Horizontal U-Handle for M1
            module_1_handle = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("module_1_handle"),
                scale=Scale(x=handle_depth, y=module_1_width - 0.06, z=handle_thickness),
                thickness=handle_thickness
            )
            for shape in module_1_handle.root.visual.shapes: shape.color = Color.GRAY()
            module_1_door.add_handle(module_1_handle)
            module_1_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.02, z=counter_top_height / 2 - 0.05)

            # 2. Module 2: Dishwasher
            dishwasher_y_position = -counter_top_length / 2 + module_1_width + module_2_width / 2
            dishwasher = Dishwasher.create_with_new_body_in_world(
                world=world, name=PrefixedName("dishwasher"),
                world_root_T_self=counter_top_root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(y=dishwasher_y_position),
                scale=Scale(x=counter_top_depth, y=module_2_width, z=counter_top_height), wall_thickness=0.02)
            for shape in dishwasher.root.visual.shapes: shape.color = Color.GRAY()

            dishwasher_hinge = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("dishwasher_hinge"),
                active_axis=Vector3.NEGATIVE_Y(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0), upper=DerivativeMap[float](position=np.pi / 2))
            )
            dishwasher.add_hinge(dishwasher_hinge)
            dishwasher_hinge.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-counter_top_depth / 2, z=-counter_top_height / 2)

            dishwasher_door = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("dishwasher_door"),
                scale=Scale(x=0.02, y=module_2_width, z=counter_top_height)
            )
            for shape in dishwasher_door.root.visual.shapes: shape.color = Color.WHITE()
            dishwasher_door.add_hinge(dishwasher_hinge)
            dishwasher_door.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(z=counter_top_height / 2)
            dishwasher.add_door(dishwasher_door)

            # Horizontal U-Handle for DW (at Top)
            dishwasher_handle = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("dishwasher_handle"),
                scale=Scale(x=handle_depth, y=module_2_width - 0.06, z=handle_thickness),
                thickness=handle_thickness
            )
            for shape in dishwasher_handle.root.visual.shapes: shape.color = Color.GRAY()
            dishwasher_door.add_handle(dishwasher_handle)
            dishwasher_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.02, z=counter_top_height / 2 - 0.03)

            # 3. Module 3: Hollow Cabinet with Drawers (40/40/20)
            module_3_y_position = counter_top_length / 2 - module_3_width / 2
            module_3_cabinet = Cabinet.create_with_new_body_in_world(
                world=world, name=PrefixedName("module_3_cabinet"),
                world_root_T_self=counter_top_root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(y=module_3_y_position),
                scale=Scale(x=counter_top_depth, y=module_3_width, z=counter_top_height), wall_thickness=0.02)
            for shape in module_3_cabinet.root.visual.shapes: shape.color = Color.GRAY()

            # Correct hierarchy
            module_3_cabinet.root.parent_connection.parent = counter_top.root
            # Move module down relative to the plate (plate is at +0.32 relative to module center)
            module_3_cabinet.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(y=module_3_y_position,
                                                                                                              z=-(counter_top_height / 2 + 0.02))

            drawer_bottom_height, drawer_middle_height, drawer_top_height = counter_top_height * 0.4, counter_top_height * 0.4, counter_top_height * 0.2
            drawer_z_positions = [-counter_top_height / 2 + drawer_bottom_height / 2, -counter_top_height / 2 + drawer_bottom_height + drawer_middle_height / 2, counter_top_height / 2 - drawer_top_height / 2]
            drawer_heights = [drawer_bottom_height, drawer_middle_height, drawer_top_height]
            for index, (drawer_height, z_position) in enumerate(zip(drawer_heights, drawer_z_positions)):
                drawer_id = f"counter_drawer_{index}"
                drawer = Drawer.create_with_new_body_in_world(
                    world=world, name=PrefixedName(drawer_id),
                    scale=Scale(x=0.3, y=module_3_width - 0.04, z=drawer_height - 0.01),
                    active_axis=Vector3.NEGATIVE_X(),
                    connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0),
                                                            upper=DerivativeMap[float](position=0.25)))
                for shape in drawer.root.visual.shapes: shape.color = Color.WHITE()

                # Correct hierarchy
                module_3_cabinet.add_drawer(drawer)
                drawer.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2 + 0.15, z=z_position)

                # Handle
                drawer_handle = Handle.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"{drawer_id}_handle"),
                    scale=Scale(x=handle_depth, y=module_3_width - 0.06, z=handle_thickness),
                    thickness=handle_thickness
                )
                for shape in drawer_handle.root.visual.shapes: shape.color = Color.GRAY()
                drawer.add_handle(drawer_handle)
                drawer_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.16, z=drawer_height / 2 - 0.03)

            # --- OVEN TOWER (Final Corrected Framework Implementation) ---
            oven_tower_width, oven_tower_depth, oven_tower_height = 1.20, 0.658, 1.49
            oven_tower_root_transformation = root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=3.51, y=-2.181, z=oven_tower_height / 2,
                                                                                           yaw=-np.pi / 2)

            oven_tower = Cupboard.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_tower"),
                world_root_T_self=oven_tower_root_transformation, scale=Scale(x=oven_tower_depth, y=oven_tower_width, z=oven_tower_height), wall_thickness=0.02)
            for shape in oven_tower.root.visual.shapes: shape.color = Color.GRAY()

            module_center_width, module_side_width = 0.60, 0.30
            cabinet_height, drawer_height = 0.60, 0.15
            oven_height = oven_tower_height - cabinet_height - drawer_height

            # 2.1 Side Drawers (Left & Right)
            for side in [-1, 1]:
                side_name = "left" if side == -1 else "right"
                side_drawer = Drawer.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"oven_side_drawer_{side_name}"),
                    scale=Scale(x=oven_tower_depth, y=module_side_width, z=oven_tower_height), active_axis=Vector3.NEGATIVE_X())
                for shape in side_drawer.root.visual.shapes: shape.color = Color.WHITE()
                oven_tower.add_drawer(side_drawer)
                side_drawer.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=side * (module_center_width / 2 + module_side_width / 2))

                # Vertical U-Handle
                side_handle_length = oven_tower_height - 0.08
                side_handle = Handle.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"oven_side_handle_{side_name}"),
                    scale=Scale(x=handle_depth, y=side_handle_length, z=handle_thickness),
                    thickness=handle_thickness
                )
                for shape in side_handle.root.visual.shapes: shape.color = Color.GRAY()
                side_drawer.add_handle(side_handle)
                side_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_tower_depth / 2, roll=np.pi / 2
                )

            # 2.2 Center Section: Bottom Cabinet
            oven_cabinet_hinge = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_cabinet_hinge"),
                active_axis=Vector3.Z(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0), upper=DerivativeMap[float](position=np.pi / 2))
            )
            oven_tower.add_hinge(oven_cabinet_hinge)
            oven_cabinet_hinge.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-oven_tower_depth / 2, y=module_center_width / 2, z=-oven_tower_height / 2 + cabinet_height / 2)

            oven_cabinet_door = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_cabinet_door"),
                scale=Scale(x=0.02, y=module_center_width, z=cabinet_height)
            )
            for shape in oven_cabinet_door.root.visual.shapes: shape.color = Color.WHITE()
            oven_cabinet_door.add_hinge(oven_cabinet_hinge)
            oven_cabinet_door.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(y=-module_center_width / 2)
            oven_tower.add_door(oven_cabinet_door)

            # Horizontal U-Handle for Cabinet
            oven_cabinet_handle = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_cabinet_handle"),
                scale=Scale(x=handle_depth, y=module_center_width - 0.06, z=handle_thickness),
                thickness=handle_thickness
            )
            for shape in oven_cabinet_handle.root.visual.shapes: shape.color = Color.GRAY()
            oven_cabinet_door.add_handle(oven_cabinet_handle)
            oven_cabinet_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.02, z=cabinet_height / 2 - 0.05)

            # 2.3 Center Section: Middle Drawer
            oven_center_drawer = Drawer.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_center_drawer"),
                scale=Scale(x=0.3, y=module_center_width - 0.04, z=drawer_height - 0.01),
                active_axis=Vector3.NEGATIVE_X(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0),
                                                        upper=DerivativeMap[float](position=0.25)))
            for shape in oven_center_drawer.root.visual.shapes: shape.color = Color.WHITE()

            # Correct hierarchy
            oven_tower.add_drawer(oven_center_drawer)
            oven_center_drawer.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-oven_tower_depth / 2 + 0.15,
                                                                                                      z=-oven_tower_height / 2 + cabinet_height + drawer_height / 2)

            # 2.4 Center Section: Oven (Top)
            oven_hinge = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_hinge"),
                active_axis=Vector3.NEGATIVE_Y(),
                connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0), upper=DerivativeMap[float](position=np.pi / 2))
            )
            oven_tower.add_hinge(oven_hinge)
            oven_hinge.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-oven_tower_depth / 2, z=oven_tower_height / 2 - oven_height)

            oven_door = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_door"),
                scale=Scale(x=0.02, y=module_center_width, z=oven_height)
            )
            # Custom geometry with glass window
            oven_frame_geometry = Box(scale=Scale(x=0.02, y=module_center_width, z=oven_height), color=Color.WHITE())
            oven_glass_geometry = Box(scale=Scale(x=0.005, y=0.35, z=0.35), color=Color.BLACK())
            oven_glass_geometry.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.011)
            oven_door_geometry = ShapeCollection([oven_frame_geometry, oven_glass_geometry], reference_frame=oven_door.root)
            oven_door_geometry.transform_all_shapes_to_own_frame()
            oven_door.root.collision = oven_door_geometry
            oven_door.root.visual = oven_door_geometry

            oven_door.add_hinge(oven_hinge)
            oven_door.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(z=oven_height / 2)
            oven_tower.add_door(oven_door)

            # Horizontal U-Handle for Oven
            oven_handle = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("oven_handle"),
                scale=Scale(x=handle_depth, y=module_center_width - 0.06, z=handle_thickness),
                thickness=handle_thickness
            )
            for shape in oven_handle.root.visual.shapes: shape.color = Color.GRAY()
            oven_door.add_handle(oven_handle)
            oven_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.02, z=oven_height / 2 - 0.05)

            # --- SIDEBOARD / KITCHEN ISLAND ---
            sideboard_length, sideboard_width, sideboard_height = 2.45, 0.796, 0.845
            sideboard_thickness = 0.04
            # Position z=sideboard_height/2, yaw=-np.pi/2. Moved y=0.2 to avoid sofa intersection
            # Local -X is now facing into the room (+Y in world)
            sideboard_root_transformation = root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=3.545, y=0.2, z=sideboard_height / 2,
                                                                                           yaw=np.pi / 2)

            # 1. Top Plate (Root)
            sideboard = Table.create_with_new_body_in_world(
                world=world, name=PrefixedName("sideboard"),
                world_root_T_self=sideboard_root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(z=sideboard_height / 2 - sideboard_thickness / 2),
                scale=Scale(sideboard_width, sideboard_length, sideboard_thickness))
            for shape in sideboard.root.visual.shapes: shape.color = Color.WHITE()

            # 2. Main Body (Hollow Cabinet, Open towards local -X)
            sideboard_cabinet = Cabinet.create_with_new_body_in_world(
                world=world, name=PrefixedName("sideboard_cabinet"),
                world_root_T_self=sideboard_root_transformation, scale=Scale(sideboard_width, sideboard_length, sideboard_height), wall_thickness=0.02)
            for shape in sideboard_cabinet.root.visual.shapes: shape.color = Color.WHITE()

            # 3. Cooktop (Ceran-Feld) on the Top Plate (on the right side in world coordinates)
            cooktop = Cooktop.create_with_new_body_in_world(
                world=world, name=PrefixedName("sideboard_cooktop"),
                scale=Scale(x=0.5, y=0.6, z=0.005)
            )
            for shape in cooktop.root.visual.shapes: shape.color = Color.BLACK()
            sideboard.add_object(cooktop)
            cooktop.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                y=-0.7, z=sideboard_thickness / 2 + 0.001)

            # 4. Drawer Layout (3x2 Grid on local -X face)
            width_outer, width_middle = sideboard_length * 0.3, sideboard_length * 0.4
            widths = [width_outer, width_middle, width_outer]
            y_offsets = [-sideboard_length / 2 + width_outer / 2, 0, sideboard_length / 2 - width_outer / 2]
            drawer_height = (sideboard_height - 0.15) / 2
            z_offsets = [-sideboard_height / 2 + 0.05 + drawer_height / 2, -sideboard_height / 2 + 0.05 + 3 * drawer_height / 2]

            for column_index, (w, y_offset) in enumerate(zip(widths, y_offsets)):
                for row_index, z_offset in enumerate(z_offsets):
                    drawer_id = f"sideboard_drawer_{column_index}_{row_index}"
                    drawer = Drawer.create_with_new_body_in_world(
                        world=world, name=PrefixedName(drawer_id),
                        world_root_T_self=sideboard_root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=-sideboard_width / 2 + 0.2,
                                                                                                   y=y_offset, z=z_offset),
                        scale=Scale(0.4, w - 0.01, drawer_height - 0.01),
                        active_axis=Vector3.NEGATIVE_X(),
                        connection_limits=DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0),
                                                                upper=DerivativeMap[float](position=0.25)))
                    for shape in drawer.root.visual.shapes: shape.color = Color.WHITE()

                    # Reconnect to sideboard body for hierarchy
                    drawer_connection = drawer.root.parent_connection
                    world.remove_connection(drawer_connection)
                    drawer_connection.parent = sideboard_cabinet.root
                    # Set the connection pose relative to the new parent (sideboard body)
                    drawer_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-sideboard_width / 2 + 0.2, y=y_offset, z=z_offset)
                    world.add_connection(drawer_connection)
                    sideboard_cabinet.add_drawer(drawer)

                    # Horizontal U-Handle
                    drawer_handle = Handle.create_with_new_body_in_world(
                        world=world, name=PrefixedName(f"{drawer_id}_handle"),
                        scale=Scale(handle_depth, w - 0.1, handle_thickness),
                        thickness=handle_thickness
                    )
                    for shape in drawer_handle.root.visual.shapes: shape.color = Color.GRAY()
                    handle_connection = drawer_handle.root.parent_connection
                    world.remove_connection(handle_connection)
                    handle_connection.parent = drawer.root
                    handle_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.2, z=drawer_height / 2 - 0.05)
                    world.add_connection(handle_connection)
                    drawer.handle = drawer_handle

            sofa = Sofa.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sofa"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=3.60, y=1.20, z=0.34,
                                                                                                     yaw=4.7124),
                scale=Scale(x=0.94, y=1.68, z=0.68),
            )
            for color in sofa.bodies[0].visual.shapes:
                color.color = Color.BEIGE()

            # --- REFINED COFFEE TABLE (White, Front-Closed, with Floor) ---
            coffee_table_length, coffee_table_width, coffee_table_height = 0.37, 0.91, 0.44
            coffee_table_thickness = 0.02
            coffee_table_color = Color.WHITE()
            coffee_table_root_transformation = root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=4.22, y=2.22, z=coffee_table_height,
                                                                                           yaw=np.pi)

            coffee_table = Table.create_with_new_body_in_world(
                world=world, name=PrefixedName("coffee_table"),
                world_root_T_self=coffee_table_root_transformation, scale=Scale(coffee_table_length, coffee_table_width, coffee_table_thickness))
            for shape in coffee_table.bodies[0].visual.shapes: shape.color = coffee_table_color

            # Middle Shelf
            coffee_table_shelf = ShelfLayer.create_with_new_body_in_world(
                world=world, name=PrefixedName("coffee_table_shelf"),
                scale=Scale(coffee_table_length, coffee_table_width, 0.01)
            )
            for shape in coffee_table_shelf.root.visual.shapes: shape.color = coffee_table_color
            shelf_connection = coffee_table_shelf.root.parent_connection
            world.remove_connection(shelf_connection)
            shelf_connection.parent = coffee_table.root
            shelf_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(z=-coffee_table_height / 2)
            world.add_connection(shelf_connection)

            # Bottom Plate (Floor)
            coffee_table_floor = ShelfLayer.create_with_new_body_in_world(
                world=world, name=PrefixedName("coffee_table_floor"),
                scale=Scale(coffee_table_length, coffee_table_width, coffee_table_thickness)
            )
            for shape in coffee_table_floor.root.visual.shapes: shape.color = coffee_table_color
            floor_connection = coffee_table_floor.root.parent_connection
            world.remove_connection(floor_connection)
            floor_connection.parent = coffee_table.root
            floor_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(z=-coffee_table_height + coffee_table_thickness / 2)
            world.add_connection(floor_connection)

            # Walls (Supporting structure) - Both short sides closed
            for i, y_dir in enumerate([-1, 1]):
                coffee_table_side_wall_body = Body(name=PrefixedName(f"coffee_table_wall_short_{i}_body"))
                side_wall_geometry = ShapeCollection([Box(scale=Scale(coffee_table_length, coffee_table_thickness, coffee_table_height), color=coffee_table_color)],
                                                 reference_frame=coffee_table_side_wall_body)
                side_wall_geometry.transform_all_shapes_to_own_frame()
                coffee_table_side_wall_body.collision, coffee_table_side_wall_body.visual = side_wall_geometry, side_wall_geometry
                world.add_connection(FixedConnection(parent=coffee_table.root, child=coffee_table_side_wall_body,
                                                     parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                                                         y=y_dir * (coffee_table_width / 2 - coffee_table_thickness / 2), z=-coffee_table_height / 2)))

            # 2. Long Sides (1/3 closed at the front, 2/3 open at the back)
            wall_length = coffee_table_width / 3
            for side in [-1, 1]:
                side_name = "left" if side == -1 else "right"
                coffee_table_long_wall_body = Body(name=PrefixedName(f"coffee_table_wall_long_{side_name}_body"))
                long_wall_geometry = ShapeCollection([Box(scale=Scale(coffee_table_thickness, wall_length, coffee_table_height), color=coffee_table_color)],
                                                 reference_frame=coffee_table_long_wall_body)
                long_wall_geometry.transform_all_shapes_to_own_frame()
                coffee_table_long_wall_body.collision, coffee_table_long_wall_body.visual = long_wall_geometry, long_wall_geometry
                # Positioned at +y side (front)
                world.add_connection(FixedConnection(parent=coffee_table.root, child=coffee_table_long_wall_body,
                                                     parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                                                         x=side * (coffee_table_length / 2 - coffee_table_thickness / 2), y=coffee_table_width / 2 - wall_length / 2,
                                                         z=-coffee_table_height / 2)))

            # --- Cupboard (tall cabinet with doors) ---
            cupboard_scale = Scale(0.43, 0.80, 2.02)

            cupboard = Cupboard.create_with_new_body_in_world(
                name=PrefixedName("cupboard"),
                world=world,
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=4.55, y=4.72,
                                                                                                     z=1.01),
                scale=cupboard_scale,
                wall_thickness=0.02,
            )

            # create shelflayers manually and attach them directly to the cupboard
            shelf_scale = Scale(0.40, 0.76, 0.02)

            # Shelf 1
            cupboard_shelf_1 = ShelfLayer.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_shelf_1"),
                scale=shelf_scale
            )
            for shape in cupboard_shelf_1.root.visual.shapes: shape.color = Color.WHITE()
            shelf_connection = cupboard_shelf_1.root.parent_connection
            world.remove_connection(shelf_connection)
            shelf_connection.parent = cupboard.root
            shelf_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=-0.5)
            world.add_connection(shelf_connection)
            cupboard.add_shelf_layer(cupboard_shelf_1)

            # Shelf 2
            cupboard_shelf_2 = ShelfLayer.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_shelf_2"),
                scale=shelf_scale
            )
            for shape in cupboard_shelf_2.root.visual.shapes: shape.color = Color.WHITE()
            shelf_connection = cupboard_shelf_2.root.parent_connection
            world.remove_connection(shelf_connection)
            shelf_connection.parent = cupboard.root
            shelf_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0, z=0.5)
            world.add_connection(shelf_connection)
            cupboard.add_shelf_layer(cupboard_shelf_2)

            # Creating doors manually and attaching them directly to the cupboard
            # Door height 105.5 cm (1.055 m)
            door_height = 1.055
            # Position Z: Bottom of cupboard is at -cupboard_scale.z / 2.
            # Door center should be at Bottom + door_height / 2
            door_z_relative = -(cupboard_scale.z / 2) + (door_height / 2)

            door_x_relative = -(cupboard_scale.x / 2) - 0.01
            door_scale = Scale(0.02, 0.40, door_height)

            # Define limits for doors
            # Left door opens outwards (0 to +90 degrees)
            left_door_limits = DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0), upper=DerivativeMap[float](position=np.pi / 2))

            # Right door opens outwards (-90 to 0 degrees)
            right_door_limits = DegreeOfFreedomLimits(lower=DerivativeMap[float](position=-np.pi / 2), upper=DerivativeMap[float](position=0.0))

            # Left Door (Open via Hinge)
            # Create Hinge for the left door
            cupboard_hinge_left = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_hinge_left"),
                active_axis=Vector3.Z(), connection_limits=left_door_limits
            )
            hinge_connection = cupboard_hinge_left.root.parent_connection
            world.remove_connection(hinge_connection)
            hinge_connection.parent = cupboard.root
            hinge_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=door_x_relative, y=-0.40, z=door_z_relative)
            world.add_connection(hinge_connection)
            cupboard.add_hinge(cupboard_hinge_left)

            # Create left door
            cupboard_door_left = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_door_left"),
                scale=door_scale
            )
            for shape in cupboard_door_left.root.visual.shapes: shape.color = Color.WHITE()
            door_connection = cupboard_door_left.root.parent_connection
            world.remove_connection(door_connection)
            door_connection.parent = cupboard_hinge_left.root
            door_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=0.20, z=0)
            world.add_connection(door_connection)
            cupboard_door_left.add_hinge(cupboard_hinge_left)
            cupboard.add_door(cupboard_door_left)

            # Handle for Left Door
            cupboard_handle_left = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_handle_left"),
                scale=Scale(0.04, 0.04, 0.04), thickness=0.02
            )
            for shape in cupboard_handle_left.root.visual.shapes: shape.color = Color.GRAY()
            handle_connection = cupboard_handle_left.root.parent_connection
            world.remove_connection(handle_connection)
            handle_connection.parent = cupboard_door_left.root
            handle_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.03, y=0.15, z=0)
            world.add_connection(handle_connection)
            cupboard_door_left.handle = cupboard_handle_left

            # Right Door (Closed via Hinge)
            cupboard_hinge_right = Hinge.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_hinge_right"),
                active_axis=Vector3.Z(), connection_limits=right_door_limits
            )
            hinge_connection = cupboard_hinge_right.root.parent_connection
            world.remove_connection(hinge_connection)
            hinge_connection.parent = cupboard.root
            hinge_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=door_x_relative, y=0.40, z=door_z_relative)
            world.add_connection(hinge_connection)
            cupboard.add_hinge(cupboard_hinge_right)

            cupboard_door_right = Door.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_door_right"),
                scale=door_scale
            )
            for shape in cupboard_door_right.root.visual.shapes: shape.color = Color.WHITE()
            door_connection = cupboard_door_right.root.parent_connection
            world.remove_connection(door_connection)
            door_connection.parent = cupboard_hinge_right.root
            door_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=0, y=-0.20, z=0)
            world.add_connection(door_connection)
            cupboard_door_right.add_hinge(cupboard_hinge_right)
            cupboard.add_door(cupboard_door_right)

            # Handle for Right Door
            cupboard_handle_right = Handle.create_with_new_body_in_world(
                world=world, name=PrefixedName("cupboard_handle_right"),
                scale=Scale(0.04, 0.04, 0.04), thickness=0.02
            )
            for shape in cupboard_handle_right.root.visual.shapes: shape.color = Color.GRAY()
            handle_connection = cupboard_handle_right.root.parent_connection
            world.remove_connection(handle_connection)
            handle_connection.parent = cupboard_door_right.root
            handle_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.03, y=-0.15, z=0)
            world.add_connection(handle_connection)
            cupboard_door_right.handle = cupboard_handle_right

            # Detailed White Desk Construction
            desk_length, desk_width, desk_height = 0.60, 1.20, 0.75
            desk_color = Color.WHITE()
            desk_plate_thickness = 0.03

            desk = Desk.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("desk"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=0.05, y=1.28,
                                                                                                     z=desk_height),
                scale=Scale(desk_length, desk_width, desk_plate_thickness),
            )
            for shape in desk.root.visual.shapes: shape.color = desk_color

            leg_scale = Scale(0.04, 0.04, desk_height - desk_plate_thickness)
            x_offset = (desk_length / 2) - 0.02
            y_offset = (desk_width / 2) - 0.02
            z_position = -(desk_plate_thickness / 2) - (leg_scale.z / 2)

            for i, (sign_x, sign_y) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                desk_leg = Leg.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"desk_leg_{i}"),
                    scale=leg_scale
                )
                for shape in desk_leg.root.visual.shapes: shape.color = desk_color
                leg_connection = desk_leg.root.parent_connection
                world.remove_connection(leg_connection)
                leg_connection.parent = desk.root
                leg_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=sign_x * x_offset, y=sign_y * y_offset, z=z_position
                )
                world.add_connection(leg_connection)
                desk.add_leg(desk_leg)

            # --- MODULAR COOKING TABLE ---
            cooking_table_length, cooking_table_depth, cooking_table_height, cooking_table_thickness = 1.75, 0.64, 0.71, 0.04
            # 1. Top Layer (The Worktop)
            cooking_table = Table.create_with_new_body_in_world(world=world, name=PrefixedName("cooking_table"),
                                                                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                                                                    x=1.28, y=5.99, z=cooking_table_height),
                                                                scale=Scale(cooking_table_length, cooking_table_depth, cooking_table_thickness))
            for shape in cooking_table.bodies[0].visual.shapes: shape.color = Color.BEIGE()

            # Ceran Field
            cooktop = Cooktop.create_with_new_body_in_world(
                world=world, name=PrefixedName("cooktop"),
                scale=Scale(x=0.5, y=0.5, z=0.01)
            )
            for shape in cooktop.root.visual.shapes: shape.color = Color.BLACK()
            cooking_table.add_object(cooktop)
            cooktop.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                z=cooking_table_thickness / 2 + 0.005)

            # 2. Bottom Layer (The Support)
            cooking_table_bottom_body = Body(name=PrefixedName("cooking_table_bottom_body"))
            cooking_table_bottom_geometry = ShapeCollection([Box(scale=Scale(cooking_table_length, cooking_table_depth, cooking_table_thickness), color=Color.BEIGE())],
                                             reference_frame=cooking_table_bottom_body)
            cooking_table_bottom_geometry.transform_all_shapes_to_own_frame()
            cooking_table_bottom_body.collision, cooking_table_bottom_body.visual = cooking_table_bottom_geometry, cooking_table_bottom_geometry
            world.add_connection(FixedConnection(parent=cooking_table.root, child=cooking_table_bottom_body,
                                                 parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                                                     z=-cooking_table_height + cooking_table_thickness)))

            # 3. Side Modules (Cupboards with Drawers)
            cooking_module_width = (cooking_table_length - 0.60) / 2
            cooking_drawer_limits = DegreeOfFreedomLimits(lower=DerivativeMap[float](position=0.0),
                                              upper=DerivativeMap[float](position=0.40))
            for side in [-1, 1]:
                side_name = "left" if side == -1 else "right"
                # Module Cupboard
                mod_cupboard = Cupboard.create_with_new_body_in_world(name=PrefixedName(f"cooking_mod_{side_name}"), world=world,
                                                                      scale=Scale(x=cooking_module_width, y=cooking_table_depth, z=cooking_table_height - 2 * cooking_table_thickness))
                for shape in mod_cupboard.bodies[0].visual.shapes: shape.color = Color.BEIGE()
                cooking_table.add_object(mod_cupboard)
                mod_cupboard.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                                                         x=side * (0.265 + cooking_module_width / 2), z=-cooking_table_height / 2 + cooking_table_thickness, yaw=1.5708)

                # Drawer in Module
                drawer = Drawer.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"cooking_drawer_{side_name}"),
                    scale=Scale(x=cooking_module_width - 0.04, y=cooking_table_depth - 0.02, z=0.18),
                    active_axis=Vector3.NEGATIVE_X(),
                    connection_limits=cooking_drawer_limits)
                for shape in drawer.root.visual.shapes: shape.color = Color.BEIGE()

                # Correct hierarchy
                mod_cupboard.add_drawer(drawer)
                drawer.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(z=0.2)

                # Drawer Handle (Rectangular)
                cooking_drawer_handle = Handle.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"cooking_drawer_handle_{side_name}"),
                    scale=Scale(x=handle_depth, y=cooking_module_width / 3, z=0.04),
                    thickness=0.02
                )
                for shape in cooking_drawer_handle.root.visual.shapes: shape.color = Color.GRAY()
                drawer.add_handle(cooking_drawer_handle)
                cooking_drawer_handle.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(x=-cooking_module_width / 2 + 0.02)

                # Shelf below Drawer
                cooking_shelf = ShelfLayer.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"cooking_shelf_{side_name}"),
                    scale=Scale(x=cooking_module_width - 0.04, y=cooking_table_depth - 0.02, z=0.02)
                )
                for shape in cooking_shelf.root.visual.shapes: shape.color = Color.WHITE()
                mod_cupboard.add_shelf_layer(cooking_shelf)
                cooking_shelf.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.1)

            # Dining Table Construction
            dining_table_length, dining_table_width, dining_table_height = 0.73, 1.18, 0.76
            dining_table_color = Color.BEIGE()
            dining_table_plate_thickness = 0.04

            dining_table = DiningTable.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("dining_table"),
                world_root_T_self=root_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(x=2.59975, y=5.705,
                                                                                                     z=dining_table_height),
                scale=Scale(dining_table_length, dining_table_width, dining_table_plate_thickness),
            )
            for shape in dining_table.root.visual.shapes: shape.color = dining_table_color

            leg_scale = Scale(0.06, 0.06, dining_table_height - dining_table_plate_thickness)
            x_offset = (dining_table_length / 2) - 0.03
            y_offset = (dining_table_width / 2) - 0.03
            z_position = -(dining_table_plate_thickness / 2) - (leg_scale.z / 2)

            for i, (sign_x, sign_y) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                dining_table_leg = Leg.create_with_new_body_in_world(
                    world=world, name=PrefixedName(f"dining_table_leg_{i}"),
                    scale=leg_scale
                )
                for shape in dining_table_leg.root.visual.shapes: shape.color = dining_table_color
                dining_table.add_leg(dining_table_leg)
                dining_table_leg.root.parent_connection.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=sign_x * x_offset, y=sign_y * y_offset, z=z_position
                )

        return world

    def _build_environment_rooms(self, world: World):
        room_annotations = []

        root_transformation = HomogeneousTransformationMatrix.from_xyz_rpy()

        with world.modify_world():
            kitchen_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 3.334, 0),
                Point3(5.214, 3.334, 0),
                Point3(5.214, 0, 0),
            ]

            living_room_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 2.971, 0),
                Point3(5.214, 2.971, 0),
                Point3(5.214, 0, 0),
            ]

            bed_room_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 2.67, 0.0),
                Point3(2.50, 2.67, 0.0),
                Point3(2.50, 0, 0.0),
            ]

            office_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 2.67, 0),
                Point3(2.71, 2.67, 0),
                Point3(2.71, 0, 0),
            ]

            kitchen_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("kitchen_floor"),
                world=world,
                floor_polytope=kitchen_floor_polytope,
                world_root_T_self=root_transformation
                                  @ HomogeneousTransformationMatrix.from_xyz_rpy(x=2.317, y=-0.843),
            )
            kitchen = Room(floor=kitchen_floor, name=PrefixedName("kitchen"))
            room_annotations.append(kitchen)

            living_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("living_room_floor"),
                world=world,
                floor_polytope=living_room_floor_polytope,
                world_root_T_self=root_transformation
                                  @ HomogeneousTransformationMatrix.from_xyz_rpy(x=2.317, y=2.3095),
            )
            living_room = Room(floor=living_room_floor, name=PrefixedName("living_room"))
            room_annotations.append(living_room)

            bed_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("bed_room_floor"),
                world=world,
                floor_polytope=bed_room_floor_polytope,
                world_root_T_self=root_transformation
                                  @ HomogeneousTransformationMatrix.from_xyz_rpy(x=0.96, y=4.96),
            )
            bed_room = Room(floor=bed_room_floor, name=PrefixedName("bed_room"))
            room_annotations.append(bed_room)

            office_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("office_floor"),
                world=world,
                floor_polytope=office_floor_polytope,
                world_root_T_self=root_transformation
                                  @ HomogeneousTransformationMatrix.from_xyz_rpy(x=3.56, y=4.96),
            )
            office = Room(floor=office_floor, name=PrefixedName("office"))
            room_annotations.append(office)

            world.add_semantic_annotations(room_annotations)

        return world
