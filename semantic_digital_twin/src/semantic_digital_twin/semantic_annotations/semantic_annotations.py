from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Iterable, Iterator, Self, TYPE_CHECKING

import math

import numpy as np

from krrood.ormatic.utils import classproperty
from krrood.symbolic_math import symbolic_math
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    HandleWiderThanLong,
    NoGraspGeometry,
    HandleWiderThanLong,
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    MissingSemanticAnnotationError,
    MissingMovableJointError,
)
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasBackplates,
    HasCables,
    HasCarryingCables,
    HasBlinds,
    HasButtons,
    HasCaps,
    HasCarryingHoses,
    HasCarryingShowerValves,
    HasControls,
    HasCooktops,
    HasCurtains,
    HasDrainPipes,
    HasFaucetSpouts,
    HasFaucets,
    HasHeaterValves,
    HasHinges,
    HasHoses,
    HasKnobs,
    HasLidFrames,
    HasLids,
    HasLocks,
    HasPowerSockets,
    HasPumps,
    HasShowerHeads,
    HasShowerHoses,
    HasShowers,
    HasSinks,
    HasSwitches,
    HasThermostats,
    HasToiletFlushes,
    HasToiletSeats,
    HasWheels,
    HasWindows,
    HasCounterTop,
    HasCabinets,
    HasSupportingSurface,
    HasRootRegion,
    HasDrawers,
    HasDoors,
    HasHandles,
    HasCaseAsRootBody,
    HasMovableJoint,
    HasApertures,
    IsPerceivable,
    HasRootBody,
    IsStorageSpace,
    HasLegs,
    HasSink,
    HasShelfLayers,
    HasGraspCandidates,
    HasPinchableWall,
    HasRim,
)
from krrood.entity_query_language.factories import and_, or_
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.surface_grasp import (
    SurfaceGrasp,
    any_surface_grasp,
    from_any_side,
)
from krrood_spatial_math.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
)
from krrood_spatial_math.spatial_types import Pose, RotationMatrix
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    Scale,
    Color,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from semantic_digital_twin.specifications.connections import ConnectionSpecification
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
    KinematicStructureEntitySpecification,
    RegionSpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(eq=False)
class Furniture(HasRootBody, ABC):
    """
    Furniture in general, a category to answer with the *kind* of rather than by name:
    a chair, a table, a shelf, or a new subclass of this where the ontology has none.

    Every piece of furniture stands somewhere, so it is rooted on a body here rather
    than in each of its subclasses. Most of them reach a root anyway through a mixin of
    their own -- a cabinet through :class:`HasCaseAsRootBody`, a table through
    :class:`HasSupportingSurface` -- and those are unaffected, since they already
    derived from :class:`HasRootBody` by that route and still reach their own geometry
    first. The ones this settles are the pieces named by nothing else: a chair, a bed.
    """


@dataclass(eq=False)
class Handle(HasGraspCandidates, HasMovableJoint[RevoluteConnection]):
    """
    A handle is a physical entity that can be grasped by a hand or a robotic gripper to
    open or close an object.
    """

    def surface_grasp_statement(
        self,
        require_lifting: bool = False,
        lowest_height: float = 0.1,
        highest_height: float = 0.9,
        shallowest_depth: float = 0.0,
        deepest_depth: float = 0.5,
    ) -> Match[SurfaceGrasp]:
        """
        :param require_lifting: Whether to ask only for grasps that lift the object.
        :param lowest_height: The lowest height to grasp the handle at, as a fraction
            of its bounding box from its bottom to its top.
        :param highest_height: The height to grasp the handle below, as such a
            fraction.
        :param shallowest_depth: The least depth to close the fingers at, as a fraction
            of the handle's thinnest extent.
        :param deepest_depth: The depth to close the fingers short of, as such a
            fraction.
        :return: The handle between the two heights, closing on it between the two
            depths.
        """
        thinnest_extent = float(min(self.grasp_surface().extents))
        grasp = any_surface_grasp(require_lifting)
        return grasp.where(
            grasp.height >= lowest_height,
            grasp.height < highest_height,
            grasp.depth >= shallowest_depth * thinnest_extent,
            grasp.depth < deepest_depth * thinnest_extent,
            *from_any_side(grasp),
        )

    @classmethod
    def _create_handle_geometry(
        cls, scale: Scale, thickness: float = 0.0
    ) -> SimpleEvent:
        """
        Create a box event representing the handle.

        :param scale: The scale of the handle.
        :param thickness: The thickness of the handle walls.
        """
        x_interval = closed(-scale.x + thickness, 0)
        y_interval = closed(-scale.y / 2, scale.y / 2)
        z_interval = closed(
            -scale.z / 2 + thickness,
            scale.z / 2 - thickness,
        )

        return SimpleEvent.from_data(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
        *,
        thickness: float = 0.005,
    ) -> BodySpecification:
        """
        Build the default handle body specification, matching the hollow handle geometry
        generated by :meth:`create_with_new_body_in_world`.

        :param name: The name of bodies created from the specification.
        :param scale: The scale of the handle: depth as x, width as y and length as z.
            Defaults to ``Scale(0.05, 0.02, 0.1)``.
        :param thickness: The thickness of the handle walls.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the handle geometry.
        :raises HandleWiderThanLong: If the width of ``scale`` exceeds its length.
        """
        scale = scale or Scale(0.05, 0.02, 0.1)
        if scale.y > scale.z:
            raise HandleWiderThanLong(scale=scale)
        handle_event = cls._create_handle_geometry(scale=scale).as_composite_set()
        inner_box = cls._create_handle_geometry(
            scale=scale, thickness=thickness
        ).as_composite_set()
        handle_event -= inner_box

        return BodySpecification.from_event(
            name, handle_event, connection_specification=connection_specification
        )

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        The grasps at the centre of the handle's bar that close around it, one
        approaching from each of the four sides around it.

        The bar runs along the longest extent of the root's collision geometry.

        :raises NoGraspGeometry: If the root body has no collision geometry to tell
            which way the bar runs.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        mesh = self.root.combined_mesh
        bar_axis = np.argmax(mesh.extents)
        bar_direction = Vector3.from_iterable(np.eye(3)[bar_axis])
        across_bar = np.delete(np.eye(3), bar_axis, axis=0)
        return [
            GraspCandidate(
                self,
                Pose(
                    position=Point3.from_iterable(mesh.bounds.mean(axis=0)),
                    orientation=RotationMatrix.from_vectors(
                        x=approach, y=bar_direction.cross(approach)
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
            for approach in map(Vector3.from_iterable, [*across_bar, *-across_bar])
        ]


@dataclass(eq=False)
class Aperture(HasRootRegion):
    """
    An opening in a physical entity.

    An example is like a hole in a wall that can be used to enter a room.
    """

    def _mount_strategy(
        self,
        main_has_root_body_annotation: HasRootBody,
        relationship: IsPartWholeRelationship,
    ) -> None:
        """
        Cut this aperture out of the whole's geometry when the relationship calls for
        it, then mount as a kinematic child.

        An aperture mounted into a field that does not remove geometry marks a passage
        rather than a hole, so the whole keeps its material.
        """
        if relationship.removes_part_geometry_from_whole:
            self._remove_aperture_geometry_from_parent(main_has_root_body_annotation)
        super()._mount_strategy(main_has_root_body_annotation, relationship)

    def _remove_aperture_geometry_from_parent(self, parent: HasRootBody):
        """
        Remove the geometry of the aperture from the parent body's collision and visual
        geometry.

        :param parent: The parent from which the aperture geometry is removed.
        """
        world = parent._world
        world.update_forward_kinematics()
        hole_event = self.root.area.as_bounding_box_collection_in_frame(
            parent.root
        ).event
        wall_event = parent.root.collision.as_bounding_box_collection_in_frame(
            parent.root
        ).event
        new_wall_event = wall_event - hole_event
        new_bounding_box_collection = BoundingBoxCollection.from_event(
            VolumetricBoundingBox, parent.root, new_wall_event
        ).as_shapes()

        parent.root.collision = new_bounding_box_collection
        parent.root.visual = new_bounding_box_collection

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> RegionSpecification:
        """
        Build the default region specification whose ``area`` geometry matches what
        :meth:`create_with_new_region_in_world` generates from ``scale``.

        :param name: The name of regions created from the specification.
        :param scale: The scale used to generate the region area geometry. Defaults to
            ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A region specification with a single box area derived from ``scale``.
        """
        scale = scale or Scale()
        return RegionSpecification.from_event(
            name,
            scale.to_simple_event().as_composite_set(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class EntryWay(Aperture):
    """
    The passage a door covers.

    It is coextensive with the door leaf, so mounting it on its door leaves the door's
    geometry intact; mounting it on a :class:`Wall` cuts the doorway out of the wall.
    """

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
        *,
        color: Color | None = None,
    ) -> RegionSpecification:
        """
        Build the default entry way region specification: a box coextensive with the
        door leaf, translucent so the passage is visible without occluding the door.

        :param name: The name of regions created from the specification.
        :param scale: The scale of the passage. Defaults to ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :param color: The colour of the passage. Defaults to translucent white.
        :return: A region specification with the passage geometry.
        """
        return RegionSpecification.box(
            name,
            scale or Scale(),
            color=color or Color.from_rgba(red=1.0, green=1.0, blue=1.0, opacity=0.2),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class Door(
    HasHandles,
    HasMovableJoint[ActiveConnection1DOF],
    HasControls,
    HasBackplates,
    HasKnobs,
    HasLocks,
    HasWindows,
):
    """
    A door is a physical entity that has covers an opening, has a movable body and a
    handle.
    """

    entry_way: EntryWay | None = field(
        default=None, metadata=IsPartWholeRelationship().as_dict()
    )
    """
    The entry way of the door: the passage the door covers.

    It is coextensive with the door leaf, so mounting it leaves the door's geometry
    intact; mounting the same entry way on a :class:`Wall` cuts the doorway out of it.
    """

    @classproperty
    def default_scale(cls) -> Scale:
        """
        The scale the door's geometry and its entry way are built from when the caller
        supplies none.
        """
        return Scale(0.03, 1, 2)

    @classmethod
    def get_annotation_specification(
        cls,
        name: str,
        root_specification: KinematicStructureEntitySpecification,
        *,
        parent_connection_specification: ConnectionSpecification | None = None,
        annotation_kwargs: dict | None = None,
        part_specifications: dict | None = None,
    ) -> SemanticAnnotationWithRootSpecification[Self]:
        """
        Extend the door specification with its entry way, declared as a nested part so a
        door always carries the passage it covers.

        The entry way is coextensive with the door leaf, so it is sized from the root
        specification's own geometry rather than from a separately supplied scale. A
        caller-supplied ``entry_way`` entry replaces the default one.

        :param name: The name of the annotation and its root entity.
        :param root_specification: The specification of the door's root body.
        :param parent_connection_specification: Connection attaching the root to its
            parent.
        :param annotation_kwargs: Inert keyword arguments for the annotation
            constructor.
        :param part_specifications: Nested annotation parts keyed by part-whole
            relationship field name.
        :return: The annotation specification.
        """
        entry_way_specification = EntryWay.get_annotation_specification(
            f"{name}_entry_way",
            EntryWay.get_default_root_kinematic_structure_entity_specification(
                scale=root_specification.scale
            ),
        )
        return super().get_annotation_specification(
            name,
            root_specification,
            parent_connection_specification=parent_connection_specification,
            annotation_kwargs=annotation_kwargs,
            part_specifications={
                "entry_way": entry_way_specification,
                **(part_specifications if part_specifications is not None else {}),
            },
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default door body specification.

        The door geometry is the base solid box; the ``EntryWay`` is a nested part
        specification and is not part of the door's own geometry.

        :param name: The name of bodies created from the specification.
        :param scale: The scale of the door. ``scale.x`` must be the smallest dimension.
            Defaults to :attr:`default_scale`.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the door box geometry.
        """
        scale = scale or cls.default_scale
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale, clazz=Door)
        return super().get_default_root_kinematic_structure_entity_specification(
            name, scale, connection_specification
        )

    def calculate_self_T_movable_joint(
        self, axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        Calculate where the door's hinge sits: on the edge opposite its handle.

        :param axis: The axis the door swings about, in the door frame, along one of the
            frame's axes in either direction.
        :return: The pose of the hinge in the door frame.
        """
        if not self.handles:
            raise MissingSemanticAnnotationError(self.__class__, Handle)

        connection = self.handles[0].root.parent_connection
        door_P_handle = connection.origin_expression.position
        scale = self.root.collision.scale

        # The axis' sign only decides which way the door swings, not which edge it
        # swings about.
        match [abs(component) for component in axis.to_np().tolist()]:
            case [0, 1, 0, 0]:
                sign = (
                    symbolic_math.sign(-1 * door_P_handle.z)
                    if door_P_handle.z != 0
                    else 1
                )
                offset = sign * (scale.z / 2)
                door_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(z=offset)

            case [0, 0, 1, 0]:
                sign = (
                    symbolic_math.sign(-1 * door_P_handle.y)
                    if door_P_handle.y != 0
                    else 1
                )
                offset = sign * (scale.y / 2)
                door_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(y=offset)
            case [1, 0, 0, 0]:
                sign = (
                    symbolic_math.sign(-1 * door_P_handle.x)
                    if door_P_handle.x != 0
                    else 1
                )
                offset = sign * (scale.x / 2)
                door_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(x=offset)

            case _:
                raise InvalidHingeActiveAxis(axis=axis)

        return door_T_hinge


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation):
    """
    A semantic annotation that represents a double door with left and right doors.
    """

    door_0: Door = field(kw_only=True)
    door_1: Door = field(kw_only=True)

    def calculate_left_right_door_from_view_point(
        self, world_T_view_point: HomogeneousTransformationMatrix
    ) -> tuple[Door, Door]:
        """
        Calculate which door is the left and which is the right door based on a given
        view point.

        :param world_T_view_point: The transformation matrix of the view point.
        :return: A tuple containing the left and right door. the first door is the left
            door, the second door is the right door.
        """
        world_T_door_0 = self.door_0.root.global_transform
        view_point_T_door_0 = world_T_view_point.inverse() @ world_T_door_0
        world_T_door_1 = self.door_1.root.global_transform
        view_point_T_door_1 = world_T_view_point.inverse() @ world_T_door_1
        if view_point_T_door_0.y > view_point_T_door_1.y:
            return self.door_0, self.door_1
        else:
            return self.door_1, self.door_0


@dataclass(eq=False)
class Drawer(
    Furniture,
    HasCaseAsRootBody,
    HasHandles,
    HasMovableJoint[ActiveConnection1DOF],
    HasKnobs,
):
    """
    A box that slides out of a piece of furniture, opened by a handle on its front.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.Z()

    @property
    def opening_ratio(self) -> float:
        """
        :return: How far this drawer stands pulled out, as a fraction of its travel.
        """
        if self.movable_joint is None:
            raise MissingMovableJointError(self)
        limits = self.movable_joint.dof.limits
        return (self.movable_joint.position - limits.lower.position) / (
            limits.upper.position - limits.lower.position
        )


@dataclass(eq=False)
class Elevator(HasCaseAsRootBody, HasDoors, HasMovableJoint[PrismaticConnection]):
    """
    An elevator in the world, consists of three walls a floor, double doors for entering and a prismatic drive that moves
    the elevator to other floors.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()

    def open(self):
        """
        Opens the elevator doors
        """
        for door in self.doors:
            if door.movable_joint is None:
                raise MissingMovableJointError(door)
            door.movable_joint.position = door.movable_joint.dof.limits.upper.position

    def close(self):
        """
        Closes the elevator doors
        """
        for door in self.doors:
            if door.movable_joint is None:
                raise MissingMovableJointError(door)
            door.movable_joint.position = door.movable_joint.dof.limits.lower.position

    def drive_position_for_floor(self, floor: Level) -> float:
        """
        The drive position at which the elevator serves the given floor.

        The half height accounts for the case body's origin sitting at its ground rather
        than at its centre.

        :param floor: The floor the elevator should serve.
        :return: The position to drive the elevator's joint to.
        """
        return float(floor.floor_plane[0].z)

    def drive_to_floor(self, floor: Level):
        """
        Drives the elevator to the floor given
        """
        if self.movable_joint is None:
            raise MissingMovableJointError(self)
        self.movable_joint.position = self.drive_position_for_floor(floor)


############################### subclasses to Furniture


@dataclass(eq=False)
class ShelfLayer(HasSupportingSurface):
    """
    A horizontal surface used for storing objects, typically found inside cabinets or on
    walls.
    """

    _synonyms = {"level", "board"}


@dataclass(eq=False)
class Table(Furniture, HasSupportingSurface):
    """
    A semantic annotation that represents a table.
    """


@dataclass(eq=False)
class CounterTop(Furniture, HasSupportingSurface, HasSink, HasCooktops):
    """
    A semantic annotation that represents a counter top.
    """

    _synonyms = {"countertop"}


@dataclass(eq=False)
class KitchenIsland(
    Furniture, HasRootBody, HasCabinets, HasCounterTop, HasDoors, HasDrawers
):
    """
    A run of kitchen cabinets standing free of the walls, under one counter top.
    """

    _synonyms = {"kitchen_island", "island"}


@dataclass(eq=False)
class Cabinet(
    Furniture,
    HasCaseAsRootBody,
    HasDoors,
    HasDrawers,
    HasBackplates,
    HasHandles,
    HasHinges,
    HasLids,
    HasSinks,
    HasWheels,
):
    """
    A piece of furniture with a case that things are stored inside, closed by doors or drawers.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Dishwasher(Cabinet):
    """
    A dishwasher is a kitchen appliance used for cleaning dishes, utensils, and
    cookware.

    It typically has a front door that opens to reveal racks for loading dirty items and
    a control panel for selecting wash cycles.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Fridge(Cabinet):
    """
    A cabinet that keeps what is put in it cold.
    """


@dataclass(eq=False)
class WashingMachine(Cabinet, HasControls, HasHandles):
    """
    A washing machine is a household appliance that washes laundry in a drum.

    It typically has a front door to load the drum and a pull-out detergent drawer.
    """


@dataclass(eq=False)
class TumbleDryer(Cabinet):
    """
    A tumble dryer is a household appliance that dries laundry in a rotating drum.

    It typically has a front door to load the drum and a pull-out water tank.
    """


@dataclass(eq=False)
class Oven(HasRootBody, HasDoors, HasControls, HasCooktops):
    """
    An appliance with a door that cooks what is put inside it with heat.
    """


@dataclass(eq=False)
class Dresser(Cabinet):
    """
    A cabinet of drawers, for clothes.
    """


@dataclass(eq=False)
class Cupboard(Cabinet):
    """
    A cabinet standing free of any run, whose whole inside is for storing things.
    """


@dataclass(eq=False)
class Wardrobe(Cabinet, HasHinges):
    """
    A tall cabinet for hanging clothes in.
    """


@dataclass(eq=False)
class Floor(HasSupportingSurface):
    """
    The surface of a room that is walked on and that everything in the room stands on.
    """

    @classmethod
    def create_with_new_body_from_polytope_in_world(
        cls,
        name: str,
        world: World,
        floor_polytope: list[Point3],
        world_root_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given list of
        Point3.

        :param name: The name of the floor body.
        :param floor_polytope: A list of 3D points defining the floor poly
        """
        return cls.get_annotation_specification(
            name,
            BodySpecification.from_3d_points(name, floor_polytope),
        ).spawn(world, parent_T_self=world_root_T_self)

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default floor body specification, matching the convex-hull polytope
        geometry generated by :meth:`create_with_new_body_in_world` (which derives
        points from the scale bounding box).

        :param name: The name of bodies created from the specification.
        :param scale: The scale defining the floor polytope. Defaults to ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the floor polytope mesh.
        """
        scale = scale or Scale()
        return BodySpecification.from_3d_points(
            name,
            scale.to_bounding_box().get_points(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class Room(SemanticAnnotation):
    """
    A closed area with a specific purpose.
    """

    floor: Floor = field(kw_only=True)
    """
    The room's floor.
    """


@dataclass(eq=False)
class Kitchen(Room):
    """
    A room for preparing food.
    """


@dataclass(eq=False)
class Bedroom(Room):
    """
    A room for sleeping, with a bed in it.
    """


@dataclass(eq=False)
class Bathroom(Room):
    """
    A room for washing, with a bath, a shower or a toilet in it.
    """


@dataclass(eq=False)
class LivingRoom(Room):
    """
    A room for sitting in, with its seating turned towards a hearth, a television or itself.
    """


@dataclass(eq=False)
class Level(HasRootRegion):
    """
    A level of a building
    """

    @property
    def floor_plane(self) -> tuple[Point3, Point3]:
        """
        The floor plane of a level, expressed as two points

        :return: The floor plane of the level
        """
        min_point_global = self._world.transform(
            self.root.area.min_point, self._world.root
        )
        max_point_global = self._world.transform(
            self.root.area.max_point, self._world.root
        )
        min_z = min_point_global.z
        return min_point_global, Point3(
            max_point_global.x,
            max_point_global.y,
            min_z,
            reference_frame=self._world.root,
        )


@dataclass(eq=False)
class GroundFloor(Level):
    """
    The level of a building that is entered from outside.
    """


@dataclass(eq=False)
class FirstFloor(Level):
    """
    The level of a building above the one entered from outside.
    """


@dataclass(eq=False)
class SecondFloor(Level):
    """
    The level of a building two above the one entered from outside.
    """


@dataclass(eq=False)
class Wall(HasApertures):
    """
    A wall is a physical entity that separates two spaces and can contain apertures.

    Doors are a computed property.
    """

    _synonyms = {"walls"}

    @property
    def doors(self) -> Iterable[Door]:
        return [
            door
            for door in self._world.get_semantic_annotations_by_type(Door)
            if door.entry_way
            and InsideOf(
                door.entry_way.root, self.root, minimum_containment_ratio=0.1
            )()
        ]

    @classmethod
    def _create_wall_event(cls, scale: Scale) -> SimpleEvent:
        """
        Return the collision shapes for the wall.

        A wall event is created based on the scale of the wall, and doors are removed
        from the wall event. The resulting bounding box collection is converted to
        shapes.
        """
        x_interval = closed(-scale.x / 2, scale.x / 2)
        y_interval = closed(-scale.y / 2, scale.y / 2)
        z_interval = closed(0, scale.z)

        return SimpleEvent.from_data(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default wall body specification, matching the geometry generated by
        :meth:`create_with_new_body_in_world` (a box whose z runs ``0..scale.z``).

        :param name: The name of bodies created from the specification.
        :param scale: The scale of the wall. ``scale.x`` must be the smallest dimension.
            Defaults to ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the wall box geometry.
        """
        scale = scale or Scale()
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale, clazz=Wall)
        return BodySpecification.from_event(
            name,
            cls._create_wall_event(scale).as_composite_set(),
            connection_specification=connection_specification,
        )

    def bloated_bounding_box_collection(
        self,
        origin: HomogeneousTransformationMatrix,
        bloat_amount: float,
        obstacle_height_clearance: float = 0.01,
    ) -> BoundingBoxCollection[VolumetricBoundingBox, Point3]:
        """
        Bloat this wall's bounding boxes along their thinner dimension only -- the
        side that faces the room -- rather than symmetrically in x and y.

        :param origin: The origin to express the bounding boxes relative to.
        :param bloat_amount: The amount to bloat by.
        :param obstacle_height_clearance: The amount to bloat by in z, regardless of
            ``bloat_amount``.
        :return: The bloated bounding boxes.
        """
        return BoundingBoxCollection(
            [
                (
                    bounding_box.bloat(bloat_amount, 0, obstacle_height_clearance)
                    if bounding_box.width > bounding_box.depth
                    else bounding_box.bloat(0, bloat_amount, obstacle_height_clearance)
                )
                for bounding_box in self.as_bounding_box_collection_at_origin(origin)
            ],
            origin.reference_frame,
        )


@dataclass(eq=False)
class Container(HasGraspCandidates, HasLids):
    """
    An object that holds contents and has an opening they go in and out through.
    """


@dataclass(eq=False)
class Cookware(HasGraspCandidates):
    """
    An object used to cook with.
    """


@dataclass(eq=False)
class Tableware(HasGraspCandidates):
    """
    An object a table is set with for a meal.
    """


@dataclass(eq=False)
class Bottle(Container, HasCaps, HasHandles):
    """
    Abstract class for bottles.
    """


@dataclass(eq=False)
class Statue(HasGraspCandidates):
    """
    A carved or cast figure of something, kept for the look of it.
    """


@dataclass(eq=False)
class SoapBottle(Bottle):
    """
    A soap bottle.
    """


@dataclass(eq=False)
class WineBottle(Bottle):
    """
    A wine bottle.
    """


@dataclass(eq=False)
class MustardBottle(Bottle):
    """
    A mustard bottle.
    """


@dataclass(eq=False)
class DrinkingContainer(Container, Tableware):
    """
    A container that is drunk from, such as a cup or a glass.
    """


@dataclass(eq=False)
class Cup(HasHandles, HasRim, DrinkingContainer, IsPerceivable):
    """
    A cup.
    """


@dataclass(eq=False)
class Mug(HasHandles, HasRim, DrinkingContainer):
    """
    A mug, held by its handle.
    """


@dataclass(eq=False)
class CookingContainer(Container, Cookware):
    """
    A container that food is cooked in, such as a pot or a pan.
    """


@dataclass(eq=False)
class Lid(Cookware, HasMovableJoint[ActiveConnection1DOF], HasHandles):
    """
    A cover that closes a container.
    """


@dataclass(eq=False)
class Pan(CookingContainer):
    """
    A pan.
    """


@dataclass(eq=False)
class PanLid(Lid):
    """
    A pan lid.
    """


@dataclass(eq=False)
class Pot(CookingContainer, HasHandles):
    """
    A pot.
    """


@dataclass(eq=False)
class PotLid(Lid):
    """
    A pot lid.
    """


@dataclass(eq=False)
class Plate(HasSupportingSurface, Tableware):
    """
    A plate.
    """


@dataclass(eq=False)
class Rim(HasPinchableWall):
    """
    The upper edge of an open container's wall, which a gripper pinches from above.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: Grasps drawn uniformly from :meth:`surface_grasp_statement`.
        """
        return self.drawn_grasp_candidates()

    def surface_grasp_statement(
        self,
        require_lifting: bool = False,
        lowest_height: float = 0.25,
        highest_height: float = 0.75,
    ) -> Match[SurfaceGrasp]:
        """
        :param require_lifting: Whether to ask only for grasps that lift the container.
        :param lowest_height: The lowest height to pinch the rim at, as a fraction of
            its height.
        :param highest_height: The height to pinch the rim below, as such a fraction.
        :return: The rim between the two heights, pinched across its wall.
        :raises NoGraspGeometry: If the rim has no wall to measure.
        """
        return self.wall_grasp_statement(lowest_height, highest_height, require_lifting)


@dataclass(eq=False)
class Bowl(HasRim, HasSupportingSurface, Container, Tableware, IsPerceivable):
    """
    A bowl, grasped by pinching the wall of its rim: it offers nothing to grip at its
    own origin, which is inside it.
    """


# Food Items
@dataclass(eq=False)
class Food(HasGraspCandidates):
    """
    Something that is eaten.
    """


@dataclass(eq=False)
class TunaCan(Food):
    """
    A tuna can.
    """


@dataclass(eq=False)
class Bread(Food):
    """
    Bread.
    """

    _synonyms = {
        "bumpybread",
        "whitebread",
        "loafbread",
        "honeybread",
        "grainbread",
    }


@dataclass(eq=False)
class CheezeIt(Food, IsPerceivable):
    """
    Some type of cracker.
    """


@dataclass(eq=False)
class Pringles(Food):
    """
    Pringles chips.
    """


@dataclass(eq=False)
class GelatinBox(Food):
    """
    Gelatin box.
    """


@dataclass(eq=False)
class TomatoSoup(Food):
    """
    Tomato soup.
    """


@dataclass(eq=False)
class Candy(Food, IsPerceivable):
    """
    A candy.
    """

    ...


@dataclass(eq=False)
class Noodles(Food, IsPerceivable):
    """
    A container of noodles.
    """

    ...


@dataclass(eq=False)
class Cereal(Food, IsPerceivable):
    """
    A container of cereal.
    """

    ...


@dataclass(eq=False)
class Milk(Food, IsPerceivable):
    """
    A container of milk.
    """


@dataclass(eq=False)
class SaltContainer(Container, IsPerceivable):
    """
    A container of salt.
    """


@dataclass(eq=False)
class Produce(Food):
    """
    In American English, produce generally refers to fresh fruits and vegetables
    intended to be eaten by humans.
    """

    pass


@dataclass(eq=False)
class Fruit(Produce):
    """
    Fruit.
    """


@dataclass(eq=False)
class Vegetable(Produce):
    """
    Vegetable.
    """


@dataclass(eq=False)
class Tomato(Fruit):
    """
    A tomato.
    """


@dataclass(eq=False)
class Lettuce(Vegetable):
    """
    Lettuce.
    """


@dataclass(eq=False)
class Carrot(Vegetable):
    """
    A carrot.
    """


@dataclass(eq=False)
class Apple(Fruit):
    """
    An apple.
    """


@dataclass(eq=False)
class Banana(Fruit):
    """
    A banana.
    """


@dataclass(eq=False)
class Orange(Fruit):
    """
    An orange.
    """


@dataclass(eq=False)
class Salt(Food):
    """
    A pack or container of salt (e.g., salt shaker or salt can).
    """


@dataclass(eq=False)
class CoffeeTable(Table):
    """
    A coffee table.
    """


@dataclass(eq=False)
class DiningTable(Table, HasLegs):
    """
    A dining table.
    """


@dataclass(eq=False)
class SideTable(Table, HasDoors):
    """
    A side table.
    """

    _synonyms = {"bedside"}


@dataclass(eq=False)
class Desk(Table, HasLegs, HasCabinets, HasDoors, HasDrawers):
    """
    A desk.
    """


@dataclass(eq=False)
class Chair(Furniture, HasRootBody):
    """
    A seat for one person, with a back.

    Not abstract, whatever an earlier version of this docstring said: a chair is a thing
    a room has and an object can be annotated as one. A stool has been answered with
    this class and is not one, having no back to it.
    """


@dataclass(eq=False)
class OfficeChair(Chair):
    """
    An office chair.
    """


@dataclass(eq=False)
class Armchair(Chair):
    """
    An armchair.
    """


@dataclass(eq=False)
class TrashCan(HasCaseAsRootBody, Furniture, HasControls, HasLids, HasLidFrames):
    """
    Abstract class for Trash Can.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.Z()


@dataclass(eq=False)
class Shelf(Cabinet, HasShelfLayers):
    """
    A shelving unit whose storage surfaces are modelled as explicit shelf layers.

    Each layer is a separate :class:`ShelfLayer` annotation, so objects can be placed on
    and reasoned about per layer.
    """


@dataclass(eq=False)
class Bed(Furniture):
    """
    A bed.
    """


@dataclass(eq=False)
class Sofa(Furniture, HasSupportingSurface):
    """
    A sofa.
    """


@dataclass(eq=False)
class Sink(HasRootBody, HasDrainPipes, HasFaucets):
    """
    A sink.
    """


@dataclass(eq=False)
class Kettle(CookingContainer, HasControls, HasHandles, HasLids):
    """
    A covered container for boiling water.
    """


@dataclass(eq=False)
class Decor(HasRootBody, ABC):
    """
    Decoration in general, a category to answer with the *kind* of rather than by name: a
    picture, a mirror, an ornament, or a new subclass of this where the ontology has none.
    """


@dataclass(eq=False)
class WallDecor(Decor):
    """
    Wall decorations.
    """


@dataclass(eq=False)
class Lamp(HasRootBody):
    """
    Something that lights a room.

    Not :class:`Decor`, though it was the only word for one until now: a lamp is there
    to do a job, and reading it as ornament loses that. Where a scene distinguishes the
    kinds -- standing on a table, hung from a ceiling, fixed to a wall -- a run composes
    that distinction against this class rather than the ontology guessing at it in
    advance.
    """


@dataclass(eq=False)
class ElectricalDevice(HasRootBody, ABC):
    """
    Electrically powered devices in general, a category to answer with the *kind* of
    rather than by name: name the device, composing a new subclass of this where the
    ontology has none.

    A television, a smoke detector, a thermostat. The ontology had no word between them,
    and runs kept inventing one under a different name each time -- ``Device`` in one,
    ``ElectronicDevice`` in the next, and in a third ``Agent``, which is meant for
    something that can act and move. Naming it once is what makes those runs comparable.

    **Electricity has to run through the thing itself.** A fire extinguisher, a tap and
    an ice maker have all been answered with this class and none of them belongs to it:
    a fire extinguisher holds pressure, a tap holds water, and an ice maker is a
    :class:`Fridge`. Something merely near electricity, plumbed in beside it, or
    resembling an appliance is not one of these.

    ..note:: A thinner case than :class:`Lamp`, and worth revisiting. Across the four
        annotated HM3D buildings it covers some forty-seven objects against that class's
        two hundred, and they have little in common beyond their power: what matters
        about a television is that it stands on something and about a light switch that
        it is on a wall. Where a scene draws a distinction worth keeping, a run composes
        it against this class.
    """


@dataclass(eq=False)
class Mirror(HasRootBody):
    """
    A surface that shows what is in front of it.

    Its own thing rather than a kind of :class:`WallDecor`: a mirror is used rather than
    looked at, and not every one of them hangs on a wall, since some stand on the floor.
    """


@dataclass(eq=False)
class Cloth(HasGraspCandidates):
    """
    A piece of fabric for wiping or covering, such as a towel.
    """


@dataclass(eq=False)
class Poster(WallDecor):
    """
    A poster.
    """


@dataclass(eq=False)
class WallPanel(HasRootBody, HasControls, HasDoors, HasLids):
    """
    A wall panel.
    """


@dataclass(eq=False)
class Potato(Vegetable):
    """
    A firm root vegetable, roughly round.
    """


@dataclass(eq=False)
class GarbageBin(HasRootBody):
    """
    A garbage bin.
    """


@dataclass(eq=False)
class Drone(HasRootBody): ...


@dataclass(eq=False)
class ProcthorBox(HasRootBody): ...


@dataclass(eq=False)
class Houseplant(HasRootBody):
    """
    A houseplant.
    """


@dataclass(eq=False)
class SprayBottle(Bottle):
    """
    A spray bottle.
    """


@dataclass(eq=False)
class Vase(Container):
    """
    A vase.
    """


@dataclass(eq=False)
class Book(HasGraspCandidates):
    """
    A book.
    """

    book_front: BookFront | None = None


@dataclass(eq=False)
class BookFront(HasRootBody): ...


@dataclass(eq=False)
class SaltPepperShaker(SaltContainer):
    """
    A salt and pepper shaker.
    """


@dataclass(eq=False)
class Cutlery(HasHandles, Tableware):
    """
    A piece of cutlery, lying flat, grasped from above and across its length.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: The grasp from above that closes across the piece, which lies flat.
        :raises NoGraspGeometry: If the root body has no collision geometry to tell
            which way the piece lies.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        bounding_box = self.root.collision.as_bounding_box_collection_in_frame(
            self.root
        ).bounding_box()
        along_x = bounding_box.x_interval.upper - bounding_box.x_interval.lower
        along_y = bounding_box.y_interval.upper - bounding_box.y_interval.lower
        finger_axis = Vector3.NEGATIVE_Y() if along_x >= along_y else Vector3.X()
        return [
            GraspCandidate(
                self,
                Pose(
                    orientation=RotationMatrix.from_vectors(
                        x=Vector3.NEGATIVE_Z(), y=finger_axis
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
        ]

    def surface_grasp_statement(
        self,
        require_lifting: bool = False,
        lowest_height: float = 0.2,
        highest_height: float = 0.8,
        shallowest_depth: float = 0.2,
        deepest_depth: float = 0.6,
        azimuth_tolerance: float = np.pi / 16,
        largest_pitch: float = 0.6,
        largest_roll: float = np.pi / 8,
    ) -> Match[SurfaceGrasp]:
        """
        :param require_lifting: Whether to ask only for grasps that lift the object.
        :param lowest_height: The lowest height to grasp at, as a fraction of the grasp
            surface's bounding box from its bottom to its top.
        :param highest_height: The height to grasp below, as such a fraction.
        :param shallowest_depth: The least depth to close the fingers at, as a fraction
            of the piece's length.
        :param deepest_depth: The depth to close the fingers short of, as such a
            fraction.
        :param azimuth_tolerance: How far off its length axis the piece may be seen
            from, either way, in radians.
        :param largest_pitch: How far the approach may tilt off vertical, in radians.
        :param largest_roll: How far the fingers may turn off closing across the
            length, either way, in radians.
        :return: Seen along its length from either end, the stretch between the two
            depths, closing across it.
        :raises NoGraspGeometry: If the root body has no shape to measure.
        """
        lowest, highest = self.grasp_surface().bounds
        extents = highest - lowest
        length_axis = max(
            (Vector3.X().to_np()[:3], Vector3.Y().to_np()[:3]),
            key=lambda axis: np.dot(extents, axis),
        )
        length = float(np.dot(extents, length_axis))
        azimuth_of_length_axis = math.atan2(length_axis[1], length_axis[0])
        end_azimuths = (azimuth_of_length_axis, azimuth_of_length_axis + np.pi)
        grasp = any_surface_grasp(require_lifting)
        return grasp.where(
            grasp.height >= lowest_height,
            grasp.height < highest_height,
            grasp.depth >= shallowest_depth * length,
            grasp.depth < deepest_depth * length,
            or_(
                *(
                    and_(
                        grasp.azimuth >= end_azimuth - azimuth_tolerance,
                        grasp.azimuth < end_azimuth + azimuth_tolerance,
                    )
                    for end_azimuth in end_azimuths
                )
            ),
            grasp.pitch >= 0.0,
            grasp.pitch < largest_pitch,
            grasp.roll >= np.pi / 2 - largest_roll,
            grasp.roll < np.pi / 2 + largest_roll,
        )


@dataclass(eq=False)
class Fork(Cutlery):
    """
    A fork.
    """


@dataclass(eq=False)
class Knife(Cutlery):
    """
    A butter knife.
    """


@dataclass(eq=False)
class Spoon(Cutlery, IsPerceivable):
    """
    A piece of cutlery with a shallow bowl, for what a fork will not hold.
    """


@dataclass(eq=False)
class Pencil(HasGraspCandidates):
    """
    A pencil.
    """


@dataclass(eq=False)
class Pen(HasGraspCandidates):
    """
    A pen.
    """


@dataclass(eq=False)
class ShapeSortingPiece(HasGraspCandidates):
    """
    A loose piece of a shape-sorting toy, which drops through the hole of its
    own outline.
    """

    grasp_height: float = field(kw_only=True)
    """
    How far above its bottom face the piece is gripped.

    Gripped low, the piece hangs as little as possible below the fingers when it is let
    go of over its hole; how low depends on how far the closing fingers reach past the
    point they are brought to.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: The grasp from above, on the piece's vertical axis at
            :attr:`grasp_height`, closing along the piece's own x.
        :raises NoGraspGeometry: If the root body has no collision geometry to find the
            bottom face in.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        bottom_face = (
            self.root.collision.as_bounding_box_collection_in_frame(self.root)
            .bounding_box()
            .min_z
        )
        return [
            GraspCandidate(
                self,
                Pose(
                    position=Point3(
                        0.0,
                        0.0,
                        bottom_face + self.grasp_height,
                        reference_frame=self.root,
                    ),
                    orientation=RotationMatrix.from_vectors(
                        x=Vector3.NEGATIVE_Z(), y=Vector3.X()
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
        ]


@dataclass(eq=False)
class Baseball(HasGraspCandidates):
    """
    A baseball.
    """


@dataclass(eq=False)
class Cap(HasMovableJoint[ActiveConnection1DOF], HasPumps):
    """
    A cap that closes an opening, such as a bottle's or a soap dispenser's, typically
    mounted on a screw joint.
    """


@dataclass(eq=False)
class Agent(HasRootBody):
    """
    Represents an entity in the world that can act, move, or be controlled.

    Agents are dynamic bodies with semantic meaning — they may have intent, behavior, or
    be controlled by external or internal logic. Examples include robots, humans, or
    other autonomous actors.
    """


@dataclass(eq=False)
class Human(Agent):
    """
    Represents a human agent in the environment.

    A Person is an Agent that is not robotically actuated and does not provide kinematic
    chains, end_effectors, or robot-specific components.

    This class exists primarily for semantic distinction, so that algorithms can treat
    human agents differently from robots if needed.
    """


@dataclass(eq=False)
class Parcel(HasGraspCandidates):
    """
    A parcel, as handled in a warehouse.
    """


@dataclass(eq=False)
class SemanticEnvironmentAnnotation(HasRootBody):
    """
    Represents a semantic annotation of the environment.
    """

    def obstacle_entities(
        self, search_space: BoundingBoxCollection[VolumetricBoundingBox, Point3]
    ) -> list[Body]:
        """
        Collect the obstacle bodies to consider within ``search_space``.

        Filters out robot bodies so a robot does not treat itself as an obstacle, and
        bodies without meaningful collision geometry.

        :param search_space: The search space; its reference frame is used to look up
            the owning world.
        :return: The obstacle bodies to consider.
        """
        world = search_space.reference_frame._world
        return [
            body
            for body in self.bodies_with_collision
            if body not in world.robot_bodies_with_collision
        ]

    def build_bloated_obstacle_collection(
        self,
        search_space: BoundingBoxCollection[VolumetricBoundingBox, Point3],
        semantic_wall_annotation: Wall | None = None,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
        obstacle_height_clearance: float = 0.01,
    ) -> BoundingBoxCollection[VolumetricBoundingBox, Point3]:
        """
        Collect and bloat this annotation's obstacle bounding boxes.

        Applies independent bloat amounts to obstacles and walls.

        :param search_space: The search space; its reference frame is used as the
            origin.
        :param semantic_wall_annotation: An optional wall annotation, bloated by its
            own rule (see :meth:`Wall.bloated_bounding_box_collection`).
        :param bloat_obstacles: Amount to expand each obstacle bounding box
            symmetrically in x and y.
        :param bloat_walls: Amount to expand wall bounding boxes in their thinner
            dimension.
        :param obstacle_height_clearance: Amount to expand every obstacle bounding box
            in z, regardless of ``bloat_obstacles``/``bloat_walls``.
        :return: A collection of the bloated obstacle and wall bounding boxes.
        """
        world_root = search_space.reference_frame
        origin = HomogeneousTransformationMatrix(reference_frame=world_root)

        entities_to_consider = self.obstacle_entities(search_space)

        collections = (
            entity.collision.as_bounding_box_collection_at_origin(origin)
            for entity in entities_to_consider
        )
        obstacle_bounding_boxes = BoundingBoxCollection.merge_all(
            collections, world_root
        )

        bloated_obstacles = BoundingBoxCollection(
            [
                bounding_box.bloat(
                    bloat_obstacles, bloat_obstacles, obstacle_height_clearance
                )
                for bounding_box in obstacle_bounding_boxes
            ],
            world_root,
        )

        if semantic_wall_annotation is not None:
            bloated_obstacles = bloated_obstacles.merge(
                semantic_wall_annotation.bloated_bounding_box_collection(
                    origin, bloat_walls, obstacle_height_clearance
                )
            )

        return bloated_obstacles


@dataclass(eq=False)
class RoomWithWallsAndDoors(Room):
    """
    A room with a type description (e.g., Ktichen) and walls and doors.
    """

    room_type: str | None = field(kw_only=True, default=None)
    """
    Description of the type of the room in natural language.
    """

    walls: list[Wall] = field(kw_only=True, default_factory=list)
    """
    The walls enclosing this room.
    """

    doors: list[Door] = field(kw_only=True, default_factory=list)
    """
    The doors of the room.
    """


@dataclass(eq=False)
class DoorWithType(Door):
    """
    A Door that has a type description, e.g. "main entrance".
    """

    type_description: str | None = field(kw_only=True, default=None)


@dataclass(eq=False)
class Leg(HasRootBody):
    """
    A leg that supports a piece of furniture.
    """


@dataclass(eq=False)
class Cooktop(HasRootBody, HasControls):
    """
    A cooktop surface for cooking.
    """


@dataclass(eq=False)
class Tool(HasGraspCandidates, ABC):
    """
    A tool that is held by a robot's end effector to act on other bodies.
    """

    def _end_effector_name(self) -> PrefixedName:
        """
        :return: The name of the body that acts as the tool's tip, derived from the
            tool's root name.
        """
        root_name = self.root.name
        return PrefixedName(f"{root_name.name}_end_effector", root_name.prefix)

    def _find_end_effector_body(self) -> Body | None:
        if self.root._world is None:
            return None

        end_effector_name = self._end_effector_name()
        for body in self.root._world.bodies:
            if (
                body.name == end_effector_name
                or body.name.name == end_effector_name.name
            ):
                return body
        return None

    def get_tool_frame(self) -> Body:
        """
        :return: The body acting as the tool's tip, or the tool's root if no dedicated
            tip body exists.
        """
        end_effector = self._find_end_effector_body()
        if end_effector is None:
            return self.root
        return end_effector

    @abstractmethod
    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        """
        :param target: The body or pose the tool acts on.
        :return: The normal pairs that must stay aligned while the tool acts on the
            target.
        """


@dataclass(eq=False)
class ToolWithHandle(Tool, HasHandles, ABC):
    """
    A tool held by its handle, acting through a tip body located ahead of its root.

    If the tip body does not exist yet, it is created and rigidly connected to the
    tool's root.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.root._world is None:
            return
        if self._find_end_effector_body() is not None:
            return
        self._create_end_effector_body()

    def _create_end_effector_body(self):
        world = self.root._world
        tip_body = Body(name=self._end_effector_name())
        with world.modify_world():
            world.add_kinematic_structure_entity(tip_body)
            world.add_connection(
                FixedConnection(
                    parent=self.root,
                    child=tip_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.1,
                        y=0.0,
                        z=0.0,
                        reference_frame=self.root,
                    ),
                )
            )


@dataclass(eq=False)
class Whisk(ToolWithHandle):
    """
    A whisk for mixing the contents of containers.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(-1, 0, 0, reference_frame=self.root),
                goal_normal=Vector3(0, 0, 1, reference_frame=target),
            )
        ]


@dataclass(eq=False)
class CuttingKnife(ToolWithHandle):
    """
    A knife for cutting food objects.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(1, 0, 0, reference_frame=self.root),
                goal_normal=Vector3(0, 1, 0, reference_frame=target),
            ),
            AlignmentPair(
                tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                goal_normal=Vector3(0, 0, 1, reference_frame=target),
            ),
        ]


@dataclass(eq=False)
class PouringCup(Tool, Container):
    """
    A cup for pouring liquids into containers.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                goal_normal=Vector3(0, 0, 1, reference_frame=target),
            )
        ]


@dataclass(eq=False)
class Sponge(Tool):
    """
    A sponge for wiping surfaces.

    .. note:: The sponge is grasped so its local Z axis points away from the gripper,
        which is why its alignments use negative Z normals unlike the other tools.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        if isinstance(target, Body):
            return [
                AlignmentPair(
                    tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                    goal_normal=Vector3(0, 0, -1, reference_frame=target),
                )
            ]
        return [
            AlignmentPair(
                tip_normal=Vector3(0, 0, -1, reference_frame=self.root),
                goal_normal=self._pose_surface_normal(target),
            )
        ]

    def _pose_surface_normal(self, pose: Pose) -> Vector3:
        """
        :param pose: The pose whose surface normal is computed.
        :return: The pose's local Z axis expressed in the pose's reference frame.
        """
        reference_frame = (
            pose.reference_frame if pose.reference_frame is not None else self.root
        )
        rotation = pose.rotation_matrix.to_np()[:3, :3]
        return Vector3.from_iterable(
            rotation @ np.array([0.0, 0.0, 1.0]),
            reference_frame=reference_frame,
        )


@dataclass(eq=False)
class Microwave(IsStorageSpace, HasDoors, HasControls):
    """
    A microwave oven, a kitchen appliance with a door that heats food placed inside it
    using microwave radiation.
    """


@dataclass(eq=False)
class Hood(HasRootBody):
    """
    A range hood mounted above a cooktop that vents cooking fumes.
    """


@dataclass(eq=False)
class Toaster(HasRootBody, HasControls):
    """
    A countertop appliance for toasting slices of bread.
    """


@dataclass(eq=False)
class CoffeeMachine(HasRootBody, HasControls):
    """
    A countertop appliance that brews coffee.
    """

    _synonyms = {"coffe"}


# %% openings and their frames


@dataclass(eq=False)
class Frame(HasRootBody, HasDoors, HasHinges, HasKnobs, HasLocks, HasWindows):
    """
    The fixed frame around an opening in a wall that a door or a window hangs in.
    """


@dataclass(eq=False)
class Window(
    HasMovableJoint[RevoluteConnection],
    HasBlinds,
    HasHandles,
    HasBackplates,
    HasKnobs,
):
    """
    A pane of glass in a frame that closes an opening in a wall and opens to let in air.
    """


@dataclass(eq=False)
class Hinge(HasRootBody):
    """
    A joint of two plates that a door, window or lid swings on.

    It is a part of the fixed object that what swings hangs in, such as a frame or a
    cabinet, and never of what swings.
    """


# %% window coverings


@dataclass(eq=False)
class Rail(HasRootBody, HasBlinds, HasControls, HasCurtains):
    """
    A rail that blinds or a curtain hang from and slide along.
    """


@dataclass(eq=False)
class Blinds(HasMovableJoint[PrismaticConnection], HasControls):
    """
    Slats over a window that are raised and lowered to let in more or less light.
    """


@dataclass(eq=False)
class Curtain(HasMovableJoint[PrismaticConnection]):
    """
    A length of fabric hanging over a window that is drawn open and closed.
    """


# %% controls


@dataclass(eq=False)
class Control(HasMovableJoint[ActiveConnection1DOF]):
    """
    Something operated by hand or foot to control a device, such as a knob, a switch or
    a button.
    """


@dataclass(eq=False)
class Switch(Control):
    """
    A control that is flipped between on and off, such as a light switch.
    """


@dataclass(eq=False)
class Button(Control):
    """
    A control that is pressed.
    """


@dataclass(eq=False)
class Pedal(Control):
    """
    A control that is pressed with the foot, such as the one that opens a trash can's
    lid.
    """


@dataclass(eq=False)
class ControlPanel(HasRootBody, HasControls, HasDoors):
    """
    A panel of a device that carries its controls.
    """


# %% bases and plates that other parts are mounted on


@dataclass(eq=False)
class Backplate(
    HasHandles,
    HasLocks,
    HasKnobs,
    HasControls,
    HasPowerSockets,
    HasThermostats,
    HasToiletFlushes,
    HasLids,
):
    """
    The plate a fitting is mounted on, such as the rose of a door handle or the plate of a light switch, a power socket or a toilet flush.
    """


@dataclass(eq=False)
class BaseStation(HasRootBody, HasControls, HasCables):
    """
    The unit of a telephone or an intercom that its handset rests on.
    """


@dataclass(eq=False)
class FaucetBody(HasRootBody, HasControls, HasFaucets):
    """
    The body of a faucet that its spout and controls are mounted on.
    """


@dataclass(eq=False)
class LidFrame(HasRootBody, HasLids):
    """
    The top of a container that holds its lid, such as the head of a trash can.
    """


# %% parts that are grasped or lock


@dataclass(eq=False)
class Knob(HasGraspCandidates, HasMovableJoint[RevoluteConnection]):
    """
    A small round grip that a door, drawer or device is pulled or turned by.
    """


@dataclass(eq=False)
class Lock(HasMovableJoint[RevoluteConnection]):
    """
    A mechanism that keeps a door or window shut, such as a thumb turn or a cylinder.
    """


@dataclass(eq=False)
class KeyCard(HasGraspCandidates):
    """
    A card that unlocks a door or switches on a room's power.
    """


# %% heating


@dataclass(eq=False)
class Heater(HasRootBody, HasControls, HasHeaterValves):
    """
    Something that warms a room, such as a radiator.
    """


@dataclass(eq=False)
class HeaterValve(HasMovableJoint[RevoluteConnection]):
    """
    The valve on a heater that is turned to set how much it warms.
    """


@dataclass(eq=False)
class Thermostat(ElectricalDevice, HasMovableJoint[ActiveConnection1DOF], HasControls):
    """
    A device that is set to the temperature a heater keeps a room at.
    """


@dataclass(eq=False)
class Thermometer(HasRootBody):
    """
    A device that shows the temperature.
    """


# %% electrical devices


@dataclass(eq=False)
class Cable(HasRootBody):
    """
    A cable, such as a telephone's cord or a device's power cord.
    """


@dataclass(eq=False)
class PowerSocket(ElectricalDevice, HasLids):
    """
    A socket in a wall that devices are plugged into.
    """


@dataclass(eq=False)
class PowerStrip(ElectricalDevice, HasSwitches):
    """
    A strip of power sockets on a cable.
    """


@dataclass(eq=False)
class ElectricalPanel(ElectricalDevice, HasLids):
    """
    The box holding a building's fuses and circuit breakers.
    """


@dataclass(eq=False)
class Blender(ElectricalDevice, HasControls):
    """
    A kitchen appliance that blends food with spinning blades.
    """


@dataclass(eq=False)
class Fan(ElectricalDevice, HasSwitches):
    """
    A device with spinning blades that moves air.
    """


@dataclass(eq=False)
class Fryer(ElectricalDevice, HasButtons):
    """
    A kitchen appliance that fries food in hot oil or air.
    """


@dataclass(eq=False)
class Multicooker(ElectricalDevice, HasControls, HasLids):
    """
    A kitchen appliance that cooks in a pot in several ways, such as boiling, steaming
    or baking.
    """


@dataclass(eq=False)
class Intercom(ElectricalDevice):
    """
    A device by the door for talking to whoever rings and letting them in.
    """


@dataclass(eq=False)
class Telephone(ElectricalDevice, HasCarryingCables):
    """
    A telephone with a handset.
    """


@dataclass(eq=False)
class Printer(ElectricalDevice, HasDrawers):
    """
    A device that prints on paper.
    """


@dataclass(eq=False)
class Projector(ElectricalDevice, HasSwitches):
    """
    A device that projects pictures onto a wall or screen.
    """


@dataclass(eq=False)
class VacuumCleaner(ElectricalDevice, HasButtons, HasSwitches):
    """
    A device that cleans floors by suction.
    """


@dataclass(eq=False)
class WaterHeater(ElectricalDevice, HasControls, HasCables):
    """
    A device that heats and stores a building's hot water.
    """


# %% plumbing and sanitary fixtures


@dataclass(eq=False)
class Faucet(
    HasMovableJoint[RevoluteConnection],
    HasControls,
    HasFaucetSpouts,
    HasHandles,
    HasCarryingHoses,
    HasShowerHoses,
):
    """
    A fixture that water runs from, opened and closed by its controls.
    """


@dataclass(eq=False)
class FaucetSpout(HasMovableJoint[RevoluteConnection]):
    """
    The spout of a faucet that the water comes out of.
    """


@dataclass(eq=False)
class Shower(
    HasRootBody,
    HasControls,
    HasDoors,
    HasDrainPipes,
    HasFaucets,
    HasHoses,
    HasShowerHoses,
    HasCarryingShowerValves,
):
    """
    A fixture for washing under a spray of water, or the cabin it is in.
    """


@dataclass(eq=False)
class ShowerHead(HasRootBody):
    """
    The head of a shower that the water sprays from.
    """


@dataclass(eq=False)
class ShowerHose(HasRootBody, HasShowerHeads):
    """
    The flexible hose between a shower's valve and its head.
    """


@dataclass(eq=False)
class ShowerValve(HasRootBody, HasControls, HasShowerHoses):
    """
    The valve that a shower's water flow and temperature are set at.
    """


@dataclass(eq=False)
class ShowerHeadHolder(HasRootBody):
    """
    The bracket a shower head rests in.
    """


@dataclass(eq=False)
class Hose(HasRootBody):
    """
    A flexible tube that carries water.
    """


@dataclass(eq=False)
class DrainPipe(HasRootBody):
    """
    The pipe that drains a sink or another fixture.
    """


@dataclass(eq=False)
class Toilet(HasRootBody, HasBackplates, HasLids, HasToiletFlushes, HasToiletSeats):
    """
    A toilet.
    """


@dataclass(eq=False)
class ToiletSeat(HasMovableJoint[RevoluteConnection]):
    """
    The hinged seat of a toilet.
    """


@dataclass(eq=False)
class ToiletFlush(HasMovableJoint[ActiveConnection1DOF]):
    """
    The button or lever that flushes a toilet.
    """


@dataclass(eq=False)
class Bathtub(HasRootBody, HasFaucets, HasShowers):
    """
    A tub to bathe in.
    """


@dataclass(eq=False)
class Dispenser(HasRootBody, HasButtons, HasCaps, HasPumps):
    """
    A container that gives out its contents, such as soap, through a pump or a button.
    """


@dataclass(eq=False)
class Pump(HasMovableJoint[ActiveConnection1DOF]):
    """
    The head of a dispenser that is pressed to pump out its contents.
    """


# %% containers


@dataclass(eq=False)
class StorageBox(Container, HasLids):
    """
    A box things are kept in.
    """


@dataclass(eq=False)
class Suitcase(Container, HasLids):
    """
    A case for carrying clothes and belongings when travelling.
    """


@dataclass(eq=False)
class Jug(Container, HasLids):
    """
    A container with a handle and a spout for pouring liquids.
    """


@dataclass(eq=False)
class CoffeeCan(Container, HasHandles):
    """
    A can that coffee is kept in.
    """


# %% other


@dataclass(eq=False)
class Wheel(HasMovableJoint[RevoluteConnection]):
    """
    A wheel that a piece of furniture or a device rolls on.
    """


@dataclass(eq=False)
class Tabletop(HasRootBody):
    """
    The flat top of a table.
    """


@dataclass(eq=False)
class Clothing(HasGraspCandidates):
    """
    Something worn on the body, such as a shirt or a hoodie.
    """


@dataclass(eq=False)
class DoorAttachment(HasRootBody):
    """
    Something attached to a door to hold things, such as a rack or a pocket.
    """
