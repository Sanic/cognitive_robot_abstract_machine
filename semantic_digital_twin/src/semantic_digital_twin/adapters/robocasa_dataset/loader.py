from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from typing_extensions import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import numpy
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaKitchenApplianceResolver,
    RoboCasaObjectCategory,
    RoboCasaObjectResolver,
)
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.semantic_annotations.mixins import (
    HasDoors,
    HasDrawers,
    HasHandle,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Drawer,
    Handle,
)
from semantic_digital_twin.utils import camel_case_split
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

logger = logging.getLogger(__name__)

try:
    import yaml
    from robocasa.models.arenas.kitchen_arena import KitchenArena
    from robocasa.models.scenes import scene_builder, scene_registry
    from robocasa.models.scenes.scene_builder import FIXTURES
    from robosuite.models.tasks import ManipulationTask
except ImportError:
    logger.warning(
        "robocasa/robosuite are required for RoboCasa dataset loading. Install robosuite from git "
        "('pip install git+https://github.com/ARISE-Initiative/robosuite.git') and then robocasa "
        "('pip install -e .' from a clone of https://github.com/robocasa/robocasa), then run "
        "'python -m robocasa.scripts.download_kitchen_assets' to fetch the fixture/object assets."
    )


def _mjcf_document_from_element_copy(element: ET.Element) -> str:
    """
    Wrap a copy of a single MJCF XML element (for example one kitchen appliance's geometry) into a
    minimal standalone MJCF document so it can be parsed on its own. A copy is used so the original
    element is not reparented out of whatever tree RoboCasa still holds it in.

    :param element: The XML element to wrap, typically a RoboCasa fixture's underlying body element.
    :return: The MJCF document as a string.
    """
    root = ET.Element("mujoco")
    worldbody = ET.SubElement(root, "worldbody")
    worldbody.append(copy.deepcopy(element))
    return ET.tostring(root, encoding="unicode")


def _category_from_class_name(class_name: str) -> str:
    """
    Convert a RoboCasa ``Fixture`` subclass name (upper camel case, for example
    ``"HingeCabinet"``) into a lower snake case category string (for example
    ``"hinge_cabinet"``) suitable for
    :meth:`~semantic_digital_twin.adapters.robocasa_dataset.semantics.RoboCasaCategoryResolver.resolve`.

    :param class_name: The RoboCasa ``Fixture`` subclass name.
    :return: The lower snake case category string.
    """
    return "_".join(token.lower() for token in camel_case_split(class_name))


@dataclass
class RoboCasaDatasetLoader:
    """
    Loader for objects, kitchen appliances, and full kitchen scenes from the RoboCasa dataset
    (https://github.com/robocasa/robocasa).

    RoboCasa composes its assets with robosuite/MuJoCo. This loader drives RoboCasa's own Python
    composition code to build the MJCF for a requested kitchen appliance or kitchen, and parses the
    resulting MJCF with the existing :class:`~semantic_digital_twin.adapters.mjcf.MJCFParser`.

    For this to work, ``robocasa`` and ``robosuite`` (installed from git) must be available, and the
    RoboCasa fixture/object assets must be downloaded via
    ``python -m robocasa.scripts.download_kitchen_assets``.

    .. note::
        RoboCasa does not version its internal Python module layout as a stable public API. The import
        paths used here match the module layout at the time this loader was written; if they no longer
        match the installed version, adjust the module-level imports at the top of this file
        accordingly.
    """

    directory: Path = field(
        default_factory=lambda: Path.home() / "robocasa-assets",
    )
    """
    The directory where the RoboCasa fixture/object assets were downloaded to via
    ``python -m robocasa.scripts.download_kitchen_assets``.
    """

    kitchen_appliance_annotator: RoboCasaKitchenApplianceResolver = field(
        default_factory=RoboCasaKitchenApplianceResolver
    )
    """
    Resolver mapping RoboCasa fixture category names to SemanticAnnotation subclasses.
    """

    object_annotator: RoboCasaObjectResolver = field(
        default_factory=RoboCasaObjectResolver
    )
    """
    Resolver mapping RoboCasa object category names to SemanticAnnotation subclasses.
    """

    def load_kitchen(
        self,
        layout_id: LayoutType,
        style_id: StyleType,
        random_number_generator: Optional[numpy.random.Generator] = None,
    ) -> World:
        """
        Compose a full RoboCasa kitchen scene and parse it into a World.

        :param layout_id: A member of ``robocasa.models.scenes.scene_registry.LayoutType``.
        :param style_id: A member of ``robocasa.models.scenes.scene_registry.StyleType``.
        :param random_number_generator: The random number generator used for the (deterministic,
            non-physical) fixture placement within the layout. Defaults to RoboCasa's own default
            generator.
        :return: The composed world, with a SemanticAnnotation attached to each appliance's root body.
        """
        with open(scene_registry.get_layout_path(layout_id)) as layout_file:
            layout_config = yaml.safe_load(layout_file)
        with open(scene_registry.get_style_path(style_id)) as style_file:
            style_config = yaml.safe_load(style_file)

        arena = KitchenArena(
            layout_id=layout_id, style_id=style_id, rng=random_number_generator
        )
        kitchen_appliances = scene_builder.create_fixtures(
            layout_config, style_config, rng=random_number_generator
        )

        task = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[],
            mujoco_objects=list(kitchen_appliances.values()),
        )

        world = MJCFParser.from_xml_string(task.get_xml()).parse()
        self._apply_kitchen_appliance_semantics(world, kitchen_appliances)
        return world

    def load_kitchen_appliance(
        self, category: RoboCasaKitchenApplianceCategory, **kitchen_appliance_kwargs
    ) -> World:
        """
        Load a single RoboCasa kitchen appliance (for example a cabinet or a microwave) as a
        standalone world.

        :param category: The appliance category, a key of
            ``robocasa.models.scenes.scene_builder.FIXTURES``.
        :param kitchen_appliance_kwargs: Extra keyword arguments forwarded to the appliance's
            (RoboCasa ``Fixture`` subclass) constructor.
        :return: The loaded world, with a SemanticAnnotation attached to the appliance's root body.
        """
        kitchen_appliance_class = FIXTURES[category]
        kitchen_appliance = kitchen_appliance_class(
            name=category, **kitchen_appliance_kwargs
        )

        world = MJCFParser.from_xml_string(
            _mjcf_document_from_element_copy(kitchen_appliance._obj)
        ).parse()
        self._apply_kitchen_appliance_semantics(world, {category: kitchen_appliance})
        return world

    def load_object(
        self, category: RoboCasaObjectCategory, instance_index: int = 0
    ) -> World:
        """
        Load a single RoboCasa object as a standalone world.

        :param category: The object category, a key of
            ``robocasa.models.objects.kitchen_objects.OBJ_CATEGORIES``.
        :param instance_index: Which of the category's downloaded asset instances to load.
        :return: The loaded world, with a SemanticAnnotation attached to the object's root body.
        """
        model_files = sorted((self.directory / "objects" / category).glob("**/model.xml"))
        if not model_files:
            raise FileNotFoundError(
                f"No downloaded assets found for object category '{category}' in "
                f"{self.directory / 'objects' / category}. "
                "Run 'python -m robocasa.scripts.download_kitchen_assets' first."
            )
        if instance_index >= len(model_files):
            raise IndexError(
                f"Requested instance_index {instance_index} for object category '{category}', "
                f"but only {len(model_files)} downloaded instance(s) were found in "
                f"{self.directory / 'objects' / category}."
            )

        world = MJCFParser(str(model_files[instance_index])).parse()
        self._apply_object_semantics(world, category)
        return world

    def _apply_kitchen_appliance_semantics(
        self, world: World, kitchen_appliances: Dict[str, Any]
    ) -> None:
        """
        Attach a SemanticAnnotation to the root body of each kitchen appliance, using the appliance
        annotator where the appliance's category matches a known SemanticAnnotation subclass, and
        falling back to NaturalLanguageWithTypeDescription otherwise.

        :param world: The world the appliances were parsed into.
        :param kitchen_appliances: Mapping from appliance name to the RoboCasa ``Fixture`` instance
            it was built from.
        """
        for appliance_name, kitchen_appliance in kitchen_appliances.items():
            body = self._find_body(world, appliance_name)
            if body is None:
                continue
            category = _category_from_class_name(type(kitchen_appliance).__name__)
            self._attach_semantic_annotation(world, body, category)

    def _apply_object_semantics(
        self, world: World, category: RoboCasaObjectCategory
    ) -> None:
        """
        Attach a SemanticAnnotation to the root body of a loaded object. The object's own body is the
        first body in the world with collision geometry: MJCFParser.parse() always creates an empty
        placeholder root body named after the MJCF worldbody, distinct from the loaded content.

        :param world: The world the object was parsed into.
        :param category: The RoboCasa object category the object belongs to.
        """
        bodies_with_collision = world.bodies_with_collision
        if not bodies_with_collision:
            raise ValueError(
                f"No body with collision geometry found for object category '{category}'."
            )
        self._attach_semantic_annotation(world, bodies_with_collision[0], category)

    def _attach_semantic_annotation(self, world: World, body: Body, category: str) -> None:
        """
        Attach the SemanticAnnotation matching ``category`` to ``body``, falling back to
        NaturalLanguageWithTypeDescription if no matching SemanticAnnotation subclass is known, and
        attaching any handle/door/drawer sub-part annotations found under ``body``.

        :param world: The world ``body`` belongs to.
        :param body: The body to annotate.
        :param category: The RoboCasa appliance or object category of ``body``.
        """
        annotation_class = self.kitchen_appliance_annotator.resolve(
            category
        ) or self.object_annotator.resolve(category)

        with world.modify_world():
            if annotation_class is not None:
                annotation = annotation_class(root=body)
            else:
                annotation = NaturalLanguageWithTypeDescription(
                    root=body, description=category, type_description=category
                )
            world.add_semantic_annotation(annotation)
            self._attach_sub_part_annotations(world, annotation, body)

    def _attach_sub_part_annotations(
        self, world: World, parent_annotation: SemanticAnnotation, parent_body: Body
    ) -> None:
        """
        Detect handle/door/drawer bodies already present under a kitchen appliance's root body
        (RoboCasa appliances like cabinets ship these as real articulated sub-bodies in their own
        MJCF, not something this adapter synthesizes) and attach them as parts of the nearest
        enclosing SemanticAnnotation by their RoboCasa body-naming convention (mirroring the
        naming-convention detection ``adapters/procthor/procthor_pipelines.py`` already uses for
        ProcTHOR dressers).

        Each match is attached via :meth:`~semantic_digital_twin.semantic_annotations.mixins.PartWholeRelationship.add`,
        the framework's normal part-whole mechanism, recursing into each direct child so that, for
        example, a handle nested inside a door is attached to the door's annotation rather than the
        enclosing cabinet's. This is safe: a sub-body is only ever offered to its own direct
        kinematic parent, so ``World.move_branch`` (invoked by ``add()``'s default mount strategy) is
        a no-op re-parent that leaves the body's real connection type and degree of freedom
        untouched.

        :param world: The world ``parent_body`` belongs to.
        :param parent_annotation: The SemanticAnnotation of ``parent_body`` to attach newly found
            direct sub-parts to.
        :param parent_body: The body to search direct children of for sub-part bodies.
        """
        for child_body in parent_body.child_kinematic_structure_entities:
            if not isinstance(child_body, Body):
                continue
            child_name = child_body.name.name.lower()
            child_annotation = parent_annotation

            if (
                "handle" in child_name
                and isinstance(parent_annotation, HasHandle)
                and parent_annotation.handle is None
            ):
                child_annotation = Handle(root=child_body)
                world.add_semantic_annotation(child_annotation)
                parent_annotation.add(child_annotation)
            elif "door" in child_name and isinstance(parent_annotation, HasDoors):
                child_annotation = Door(root=child_body)
                world.add_semantic_annotation(child_annotation)
                parent_annotation.add(child_annotation)
            elif "drawer" in child_name and isinstance(parent_annotation, HasDrawers):
                child_annotation = Drawer(root=child_body)
                world.add_semantic_annotation(child_annotation)
                parent_annotation.add(child_annotation)

            self._attach_sub_part_annotations(world, child_annotation, child_body)

    @staticmethod
    def _find_body(world: World, name: str) -> Optional[Body]:
        """
        Look up a body by name, returning None instead of raising if it is not present. Falls back to
        the first body whose name starts with ``name`` if no exact match exists, since robosuite may
        rename a merged object's root body (for example to ``f"{name}_main"``).

        :param world: The world to search.
        :param name: The name of the body to look up.
        :return: The body, or None if no matching body exists in the world.
        """
        try:
            return world.get_body_by_name(name)
        except WorldEntityNotFoundError:
            pass

        matching_bodies = [body for body in world.bodies if body.name.name.startswith(name)]
        return matching_bodies[0] if matching_bodies else None
