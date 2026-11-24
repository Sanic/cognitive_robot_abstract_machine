"""
Explicit geometric membership for semantic and saved inspection assemblies.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
import re
from typing_extensions import TYPE_CHECKING

from scan_refiner.exceptions import InvalidEvidence
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% Inspection membership
@dataclass(kw_only=True)
class AssemblyMembership:
    """
    A selectable set of exact qualified geometry identities.
    """

    identifier: str
    """
    Stable identity independent of the display label.
    """

    label: str
    """
    Human-facing name shown in inspection and copied context.
    """

    body_names: list[str]
    """
    Bodies that contribute meshes to the assembly bounds.
    """

    root_name: str = ""
    """
    Semantic root frame, when this group was captured from a world annotation.
    """

    def validate(self) -> None:
        """
        Require a named group with at least two distinct geometric members.
        """
        if (
            not self.identifier
            or not self.label.strip()
            or len(self.body_names) < 2
            or not all(self.body_names)
            or len(set(self.body_names)) != len(self.body_names)
        ):
            raise InvalidEvidence(
                reason="An assembly needs a name and at least two distinct bodies."
            )

    @classmethod
    def discover(
        cls, *, world: World, geometric_bodies: set[str]
    ) -> list[AssemblyMembership]:
        """
        Capture declared semantic branches that contain more than one geometric body.
        """
        groups = {}
        for annotation in world.semantic_annotations:
            if not isinstance(annotation, HasRootBody):
                continue
            members = geometric_bodies.intersection(
                str(entity.name) for entity in annotation.kinematic_structure_entities
            )
            if len(members) < 2:
                continue
            root = str(annotation.root.name)
            kind = type(annotation)
            identifier = f"{kind.__module__}.{kind.__qualname__}:{root}"
            label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", kind.__name__).capitalize()
            groups[identifier] = cls(
                identifier=identifier,
                label=label,
                body_names=sorted(members),
                root_name=root,
            )
        return sorted(groups.values(), key=lambda group: (group.label, group.root_name))
