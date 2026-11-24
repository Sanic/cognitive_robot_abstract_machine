"""Reversible inspection settings expressed in original scan coordinates."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame
from scan_refiner.assemblies import AssemblyMembership


# %% Display vocabulary
class GeometryVisibility(StrEnum):
    """Surface representations offered by the inspection controls."""

    VISUAL = "visual"
    COLLISION = "collision"
    BOTH = "both"
    NONE = "none"


class ScanDisplay(StrEnum):
    """Available evidence colouring and visibility choices."""

    RGB = "rgb"
    HEATMAP = "heatmap"
    NONE = "none"


class SectionPlane(StrEnum):
    """Fixed scan planes or an unrestricted spatial view."""

    SPACE = "3d"
    XY = "xy"
    XZ = "xz"
    YZ = "yz"


class CameraProjection(StrEnum):
    """Projection choices retained in saved viewpoints."""

    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"


class InspectionFrame(StrEnum):
    """Reference axes offered for metric selection and sectioning."""

    SCAN = "scan"
    WORLD = "world"


# %% Camera coordinates
@dataclass(kw_only=True)
class CameraSettings:
    """Viewpoint and orientation for a saved inspection."""

    eye: list[float] = field(default_factory=lambda: [2.2, 2.2, 1.8])
    """Viewpoint relative to the scene centre."""
    up: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    """Camera vertical direction."""
    center: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Camera target relative to the scene centre."""
    projection: str = CameraProjection.PERSPECTIVE.value
    """Perspective or orthographic projection."""


# %% Inspection records
@dataclass(kw_only=True)
class ViewSettings:
    """Display choices and query descriptors without storing point buffers."""

    geometry_mode: str = GeometryVisibility.VISUAL.value
    """Visible geometry role or combination."""
    scan_mode: str = ScanDisplay.RGB.value
    """Original colour, distance colouring or hidden evidence."""
    selected_candidate: int | None = None
    """Provisional placement being inspected."""
    selected_body: str = ""
    """Current model body context, independent of visible geometry layers."""
    coordinate_frame: str = InspectionFrame.SCAN.value
    """Reference axes for displayed geometry, selections and sections."""
    frame_yaw: float = 0.0
    """Additional rotation of inspection axes about vertical, in degrees."""
    selection_lower: list[float] = field(default_factory=list)
    """Current selection lower bounds, separate from loaded evidence."""
    selection_upper: list[float] = field(default_factory=list)
    """Current selection upper bounds in the active inspection frame."""
    region_frame_T_scan: list[list[float]] = field(default_factory=list)
    """Frozen crop frame for reproducing the loaded evidence after view changes."""
    comparison_job: str = ""
    """Derived comparison identity retained for undo and named-view restoration."""
    focus_enabled: bool = False
    """Whether the scene axes focus the selected crop rather than the whole world."""
    point_size: float = 2.0
    """Rendered marker diameter in pixels."""
    mesh_opacity: float = 1.0
    """Surface opacity independent of evidence colour."""
    point_budget: int = 50000
    """Maximum indexed records loaded for this view."""
    region_lower: list[float] = field(default_factory=list)
    """Inclusive lower scan coordinates of the loaded region."""
    region_upper: list[float] = field(default_factory=list)
    """Inclusive upper scan coordinates of the loaded region."""
    clip_enabled: bool = False
    """Whether the displayed scan is restricted to the loaded box."""
    clip_lower: list[float] = field(default_factory=list)
    """Lower display clip bounds independent of the loaded query."""
    clip_upper: list[float] = field(default_factory=list)
    """Upper display clip bounds independent of the loaded query."""
    section_axis: str = SectionPlane.SPACE.value
    """Full scene or fixed metric XY, XZ or YZ section."""
    section_position: float = 0.0
    """Scan coordinate along the section normal, in metres."""
    section_thickness: float = 0.02
    """Total scan slab thickness in metres."""
    camera: CameraSettings = field(default_factory=CameraSettings)
    """Plotly camera coordinates for restoring annotated views."""
    selected_assembly: str = field(default="", kw_only=True)
    """Assembly identity whose combined scope the browser is inspecting."""

    def validate(self) -> None:
        """Reject invalid settings before any project state is published."""
        if self.selected_assembly and self.selected_body:
            raise InvalidEvidence(reason="Select one body or one assembly.")
        if (
            self.coordinate_frame not in list(InspectionFrame)
            or not np.isfinite(self.frame_yaw)
            or self.geometry_mode not in list(GeometryVisibility)
            or self.scan_mode not in list(ScanDisplay)
            or self.section_axis not in list(SectionPlane)
            or not 0 < self.point_budget <= 150000
            or not 1 <= self.point_size <= 5
            or not 0 <= self.mesh_opacity <= 1
            or not np.isfinite([self.section_position, self.section_thickness]).all()
            or self.section_thickness <= 0
        ):
            raise InvalidEvidence(reason="Invalid inspection settings.")
        if self.region_frame_T_scan:
            CoordinateFrame(frame_T_scan=np.array(self.region_frame_T_scan))
        if self.camera.projection not in list(CameraProjection) or any(
            len(vector) != 3 or not np.isfinite(vector).all()
            for vector in (self.camera.eye, self.camera.up, self.camera.center)
        ):
            raise InvalidEvidence(
                reason="Supply finite camera coordinates and a valid projection."
            )
        for lower, upper in (
            (self.region_lower, self.region_upper),
            (self.clip_lower, self.clip_upper),
            (self.selection_lower, self.selection_upper),
        ):
            if lower or upper:
                if (
                    len(lower) != 3
                    or len(upper) != 3
                    or not np.isfinite([lower, upper]).all()
                    or np.any(np.array(lower) > upper)
                ):
                    raise InvalidEvidence(reason="Supply ordered finite scan bounds.")


@dataclass(kw_only=True)
class Measurement:
    """A labelled straight-line distance between two scan coordinates."""

    label: str = "Distance"
    """User annotation identifying the measured feature."""
    start: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """First endpoint in original scan coordinates, metres."""
    end: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Second endpoint in original scan coordinates, metres."""

    @property
    def distance(self) -> float:
        """Return the Euclidean endpoint distance in metres."""
        return float(np.linalg.norm(np.array(self.end) - self.start))


@dataclass(kw_only=True)
class SavedView:
    """Named inspection settings with an explanatory note."""

    label: str = "Inspection"
    """Name shown in the saved-view selector."""
    note: str = ""
    """Observation associated with this view."""
    view: ViewSettings = field(default_factory=ViewSettings)
    """Independent copy of the camera and evidence query."""


@dataclass(kw_only=True)
class InspectionState:
    """View settings and annotations referring to frozen evidence."""

    view: ViewSettings = field(default_factory=ViewSettings)
    """Current inspection controls."""
    measurements: list[Measurement] = field(default_factory=list)
    """Recorded metric endpoint pairs."""
    bookmarks: list[SavedView] = field(default_factory=list)
    """Annotated views available for later restoration."""
    saved_assemblies: list[AssemblyMembership] = field(
        default_factory=list, kw_only=True
    )
    """User-defined assembly selections stored with reversible inspection decisions."""

    def validate(self) -> None:
        """Validate current settings, saved views and metric endpoints."""
        self.view.validate()
        if len({item.identifier for item in self.saved_assemblies}) != len(
            self.saved_assemblies
        ):
            raise InvalidEvidence(reason="Saved assembly identities must be distinct.")
        for assembly in self.saved_assemblies:
            assembly.validate()
            if assembly.root_name:
                raise InvalidEvidence(
                    reason="Saved assemblies use explicit members rather than a semantic root."
                )
        for bookmark in self.bookmarks:
            bookmark.view.validate()
        for measurement in self.measurements:
            if (
                len(measurement.start) != 3
                or len(measurement.end) != 3
                or not np.isfinite([measurement.start, measurement.end]).all()
            ):
                raise InvalidEvidence(
                    reason="Measurements require two finite XYZ endpoints."
                )
