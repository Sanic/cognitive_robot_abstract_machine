"""
Body-labelled cross sections in the original metric scan frame.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
import numpy as np
import trimesh
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame
from scan_refiner.project import RefinementProject
from scan_refiner.inspection_state import SectionPlane, GeometryVisibility


# %% Section geometry
@dataclass(kw_only=True)
class SectionOutline:
    """
    Segments cut from one body at the requested inspection plane.
    """

    body_name: str
    """
    Qualified world body identity.
    """

    role: str
    """
    Visual or collision surface representation.
    """

    segments: list[list[list[float]]]
    """
    Unshifted inspection-frame endpoint pairs, in metres.
    """


@dataclass(kw_only=True)
class ModelSection:
    """
    Intersect frozen world triangles with a fixed inspection-coordinate plane.
    """

    project: RefinementProject
    """
    Geometry baseline and candidate placements.
    """

    def build(
        self,
        *,
        plane: str,
        position: float,
        candidate: int | None,
        role: str,
        frame_T_scan: np.ndarray | None = None,
    ) -> list[SectionOutline]:
        """
        Return labelled segments without modifying source meshes or placements.
        """
        if (
            plane not in (SectionPlane.XY, SectionPlane.XZ, SectionPlane.YZ)
            or not np.isfinite(position)
            or role not in list(GeometryVisibility)
        ):
            raise InvalidEvidence(
                reason="Choose a metric section plane and geometry role."
            )
        transform = np.eye(4)
        if candidate is not None:
            if not 0 <= candidate < len(self.project.registration.candidates):
                raise InvalidEvidence(
                    reason="Choose an existing registration candidate."
                )
            transform = np.array(
                self.project.registration.candidates[candidate].scan_T_root
            )
        frame = (
            CoordinateFrame()
            if frame_T_scan is None
            else CoordinateFrame(frame_T_scan=frame_T_scan)
        )
        transform = frame.frame_T_scan @ transform
        normal_axis = (SectionPlane.YZ, SectionPlane.XZ, SectionPlane.XY).index(plane)
        normal = np.eye(3)[normal_axis]
        origin = normal * position
        outlines = []
        for surface in self.project.effective_snapshot.meshes:
            if role != GeometryVisibility.BOTH and surface.role != role:
                continue
            mesh = surface.mesh.copy()
            mesh.apply_transform(transform)
            segments = trimesh.intersections.mesh_plane(mesh, normal, origin)
            if len(segments):
                outlines.append(
                    SectionOutline(
                        body_name=surface.body_name,
                        role=str(surface.role),
                        segments=segments.tolist(),
                    )
                )
        return outlines
