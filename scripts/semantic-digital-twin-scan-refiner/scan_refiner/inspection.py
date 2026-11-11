"""Shared triangle-distance inspection and portable browser review."""

from __future__ import annotations

# %% Imports
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as graph_objects
import trimesh
from krrood.adapters.json_serializer import DataclassJSONSerializer

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole, ProjectFile, RegistrationResult
from scan_refiner.rendering import DisplayCoordinates


# %% Distance computation
@dataclass
class SurfaceComparison:
    """Unsigned point-to-triangle distances in a declared common frame."""

    batch_size: int = 2000
    """Maximum query points passed to the mesh proximity accelerator at once."""

    def scan_to_model(
        self,
        scan_points: np.ndarray,
        root_mesh: trimesh.Trimesh,
        scan_T_root: np.ndarray,
    ) -> np.ndarray:
        """Return metres to the selected surface without interpreting missing coverage."""
        root_points = trimesh.transform_points(scan_points, np.linalg.inv(scan_T_root))
        distances = np.empty(len(root_points))
        for start in range(0, len(root_points), self.batch_size):
            stop = start + self.batch_size
            _, distances[start:stop], _ = trimesh.proximity.closest_point(
                root_mesh, root_points[start:stop]
            )
        return distances


# %% Browser review
@dataclass
class ViewerState:
    """Registration review state shared with the browser controls."""

    registration: RegistrationResult
    """Ranked candidate placements and numerical diagnostics."""
    selected_candidate: int | None
    """Candidate used to compute the displayed surface-distance layer."""
    accepted_candidate: int | None
    """Candidate explicitly accepted in the persistent project record."""
    has_heatmap: bool
    """Whether the selected placement has point-to-triangle distances."""
    display_origin: list[float]
    """Scan coordinates added to local browser positions to recover original positions."""
    scan_distances: list[float] | None
    """Optional heatmap colours sharing the single scan-coordinate buffer."""


@dataclass
class InspectionScene:
    """Plot geometry and review state prepared independently of HTML publication."""

    figure: graph_objects.Figure
    """A single scan trace and separately labelled geometry representations."""
    state: ViewerState
    """Display-only conversions and registration information for browser controls."""


@dataclass
class InspectionViewer:
    """One reusable browser inspection page generated from a frozen project."""

    project: RefinementProject
    """Project whose source data remain unchanged by page generation."""
    display_count: int = 20000
    """Maximum preview records displayed; this is not dense evidence access."""
    display_spacing: float = 0.015
    """Spatial spacing in metres for reproducible display sampling."""
    distance_limit: float = 0.1
    """Upper scan-to-model colour scale in metres."""

    def build_scene(
        self, candidate_index: int | None = None, heatmap: bool = False
    ) -> InspectionScene:
        """Prepare compact display buffers without altering the project's evidence."""
        registration = self.project.registration
        accepted = self.project.record.accepted_candidate
        if candidate_index is None and registration.candidates:
            candidate_index = accepted if accepted is not None else 0
        if candidate_index is not None and not 0 <= candidate_index < len(
            registration.candidates
        ):
            raise InvalidEvidence("Choose an existing candidate for inspection.")
        transform = (
            np.eye(4)
            if candidate_index is None
            else np.array(registration.candidates[candidate_index].scan_T_root)
        )
        snapshot = self.project.snapshot
        scan = self.project.scan
        indices = scan.sample_indices(self.display_spacing, self.display_count)
        points, colors = scan.points[indices], scan.colors[indices]
        display = DisplayCoordinates.from_points(points)
        rgb = [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in colors]
        figure = graph_objects.Figure()
        figure.add_trace(
            graph_objects.Scatter3d(
                x=display.points[:, 0],
                y=display.points[:, 1],
                z=display.points[:, 2],
                mode="markers",
                name="Scan RGB",
                marker=graph_objects.scatter3d.Marker(size=2, color=rgb),
                customdata=scan.source_indices[indices],
                hovertemplate="Preview record %{customdata}<extra>Scan</extra>",
                hoverinfo="skip",
            )
        )
        distances = None
        if heatmap and candidate_index is not None:
            role = GeometryRole(self.project.record.registration_geometry)
            combined = trimesh.util.concatenate(
                [item.mesh for item in snapshot.select(role)]
            )
            distances = SurfaceComparison().scan_to_model(points, combined, transform)
            figure.data[0].marker.update(
                colorscale="Turbo",
                cmin=0,
                cmax=self.distance_limit,
                colorbar=graph_objects.scatter3d.marker.ColorBar(title="Distance (m)"),
                showscale=False,
            )
        for surface in snapshot.meshes:
            mesh = surface.mesh
            vertices = (
                trimesh.transform_points(mesh.vertices, transform) - display.origin
            ).astype(np.float32)
            figure.add_trace(
                graph_objects.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=mesh.faces[:, 0].astype(np.uint32),
                    j=mesh.faces[:, 1].astype(np.uint32),
                    k=mesh.faces[:, 2].astype(np.uint32),
                    name=surface.body_name,
                    legendgroup=str(surface.role),
                    legendgrouptitle_text=str(surface.role).title(),
                    showlegend=True,
                    color="#f3a33c"
                    if surface.role == GeometryRole.VISUAL
                    else "#59b6dc",
                    opacity=1.0,
                    visible=surface.role == GeometryRole.VISUAL,
                    hovertemplate="%{fullData.name}<extra></extra>",
                )
            )
        figure.update_layout(
            margin=graph_objects.layout.Margin(l=0, r=0, t=0, b=0),
            scene=graph_objects.layout.Scene(
                aspectmode="data",
                dragmode="orbit",
                xaxis_title="Local X (m)",
                yaxis_title="Local Y (m)",
                zaxis_title="Local Z (m)",
            ),
            uirevision="inspection",
            legend=graph_objects.layout.Legend(groupclick="toggleitem"),
        )
        state = ViewerState(
            registration,
            candidate_index,
            accepted,
            heatmap and candidate_index is not None,
            display.origin.tolist(),
            None if distances is None else distances.tolist(),
        )
        return InspectionScene(figure, state)

    def write(self, candidate_index: int | None = None, heatmap: bool = False) -> Path:
        """Write a self-contained page; alternative hypotheses remain inspectable."""
        scene = self.build_scene(candidate_index, heatmap)
        template = (Path(__file__).parent / "web" / "viewer.html").read_text()
        page = template.replace(
            "__PROJECT_TITLE__", html.escape(self.project.directory.name)
        )
        page = page.replace(
            "__FIGURE__",
            scene.figure.to_html(
                full_html=False, include_plotlyjs=True, div_id="scene"
            ),
        )
        payload = json.dumps(DataclassJSONSerializer.to_json(scene.state)).replace(
            "<", "\\u003c"
        )
        page = page.replace("__VIEWER_STATE__", payload)
        page = page.replace(
            "__CONTROLS__", (Path(__file__).parent / "web" / "viewer.js").read_text()
        )
        destination = self.project.directory / ProjectFile.VIEWER
        temporary = destination.with_suffix(".partial")
        temporary.write_text(page)
        temporary.replace(destination)
        return destination
