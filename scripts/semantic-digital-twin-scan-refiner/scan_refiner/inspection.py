"""Shared triangle-distance inspection and portable browser review."""

from __future__ import annotations

# %% Imports
import html
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as graph_objects
import trimesh
from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood.patterns.path_enumeration import PathEnumeration

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.assemblies import AssemblyMembership
from scan_refiner.comparison import SurfaceComparison
from scan_refiner.point_records import PointLayoutMetadata, PointRecordLayout
from scan_refiner.dimensions import BodyDimensions
from scan_refiner.refinement import BodyTranslation
from scan_refiner.project import RefinementProject
from scan_refiner.records import (
    GeometryRole,
    ProjectFile,
    ProjectRecord,
    RegistrationResult,
)
from scan_refiner.rendering import DisplayCoordinates
from scan_refiner.world import KinematicNode


# %% Browser review
class InspectionWebFile(PathEnumeration):
    """Browser assets included in every standalone inspection page."""

    TEMPLATE = "web/viewer.html"
    STYLES = "web/workspace.css"
    FRAME = "web/coordinate_frame.js"
    WORKSPACE = "web/workspace.js"
    ASSEMBLIES = "web/assemblies.js"
    WORLD_TREE = "web/world_tree.js"
    JOBS = "web/jobs.js"
    INVESTIGATION = "web/investigation.js"
    CHANGE_SETS = "web/change_sets.js"
    REPORT = "web/comparison_report.js"
    REFINEMENT = "web/refinement.js"
    BUFFERS = "web/plot_buffers.js"
    READY = "web/plot_ready.js"
    CLOUD = "web/scan_cloud.js"
    TOOLS = "web/inspection_tools.js"
    CONTROLS = "web/viewer.js"


@dataclass(kw_only=True)
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
    project_record: ProjectRecord | None = None
    """Source identities and baseline metadata available to the workspace."""
    distance_limit: float = 0.1
    """Fixed distance colour-scale maximum in metres."""
    heatmap_role: str = GeometryRole.VISUAL.value
    """Surface role used by the embedded initial distance sample."""
    overrides: list[BodyTranslation] = field(default_factory=list)
    """Translations already included in the generated mesh buffers."""
    dimension_overrides: list[BodyDimensions] = field(default_factory=list)
    """Dimensional changes already included in the generated mesh buffers."""
    point_layout: PointLayoutMetadata = field(
        default_factory=lambda: PointRecordLayout().metadata
    )
    """Binary result layout available even without a dense index."""
    assemblies: list[AssemblyMembership] = field(default_factory=list, kw_only=True)
    """Semantic assembly membership captured with the frozen world."""
    kinematic_nodes: list[KinematicNode] = field(default_factory=list, kw_only=True)
    """Parent links for all entities in the frozen kinematic tree."""


@dataclass(kw_only=True)
class InspectionScene:
    """Plot geometry and review state prepared independently of HTML publication."""

    figure: graph_objects.Figure
    """A single scan trace and separately labelled geometry representations."""
    state: ViewerState
    """Display-only conversions and registration information for browser controls."""


@dataclass(kw_only=True)
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
        self, *, candidate_index: int | None = None, heatmap: bool = False
    ) -> InspectionScene:
        """Prepare compact display buffers without altering the project's evidence."""
        registration = self.project.registration
        accepted = self.project.record.accepted_candidate
        if candidate_index is None and registration.candidates:
            candidate_index = accepted if accepted is not None else 0
        if candidate_index is not None and not 0 <= candidate_index < len(
            registration.candidates
        ):
            raise InvalidEvidence(reason="Choose an existing candidate for inspection.")
        transform = (
            np.eye(4)
            if candidate_index is None
            else np.array(registration.candidates[candidate_index].scan_T_root)
        )
        snapshot = self.project.effective_snapshot
        scan = self.project.scan
        indices = scan.sample_indices(
            spacing=self.display_spacing, maximum_count=self.display_count
        )
        points, colors = scan.points[indices], scan.colors[indices]
        display = DisplayCoordinates.from_points(points=points)
        rgb = [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in colors]
        figure = graph_objects.Figure()
        figure.add_trace(
            graph_objects.Scatter3d(
                x=display.points[:, 0],
                y=display.points[:, 1],
                z=display.points[:, 2],
                mode="markers",
                name="Scan RGB",
                marker=graph_objects.scatter3d.Marker(
                    size=2, color=rgb, colorscale="RdYlGn_r", showscale=False
                ),
                customdata=scan.source_indices[indices],
                hovertemplate="Scan record %{customdata}<extra>Scan</extra>",
                hoverinfo="skip",
            )
        )
        distances = None
        if heatmap and candidate_index is not None:
            role = GeometryRole(self.project.record.registration_geometry)
            combined = trimesh.util.concatenate(
                [item.mesh for item in snapshot.select(role=role)]
            )
            distances = SurfaceComparison().scan_to_model(
                scan_points=points, root_mesh=combined, scan_T_root=transform
            )
            figure.data[0].marker.update(
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
                camera=graph_objects.layout.scene.Camera(
                    eye=graph_objects.layout.scene.camera.Eye(x=2.2, y=2.2, z=1.8)
                ),
                xaxis_title="Local X (m)",
                yaxis_title="Local Y (m)",
                zaxis_title="Local Z (m)",
            ),
            uirevision="inspection",
            legend=graph_objects.layout.Legend(groupclick="toggleitem"),
            showlegend=False,
            paper_bgcolor="#ffffff",
            font=graph_objects.layout.Font(
                family="Arial, sans-serif", color="#68818c", size=10
            ),
        )
        for axis in (
            figure.layout.scene.xaxis,
            figure.layout.scene.yaxis,
            figure.layout.scene.zaxis,
        ):
            axis.update(
                backgroundcolor="#f1f6f8",
                gridcolor="#dce6e9",
                zerolinecolor="#c5d5db",
                showspikes=False,
            )
        state = ViewerState(
            registration=registration,
            selected_candidate=candidate_index,
            accepted_candidate=accepted,
            has_heatmap=heatmap and candidate_index is not None,
            display_origin=display.origin.tolist(),
            scan_distances=None if distances is None else distances.tolist(),
            project_record=self.project.record,
            overrides=self.project.current_state.overrides,
            dimension_overrides=self.project.current_state.dimension_overrides,
            distance_limit=self.distance_limit,
            heatmap_role=str(self.project.record.registration_geometry),
            assemblies=snapshot.assemblies,
            kinematic_nodes=snapshot.kinematic_nodes,
        )
        return InspectionScene(figure=figure, state=state)

    def write(
        self, *, candidate_index: int | None = None, heatmap: bool = False
    ) -> Path:
        """Write a self-contained page; alternative hypotheses remain inspectable."""
        scene = self.build_scene(candidate_index=candidate_index, heatmap=heatmap)
        template = (Path(__file__).parent / InspectionWebFile.TEMPLATE).read_text()
        page = template.replace(
            "__PROJECT_TITLE__", html.escape(self.project.directory.name)
        )
        page = page.replace(
            "__STYLES__", (Path(__file__).parent / InspectionWebFile.STYLES).read_text()
        )
        page = page.replace("__DISTANCE_LIMIT_MM__", f"{self.distance_limit * 1000:g}")
        page = page.replace(
            "__FIGURE__",
            scene.figure.to_html(
                full_html=False,
                include_plotlyjs=True,
                div_id="scene",
                post_script=(
                    Path(__file__).parent / InspectionWebFile.READY
                ).read_text(),
            ),
        )
        payload = json.dumps(
            DataclassJSONSerializer.to_json(instance=scene.state)
        ).replace("<", "\\u003c")
        page = page.replace("__VIEWER_STATE__", payload)
        page = page.replace(
            "__CONTROLS__",
            "\n".join(
                (Path(__file__).parent / asset).read_text()
                for asset in (
                    InspectionWebFile.BUFFERS,
                    InspectionWebFile.CLOUD,
                    InspectionWebFile.FRAME,
                    InspectionWebFile.REPORT,
                    InspectionWebFile.JOBS,
                    InspectionWebFile.REFINEMENT,
                    InspectionWebFile.INVESTIGATION,
                    InspectionWebFile.CHANGE_SETS,
                    InspectionWebFile.ASSEMBLIES,
                    InspectionWebFile.WORLD_TREE,
                    InspectionWebFile.WORKSPACE,
                    InspectionWebFile.TOOLS,
                    InspectionWebFile.CONTROLS,
                )
            ),
        )
        destination = self.project.directory / ProjectFile.VIEWER
        temporary = destination.with_suffix(".partial")
        temporary.write_text(page)
        temporary.replace(destination)
        return destination
