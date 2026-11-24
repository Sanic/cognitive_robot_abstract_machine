"""
Focused agent images using the same scene builder and geometry as the web viewer.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
import os
import subprocess
from uuid import uuid4
import json
import numpy as np
import trimesh
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.agent.models import (
    RegionRenderQuery,
    SampleSummary,
    RenderArtifact,
    AgentFile,
    RenderView,
)
from scan_refiner.inspection import InspectionViewer
from scan_refiner.project import RefinementProject
from scan_refiner.records import RegistrationResult
from scan_refiner.world import WorldSnapshot
from scan_refiner.point_records import PointField
from scan_refiner.coverage import ComparisonArray
from scan_refiner.exceptions import InvalidEvidence


# %% Bounded scene capture
@dataclass(kw_only=True)
class AgentRenderer:
    """
    Generate immutable evidence views without changing the inspected project's camera.
    """

    project: RefinementProject
    """
    Project that owns the resulting image and provenance artifacts.
    """

    browser_executable: str | None = None
    """
    Existing headless Chromium; absent means PNG requests are rejected explicitly.
    """

    width: int = 1280
    """
    Fixed comparison image width, in pixels.
    """

    height: int = 900
    """
    Fixed comparison image height, in pixels.
    """

    timeout_seconds: float = 90
    """
    Maximum browser screenshot process duration.
    """

    def render_evidence(
        self,
        *,
        query: RegionRenderQuery,
        sample: SampleSummary,
        snapshot: WorldSnapshot,
        registration: RegistrationResult,
        candidate: int | None,
        revision: str,
    ) -> RenderArtifact:
        """
        Render bounded original scan evidence and supplied geometry with explicit frame
        compensation.
        """
        if (
            query.geometry_role not in ("visual", "collision", "both")
            or query.view not in list(RenderView)
            or not np.isfinite(query.display_spacing_m)
            or query.display_spacing_m <= 0
        ):
            raise InvalidEvidence(
                reason="Choose a supported geometry role, camera and positive display spacing."
            )
        if query.image and self.browser_executable is None:
            raise InvalidEvidence(
                reason="PNG rendering needs a configured existing Chromium executable; request HTML only or configure the server browser."
            )
        with np.load(sample.evidence_path, allow_pickle=False) as arrays:
            records = arrays[ComparisonArray.RECORDS]
            distances = (
                arrays[ComparisonArray.DISTANCES] if query.comparison_job else None
            )
        if not len(records):
            raise InvalidEvidence(
                reason="This sample contains no scan evidence; choose a region with returns."
            )
        identifier = uuid4().hex
        directory = (
            self.project.directory
            / AgentFile.DIRECTORY
            / AgentFile.RENDERS
            / identifier
        )
        directory.mkdir(parents=True)
        source = directory / AgentFile.POINTS
        np.savez_compressed(
            source, **{str(field): records[field] for field in PointField}
        )
        derived = RefinementProject.create(
            directory=directory / AgentFile.MODEL,
            scan_path=source,
            snapshot=snapshot,
            model_source=f"Agent render of {self.project.record.model_source}",
        )
        if registration.candidates:
            derived.update_registration(result=registration, role="visual")
        scene = InspectionViewer(
            project=derived,
            display_count=query.maximum_points,
            display_spacing=query.display_spacing_m,
        ).build_scene(candidate_index=candidate)
        if distances is not None:
            indices = derived.scan.sample_indices(
                spacing=query.display_spacing_m, maximum_count=query.maximum_points
            )
            scene.figure.data[0].marker.update(
                color=distances[indices],
                colorscale="RdYlGn_r",
                cmin=0,
                cmax=0.1,
                showscale=True,
                colorbar=dict(title="Distance (m)"),
            )
        frame = np.array(sample.region.query.frame_T_scan)
        lower = np.array(sample.region.query.lower)
        upper = np.array(sample.region.query.upper)
        origin = (lower + upper) / 2
        old_origin = np.array(scene.state.display_origin)
        for trace in scene.figure.data:
            points = np.column_stack((trace.x, trace.y, trace.z)) + old_origin
            points = trimesh.transform_points(points, frame) - origin
            trace.x = points[:, 0]
            trace.y = points[:, 1]
            trace.z = points[:, 2]
            if trace.type == "mesh3d":
                trace.visible = (
                    query.geometry_role == "both"
                    or trace.legendgroup == query.geometry_role
                )
                trace.opacity = 0.25
        camera = {
            "eye": {"x": 1.7, "y": -1.7, "z": 1.2},
            "up": {"x": 0, "y": 0, "z": 1},
            "projection": {"type": "perspective"},
        }
        if query.view == RenderView.TOP:
            camera = {
                "eye": {"x": 0, "y": 0, "z": 2.5},
                "up": {"x": 0, "y": 1, "z": 0},
                "projection": {"type": "orthographic"},
            }
        elif query.view == RenderView.FRONT:
            camera = {
                "eye": {"x": 0, "y": -2.5, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1},
                "projection": {"type": "orthographic"},
            }
        elif query.view == RenderView.SIDE:
            camera = {
                "eye": {"x": 2.5, "y": 0, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1},
                "projection": {"type": "orthographic"},
            }
        ranges = {
            axis: dict(
                range=[lower[index] - origin[index], upper[index] - origin[index]],
                autorange=False,
                title=f"Inspection {axis.upper()} (m)",
            )
            for index, axis in enumerate(("x", "y", "z"))
        }
        scene.figure.update_layout(
            width=self.width,
            height=self.height,
            scene=dict(
                camera=camera,
                xaxis=ranges["x"],
                yaxis=ranges["y"],
                zaxis=ranges["z"],
                aspectmode="data",
            ),
            title=dict(
                text=f"{query.view} · {query.geometry_role} · {sample.resolution} evidence",
                font=dict(size=15),
            ),
        )
        html_path = directory / AgentFile.PAGE
        scene.figure.write_html(
            str(html_path),
            include_plotlyjs=True,
            full_html=True,
            post_script="document.body.style.margin='0';document.body.dataset.renderReady='true';",
        )
        image_path = directory / AgentFile.IMAGE
        if query.image:
            environment = dict(os.environ)
            completed = subprocess.run(
                [
                    self.browser_executable,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--run-all-compositor-stages-before-draw",
                    f"--window-size={self.width},{self.height}",
                    "--hide-scrollbars",
                    "--virtual-time-budget=5000",
                    f"--user-data-dir={directory.absolute()}/browser",
                    f"--screenshot={image_path.absolute()}",
                    html_path.absolute().as_uri(),
                ],
                capture_output=True,
                timeout=self.timeout_seconds,
                env=environment,
                check=False,
            )
            if completed.returncode or not image_path.is_file():
                raise InvalidEvidence(
                    reason="Headless browser rendering failed; the HTML/provenance artifact is retained for inspection."
                )
        artifact = RenderArtifact(
            identifier=identifier,
            region_id=sample.region.identifier,
            revision=revision,
            sample_count=sample.count,
            display_count=len(scene.figure.data[0].x),
            view=query.view,
            html_path=str(html_path.absolute()),
            image_path=str(image_path.absolute()) if query.image else "",
            display_origin=origin.tolist(),
            resolution=sample.resolution,
        )
        (directory / AgentFile.RECORD).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=artifact), indent=2)
        )
        return artifact
