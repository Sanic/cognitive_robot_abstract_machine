"""Directional distance summaries over labelled surfaces and bounded evidence."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import StrEnum
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scan_refiner.comparison import (
    SurfaceComparison,
    ComparisonProgress,
    ComparisonPhase,
)
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import OrientedRegion
from scan_refiner.world import BodyMesh


# %% Artifact vocabulary
class ComparisonArray(StrEnum):
    """Numeric arrays paired with an immutable comparison report."""

    RECORDS = "records"
    DISTANCES = "distances"
    PROPOSED_DISTANCES = "proposed_distances"
    SCAN_BODY_INDICES = "scan_body_indices"
    MODEL_POINTS = "model_points"
    MODEL_DISTANCES = "model_distances"
    MODEL_BODY_INDICES = "model_body_indices"


# %% Directional metrics
@dataclass(kw_only=True)
class DistanceSummary:
    """Unsigned distances for one direction; an empty sample has unknown metrics."""

    sample_count: int
    """Number of evaluated points, independent of the distance threshold."""
    median_distance: float | None
    """Median distance in metres, absent when no points were evaluated."""
    percentile90_distance: float | None
    """Ninetieth percentile distance in metres, absent for an empty sample."""
    within_tolerance: float | None
    """Fraction within the declared tolerance, absent for an empty sample."""

    @classmethod
    def from_distances(
        cls, *, distances: np.ndarray, tolerance: float
    ) -> DistanceSummary:
        """Summarize finite unsigned distances without turning missing data into zero."""
        if not len(distances):
            return cls(
                sample_count=0,
                median_distance=None,
                percentile90_distance=None,
                within_tolerance=None,
            )
        return cls(
            sample_count=len(distances),
            median_distance=float(np.median(distances)),
            percentile90_distance=float(np.quantile(distances, 0.9)),
            within_tolerance=float(np.mean(distances <= tolerance)),
        )


@dataclass(kw_only=True)
class PairedDistanceSummary:
    """Scan-to-surface metrics for two states evaluated on identical evidence."""

    original: DistanceSummary
    """Distances to the current geometry."""
    proposed: DistanceSummary
    """Distances after the explicit unapplied geometry changes."""
    tolerance: float
    """Shared distance threshold in metres."""


@dataclass(kw_only=True)
class BodyComparison:
    """Residual and evidence support for one qualified model body."""

    body_name: str
    """Identity retained from the frozen world snapshot."""
    scan: DistanceSummary
    """Scan records whose nearest triangle belongs to this body."""
    model: DistanceSummary
    """Area-distributed model samples inside the requested crop, queried against scan records."""
    evaluated_area: float
    """Estimated square metres represented by the in-crop surface samples."""
    supported_area: float
    """Estimated square metres with scan evidence within tolerance."""


@dataclass(kw_only=True)
class ComparisonContext:
    """Frozen placement and evidence identity accompanying a downloadable report."""

    job_identifier: str
    """Analysis directory identity pairing this report with its numeric arrays."""
    baseline_signature: str
    """Fingerprint of world, scan index and registration inputs."""
    world_sha256: str
    """Frozen model snapshot content identity."""
    scan_sha256: str
    """Imported preview content identity."""
    geometry_role: str
    """Visual or collision representation used for both directions."""
    scan_T_root: list[list[float]]
    """Frozen root-to-scan placement for reproducing all metric queries."""
    frame_T_scan: list[list[float]]
    """Scan-to-crop frame for interpreting the regional bounds."""
    lower: list[float]
    """Crop lower bound, or empty for unrestricted evidence."""
    upper: list[float]
    """Crop upper bound in the same inspection frame."""
    requested_scan_points: int
    """Maximum evidence-record budget before regional availability is known."""
    requested_model_points: int
    """Surface-sample budget before filtering samples against the crop."""


@dataclass(kw_only=True)
class ComparisonReport:
    """Bidirectional evidence support without visibility or occupancy claims."""

    tolerance: float
    """Frozen distance threshold in metres, shared by both directions."""
    scan: DistanceSummary
    """Residuals from all selected original scan records to the model."""
    model_sample_count: int
    """Number of model samples retained inside the requested crop."""
    supported_area_fraction: float | None
    """Area-weighted nearby-evidence fraction, absent if the crop retained no samples."""
    bodies: list[BodyComparison]
    """Per-body metrics in the same order as the numeric body indices."""
    context: ComparisonContext | None = None
    """Input provenance when the comparison was published by a project job."""
    schema_version: int = 1
    """Version of the report's interpretation and serialized fields."""


# %% Labelled geometry
@dataclass(kw_only=True)
class ModelSamples:
    """Area-distributed samples expressed in original scan coordinates."""

    points: np.ndarray
    """XYZ positions after applying the frozen root-to-scan placement."""
    body_indices: np.ndarray
    """Qualified-body indices matching the report order."""
    area_weights: np.ndarray
    """Square metres represented by each sample before crop filtering."""


@dataclass(kw_only=True)
class LabelledSurfaces:
    """A triangle accelerator with explicit qualified-body ownership."""

    surfaces: list[BodyMesh]
    """Requested geometry role only, expressed in root coordinates."""
    body_names: list[str] = field(init=False)
    """Stable unique identities matching the numeric labels."""
    meshes: list[trimesh.Trimesh] = field(init=False)
    """One concatenated surface per qualified body."""
    mesh: trimesh.Trimesh = field(init=False)
    """Combined triangles preserving the ownership array's ordering."""
    face_body_indices: np.ndarray = field(init=False)
    """Qualified-body index for each combined triangle."""

    def __post_init__(self) -> None:
        """Reject empty geometry and retain ownership across concatenation."""
        if not self.surfaces or any(
            not len(surface.mesh.faces) or surface.mesh.area <= 0
            for surface in self.surfaces
        ):
            raise InvalidEvidence(
                reason="Comparison requires nonempty surfaces with positive area."
            )
        self.body_names = list(
            dict.fromkeys(surface.body_name for surface in self.surfaces)
        )
        self.meshes = [
            trimesh.util.concatenate(
                [surface.mesh for surface in self.surfaces if surface.body_name == body]
            )
            for body in self.body_names
        ]
        self.mesh = trimesh.util.concatenate(self.meshes)
        self.face_body_indices = np.concatenate(
            [
                np.full(len(mesh.faces), index, dtype=np.int32)
                for index, mesh in enumerate(self.meshes)
            ]
        )

    def sample(
        self,
        *,
        count: int,
        seed: int,
        scan_T_root: np.ndarray,
        region: OrientedRegion | None,
    ) -> ModelSamples:
        """Distribute a bounded budget across bodies and retain only in-crop samples."""
        if count < len(self.meshes):
            raise InvalidEvidence(
                reason="Increase the model sample budget or select fewer bodies."
            )
        points, body_indices, weights = [], [], []
        for index, mesh in enumerate(self.meshes):
            allocated = count // len(self.meshes) + (index < count % len(self.meshes))
            samples, _ = trimesh.sample.sample_surface(
                mesh, allocated, seed=seed + index
            )
            points.append(samples)
            body_indices.append(np.full(allocated, index, dtype=np.int32))
            weights.append(np.full(allocated, mesh.area / allocated))
        placed = trimesh.transform_points(np.concatenate(points), scan_T_root)
        keep = (
            np.ones(len(placed), dtype=bool)
            if region is None
            else region.contains(scan_points=placed)
        )
        return ModelSamples(
            points=placed[keep],
            body_indices=np.concatenate(body_indices)[keep],
            area_weights=np.concatenate(weights)[keep],
        )


# %% Derived comparison
@dataclass(kw_only=True)
class EvidenceResult:
    """Numeric evidence paired with a serializable interpretation of its labels."""

    report: ComparisonReport
    """Metrics for this exact sample, placement, scope and threshold."""
    scan_distances: np.ndarray
    """Point-to-triangle distances matching original scan-record order."""
    scan_body_indices: np.ndarray
    """Nearest triangle ownership for each original record."""
    model_points: np.ndarray
    """Evaluated model samples in original scan coordinates."""
    model_distances: np.ndarray
    """Model-sample distances to the evaluated scan records."""
    model_body_indices: np.ndarray
    """Body ownership for each evaluated model sample."""


@dataclass(kw_only=True)
class EvidenceComparison:
    """Compare both directions while keeping absence of evidence distinct from errors."""

    model_sample_count: int = 20000
    """Total surface-sampling budget before crop filtering, shared across bodies."""
    sample_seed: int = 0
    """Reproducible surface sampling independent of process-global randomness."""
    batch_size: int = 2000
    """Maximum scan records queried against model triangles in one batch."""

    progress: ComparisonProgress | None = None
    """Optional phase observer; numeric results do not depend on its presence."""

    def compare(
        self,
        *,
        scan_points: np.ndarray,
        surfaces: list[BodyMesh],
        scan_T_root: np.ndarray,
        tolerance: float,
        region: OrientedRegion | None = None,
    ) -> EvidenceResult:
        """Return directional support for finite evidence and the frozen crop."""
        if (
            scan_points.ndim != 2
            or scan_points.shape[1] != 3
            or not len(scan_points)
            or not np.isfinite(scan_points).all()
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise InvalidEvidence(
                reason="Comparison requires finite XYZ evidence and a positive tolerance."
            )
        model = LabelledSurfaces(surfaces=surfaces)
        nearest = SurfaceComparison(batch_size=self.batch_size).nearest(
            scan_points=scan_points,
            root_mesh=model.mesh,
            scan_T_root=scan_T_root,
            progress=self.progress,
        )
        return self.summarize(
            scan_points=scan_points,
            model=model,
            scan_T_root=scan_T_root,
            tolerance=tolerance,
            region=region,
            scan_distances=nearest.distances,
            scan_body_indices=model.face_body_indices[nearest.triangle_indices],
        )

    def summarize(
        self,
        *,
        scan_points: np.ndarray,
        model: LabelledSurfaces,
        scan_T_root: np.ndarray,
        tolerance: float,
        region: OrientedRegion | None,
        scan_distances: np.ndarray,
        scan_body_indices: np.ndarray,
    ) -> EvidenceResult:
        """Pair labelled residuals with reproducible reverse-direction evidence support."""
        if self.progress is not None:
            self.progress.advance(
                phase=ComparisonPhase.MODEL_SUPPORT, completed=0, total=0
            )
        sampled = model.sample(
            count=self.model_sample_count,
            seed=self.sample_seed,
            scan_T_root=scan_T_root,
            region=region,
        )
        model_distances, _ = cKDTree(scan_points).query(sampled.points, workers=1)
        bodies = []
        for index, body_name in enumerate(model.body_names):
            selected = sampled.body_indices == index
            weight = float(sampled.area_weights[selected][0]) if selected.any() else 0.0
            bodies.append(
                BodyComparison(
                    body_name=body_name,
                    scan=DistanceSummary.from_distances(
                        distances=scan_distances[scan_body_indices == index],
                        tolerance=tolerance,
                    ),
                    model=DistanceSummary.from_distances(
                        distances=model_distances[selected], tolerance=tolerance
                    ),
                    evaluated_area=float(selected.sum()) * weight,
                    supported_area=float(
                        np.count_nonzero(selected & (model_distances <= tolerance))
                    )
                    * weight,
                )
            )
        area = sum(body.evaluated_area for body in bodies)
        report = ComparisonReport(
            tolerance=tolerance,
            scan=DistanceSummary.from_distances(
                distances=scan_distances, tolerance=tolerance
            ),
            model_sample_count=len(sampled.points),
            supported_area_fraction=None
            if area == 0
            else sum(body.supported_area for body in bodies) / area,
            bodies=bodies,
        )
        return EvidenceResult(
            report=report,
            scan_distances=scan_distances,
            scan_body_indices=scan_body_indices,
            model_points=sampled.points,
            model_distances=model_distances,
            model_body_indices=sampled.body_indices,
        )
