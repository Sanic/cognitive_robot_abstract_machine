"""
Isolated bounded computation over frozen scan-refinement evidence.
"""

from __future__ import annotations

# %% Imports
import json
from scan_refiner.process_memory import ProcessMemory
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.dimensions import DimensionFitter
from scan_refiner.comparison import (
    ComparisonProgress,
    ComparisonPhase,
    SurfaceComparison,
)
from scan_refiner.coverage import (
    ComparisonContext,
    ComparisonArray,
    EvidenceComparison,
    DistanceSummary,
    LabelledSurfaces,
    PairedDistanceSummary,
)
from scan_refiner.refinement import TranslationFitter
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame, OrientedRegion
from scan_refiner.jobs import JobFile, JobKind, JobRecord, JobStatus
from scan_refiner.point_records import PointBatch, PointField
from scan_refiner.world import WorldSnapshot
from scan_refiner.project import RefinementProject
from scan_refiner.records import GeometryRole, ProjectFile, RegistrationResult
from scan_refiner.scan_sources import PreviewPointSource
from scan_refiner.spatial_index import IndexFile, RecordSample, SpatialScanIndex


# %% Persisted comparison progress
@dataclass(kw_only=True)
class JobComparisonProgress(ComparisonProgress):
    """
    Persist numeric progress without changing the job's request or project state.
    """

    record: JobRecord
    """
    Computation whose measured phase counters are updated.
    """

    directory: Path
    """
    Atomic progress artifact destination.
    """

    def advance(self, *, phase: ComparisonPhase, completed: int, total: int) -> None:
        self.record.message = phase.value
        self.record.completed = completed
        self.record.total = total
        self.record.save(directory=self.directory)


# %% Worker operations
@dataclass(kw_only=True)
class JobWorker:
    """
    Compute derived results without publishing project decisions.
    """

    project: RefinementProject
    """
    Frozen inputs and job storage owned by the serving application.
    """

    directory: Path
    """
    Directory containing this request, progress and result files.
    """

    maximum_memory: int = ProcessMemory().memory_limit_bytes
    """
    Address-space ceiling in bytes for this worker process.
    """

    comparison_batch: int = 2000
    """
    Maximum points processed between progress publications.
    """

    def run(self) -> None:
        """
        Execute one request with atomic progress and bounded process memory.
        """
        ProcessMemory(memory_limit_bytes=self.maximum_memory).limit_address_space()
        record = JobRecord.read(directory=self.directory)
        record.request.validate()
        record.status = JobStatus.RUNNING.value
        record.message = "Preparing frozen evidence"
        record.save(directory=self.directory)
        if record.request.kind == JobKind.REGISTRATION:
            self.register(record=record)
        elif record.request.kind in (JobKind.REFINEMENT, JobKind.DIMENSIONS):
            self.refine(record=record)
        elif record.request.kind == JobKind.CHANGE_SET_HEATMAP:
            self.compare_change_set(record=record)
        else:
            self.compare(record=record)
        record.status = JobStatus.SUCCEEDED.value
        record.message = "Result ready for review"
        record.save(directory=self.directory)

    def load_geometry(self, *, record: JobRecord) -> WorldSnapshot:
        """
        Restore the submitted model revision even if project decisions change mid-job.
        """
        snapshot = self.project.snapshot
        snapshot.apply_dimensions(overrides=record.dimension_overrides)
        snapshot.apply_translations(overrides=record.overrides)
        return snapshot

    def register(self, *, record: JobRecord) -> None:
        """
        Save upright proposals independently of accepted project registration.
        """
        search = record.request.search
        scan = self.project.scan
        indices = scan.sample_indices(
            spacing=search.cell_width / 3, maximum_count=20000
        )
        model_points = self.load_geometry(record=record).samples(
            role=GeometryRole(record.request.geometry_role)
        )
        record.message = (
            "Searching upright yaw and translation; generation is indeterminate"
        )
        record.save(directory=self.directory)
        candidates = search.search(
            model_points=model_points, scan_points=scan.points[indices]
        )
        result = RegistrationResult(
            candidates=candidates,
            settings=DataclassJSONSerializer.to_json(instance=search),
        )
        (self.directory / JobFile.CANDIDATES).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=result), indent=2)
        )
        record.completed = record.total = len(candidates)

    def evidence(self, *, record: JobRecord) -> PointBatch:
        """
        Select original records with the request's frozen region and point budget.
        """
        request = record.request
        region = None
        if request.lower:
            region = OrientedRegion(
                lower=np.array(request.lower),
                upper=np.array(request.upper),
                frame=CoordinateFrame(frame_T_scan=np.array(request.frame_T_scan)),
            )
        index = SpatialScanIndex(
            directory=self.project.directory / ProjectFile.SCAN_INDEX
        )
        if region is not None and (index.directory / IndexFile.RECORD).exists():
            return region.sample(index=index, maximum_count=request.maximum_count)
        sampled = RecordSample(maximum_count=request.maximum_count)
        for batch in PreviewPointSource(preview=self.project.scan).batches():
            records = batch.records
            if region is not None:
                records = records[
                    region.contains(scan_points=records[PointField.POINTS])
                ]
            sampled.add(records=records)
        return PointBatch(records=sampled.records)

    def compare(self, *, record: JobRecord) -> None:
        """
        Persist distances paired with the exact records used to compute them.
        """
        evidence = self.evidence(record=record)
        points = evidence.records[PointField.POINTS]
        if not len(points):
            raise InvalidEvidence(
                reason="The selected region contains no scan evidence."
            )
        surfaces = self.load_geometry(record=record).select(
            role=GeometryRole(record.request.geometry_role)
        )
        if record.request.body_name:
            surfaces = [
                surface
                for surface in surfaces
                if surface.body_name == record.request.body_name
            ]
        region = None
        if record.request.lower:
            region = OrientedRegion(
                lower=np.array(record.request.lower),
                upper=np.array(record.request.upper),
                frame=CoordinateFrame(
                    frame_T_scan=np.array(record.request.frame_T_scan)
                ),
            )
        result = EvidenceComparison(
            model_sample_count=record.request.model_sample_count,
            batch_size=self.comparison_batch,
            progress=JobComparisonProgress(record=record, directory=self.directory),
        ).compare(
            scan_points=points,
            surfaces=surfaces,
            scan_T_root=np.array(record.scan_T_root),
            tolerance=record.request.tolerance,
            region=region,
        )
        result.report.context = ComparisonContext(
            job_identifier=record.identifier,
            baseline_signature=record.baseline_signature,
            world_sha256=self.project.record.world_sha256,
            scan_sha256=self.project.record.scan_sha256,
            geometry_role=record.request.geometry_role,
            scan_T_root=record.scan_T_root,
            frame_T_scan=record.request.frame_T_scan,
            lower=record.request.lower,
            upper=record.request.upper,
            requested_scan_points=record.request.maximum_count,
            requested_model_points=record.request.model_sample_count,
        )
        destination = self.directory / JobFile.RESULT
        temporary = destination.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **{
                    ComparisonArray.RECORDS: evidence.records,
                    ComparisonArray.DISTANCES: result.scan_distances,
                    ComparisonArray.SCAN_BODY_INDICES: result.scan_body_indices,
                    ComparisonArray.MODEL_POINTS: result.model_points,
                    ComparisonArray.MODEL_DISTANCES: result.model_distances,
                    ComparisonArray.MODEL_BODY_INDICES: result.model_body_indices,
                },
            )
        temporary.replace(destination)
        destination = self.directory / JobFile.REPORT
        temporary = destination.with_suffix(".partial")
        temporary.write_text(
            json.dumps(
                DataclassJSONSerializer.to_json(instance=result.report), indent=2
            )
        )
        temporary.replace(destination)
        record.completed = record.total = len(points)
        record.median_distance = result.report.scan.median_distance
        record.percentile90_distance = result.report.scan.percentile90_distance
        record.within_tolerance = result.report.scan.within_tolerance

    def compare_change_set(self, *, record: JobRecord) -> None:
        """
        Persist one sampled scan cloud with distances for both geometry states.
        """
        evidence = self.evidence(record=record)
        points = evidence.records[PointField.POINTS]
        if not len(points):
            raise InvalidEvidence(reason="The linked crop contains no scan evidence.")
        snapshot = self.load_geometry(record=record)
        transform = np.array(record.scan_T_root)
        role = GeometryRole(record.request.geometry_role)
        comparison = SurfaceComparison(batch_size=self.comparison_batch)
        record.message = "Measuring the original geometry"
        record.save(directory=self.directory)
        original = comparison.scan_to_model(
            scan_points=points,
            root_mesh=LabelledSurfaces(surfaces=snapshot.select(role=role)).mesh,
            scan_T_root=transform,
        )
        snapshot.apply_translations(overrides=record.request.comparison_edits)
        record.message = "Measuring the proposed geometry"
        record.save(directory=self.directory)
        proposed = comparison.scan_to_model(
            scan_points=points,
            root_mesh=LabelledSurfaces(surfaces=snapshot.select(role=role)).mesh,
            scan_T_root=transform,
        )
        tolerance = record.request.tolerance
        summary = PairedDistanceSummary(
            original=DistanceSummary.from_distances(
                distances=original, tolerance=tolerance
            ),
            proposed=DistanceSummary.from_distances(
                distances=proposed, tolerance=tolerance
            ),
            tolerance=tolerance,
        )
        np.savez_compressed(
            self.directory / JobFile.RESULT,
            **{
                ComparisonArray.RECORDS: evidence.records,
                ComparisonArray.DISTANCES: original,
                ComparisonArray.PROPOSED_DISTANCES: proposed,
            },
        )
        (self.directory / JobFile.REPORT).write_text(
            json.dumps(DataclassJSONSerializer.to_json(instance=summary))
        )
        record.completed = record.total = len(points)

    def refine(self, *, record: JobRecord) -> None:
        """
        Save a bounded selected-body proposal separately from project decisions.
        """
        evidence = self.evidence(record=record)
        surface = self.load_geometry(record=record).body_mesh(
            body_name=record.request.body_name,
            role=GeometryRole(record.request.geometry_role),
        )
        record.message = (
            "Estimating supported box dimensions"
            if record.request.kind == JobKind.DIMENSIONS
            else "Fitting unlocked translation axes against associated scan records"
        )
        record.save(directory=self.directory)
        if record.request.kind == JobKind.DIMENSIONS:
            result = DimensionFitter(batch_size=self.comparison_batch).fit(
                body=surface,
                scan_points=evidence.records[PointField.POINTS],
                scan_T_root=np.array(record.scan_T_root),
                constraints=record.request.dimension_constraints,
                tolerance=record.request.tolerance,
            )
        else:
            result = TranslationFitter(batch_size=self.comparison_batch).fit(
                body=surface,
                scan_points=evidence.records[PointField.POINTS],
                scan_T_root=np.array(record.scan_T_root),
                constraints=record.request.constraints,
                tolerance=record.request.tolerance,
            )
        destination = self.directory / JobFile.RESULT
        temporary = destination.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **{
                    ComparisonArray.RECORDS: evidence.records[result.selected],
                    ComparisonArray.DISTANCES: result.after_distances,
                },
            )
        temporary.replace(destination)
        destination = self.directory / JobFile.PROPOSAL
        temporary = destination.with_suffix(".partial")
        temporary.write_text(
            json.dumps(
                DataclassJSONSerializer.to_json(instance=result.proposal), indent=2
            )
        )
        temporary.replace(destination)
        record.completed = record.total = result.proposal.matched_points
        record.median_distance = result.proposal.after.median_distance
        record.percentile90_distance = result.proposal.after.percentile90_distance
        record.within_tolerance = result.proposal.after.within_tolerance
