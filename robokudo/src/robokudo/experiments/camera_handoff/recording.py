# %% imports

from __future__ import annotations

import csv
from dataclasses import astuple, dataclass, fields
from enum import StrEnum

import cv2
import numpy as np
from PIL import Image
from typing_extensions import TYPE_CHECKING

from krrood.patterns.path_enumeration import PathEnumeration

if TYPE_CHECKING:
    from pathlib import Path

    from robokudo.experiments.camera_handoff.experiment import ComparisonResult
    from robokudo.experiments.camera_handoff.perception import (
        CameraObservation,
        ImageRegion,
        TargetMeasurement,
    )


# %% output names


class ApproachFiles(PathEnumeration):
    """
    Files and directories produced by an approach comparison.
    """

    HEAD_IMAGE = "head.png"
    HEAD_ONLY = "head_only"
    FEEDBACK = "feedback"
    MEASUREMENTS = "measurements.csv"
    SUMMARY = "summary.txt"


class FrameKind(StrEnum):
    """
    Saved image variants for each approach step.
    """

    SENSOR = "raw"
    ANNOTATED = "overlay"


# %% image recording


@dataclass
class ImageRecorder:
    """
    Write unmodified sensor images, annotated frames, and approach measurements.
    """

    output_directory: Path
    """
    Destination for the complete experiment recording.
    """

    search_color: tuple[int, int, int] = (255, 255, 0)
    """
    BGR color of the projected search rectangle.
    """

    target_color: tuple[int, int, int] = (0, 255, 0)
    """
    BGR color of the observed target and centroid.
    """

    desired_color: tuple[int, int, int] = (255, 255, 255)
    """
    BGR color of the desired image alignment.
    """

    centroid_marker_size: int = 9
    """
    Target-centroid cross size in pixels.
    """

    alignment_marker_size: int = 13
    """
    Desired-alignment cross size in pixels.
    """

    line_width: int = 1
    """
    Overlay stroke width in pixels.
    """

    def write_frame(
        self,
        approach: ApproachFiles,
        step: int,
        observation: CameraObservation,
        region: ImageRegion | None,
        measurement: TargetMeasurement | None,
    ) -> None:
        """
        Save a raw frame and a separate overlay with search and target geometry.
        """
        directory = self.output_directory / approach
        directory.mkdir(parents=True, exist_ok=True)
        overlay = observation.color.copy()
        if region is not None:
            cv2.rectangle(
                overlay,
                (region.left, region.top),
                (region.right - 1, region.bottom - 1),
                self.search_color,
                self.line_width,
            )
        if measurement is not None:
            boundary = measurement.region
            cv2.rectangle(
                overlay,
                (boundary.left, boundary.top),
                (boundary.right - 1, boundary.bottom - 1),
                self.target_color,
                self.line_width,
            )
            cv2.drawMarker(
                overlay,
                tuple(np.rint(measurement.center).astype(int)),
                self.target_color,
                cv2.MARKER_CROSS,
                self.centroid_marker_size,
                self.line_width,
            )
        center = tuple(np.rint(observation.intrinsic_matrix[:2, 2]).astype(int))
        cv2.drawMarker(
            overlay,
            center,
            self.desired_color,
            cv2.MARKER_CROSS,
            self.alignment_marker_size,
            self.line_width,
        )
        for kind, image in (
            (FrameKind.SENSOR, observation.color),
            (FrameKind.ANNOTATED, overlay),
        ):
            Image.fromarray(image[:, :, ::-1]).save(
                directory / f"{step:04d}_{kind}.png"
            )

    def write_comparison(
        self, result: ComparisonResult, head: CameraObservation
    ) -> None:
        """
        Save the head image, per-step measurements, and final alignment errors.
        """
        self.output_directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(head.color[:, :, ::-1]).save(
            self.output_directory / ApproachFiles.HEAD_IMAGE
        )
        with (self.output_directory / ApproachFiles.MEASUREMENTS).open(
            "w", newline=""
        ) as stream:
            writer = csv.writer(stream)
            samples = result.head_only.samples + result.feedback.samples
            writer.writerow([attribute.name for attribute in fields(samples[0])])
            for sample in samples:
                writer.writerow(astuple(sample))
        (self.output_directory / ApproachFiles.SUMMARY).write_text(
            f"Head-only final lateral error: {result.head_only.final_lateral_error:.6f} m\n"
            f"Feedback final lateral error: {result.feedback.final_lateral_error:.6f} m\n"
            f"Head-only completed: {result.head_only.completed}\n"
            f"Feedback completed: {result.feedback.completed}\n"
            "Overlay: cyan = projected search region; green = observed target; white = desired alignment.\n"
        )
