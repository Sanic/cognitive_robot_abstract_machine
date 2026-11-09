"""Saved geometric inspection candidates for worlds with unrelated triangle topology."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.evaluation.exceptions import ComparisonGeometryMissingError
from experiments.warsaw.evaluation.size import ObjectSize

# %% metric objects and comparison policy


@dataclass(frozen=True)
class ComparisonObject(JsonRecord):
    """One exported object placed in the ground-truth coordinate frame."""

    identifier: int
    """The object's identifier in its own export manifest."""
    name: str
    """The exported name, retained for human inspection."""
    classes: list[str]
    """The object's own semantic classes, excluded from geometric ranking."""
    parent_identifier: int | None
    """The nearest comparable ancestor, excluding the synthetic world root."""
    bounds: list[list[float]] | None
    """The low and high corners in metric GT coordinates, absent without geometry."""
    size: ObjectSize | None
    """The rigid-transform-invariant dimensions, absent without geometry."""


@dataclass(frozen=True)
class ComparisonSettings(JsonRecord):
    """Geometric inspection policy; costs are not calibrated probabilities."""

    distance_limit: float = 0.5
    """The distance in metres at which the positional cost reaches one."""
    size_weight: float = 0.3
    """The contribution of size difference to the candidate cost."""
    maximum_cost: float = 0.8
    """The largest cost accepted as a provisional association."""
    maximum_candidates: int = 3
    """The number of ranked alternatives retained per predicted object."""


# %% ranked candidates and provisional associations


@dataclass(frozen=True)
class ComparisonCandidate(JsonRecord):
    """One geometric alternative, including the components of its cost."""

    ground_truth_identifier: int
    """The candidate object's export identifier in the GT manifest."""
    distance: float
    """Distance from the prediction's bounding-box centre to the GT box, in metres."""
    size_difference: float
    """The normalized difference between the objects' oriented box dimensions."""
    cost: float
    """The positional cost plus the weighted size difference; smaller is better."""

    @classmethod
    def between(
        cls,
        predicted: ComparisonObject,
        ground_truth: ComparisonObject,
        settings: ComparisonSettings,
    ) -> ComparisonCandidate:
        """Compare geometry already expressed in one frame, without semantic labels."""
        missing = [
            body.name
            for body in (predicted, ground_truth)
            if body.bounds is None or body.size is None
        ]
        if missing:
            raise ComparisonGeometryMissingError(objects=missing)
        centre = np.asarray(predicted.bounds).mean(axis=0)
        low, high = np.asarray(ground_truth.bounds)
        distance = float(
            np.linalg.norm(np.maximum(np.maximum(low - centre, centre - high), 0.0))
        )
        difference = predicted.size.difference_from(ground_truth.size)
        return cls(
            ground_truth_identifier=ground_truth.identifier,
            distance=distance,
            size_difference=difference,
            cost=min(1.0, distance / settings.distance_limit)
            + settings.size_weight * difference,
        )


@dataclass(frozen=True)
class BodyComparison(JsonRecord):
    """A prediction's provisional match, alternatives and observable disagreements."""

    predicted_identifier: int
    """The object's export identifier in the prediction manifest."""
    ground_truth_identifier: int | None
    """The accepted best candidate, absent when none meets the cost limit."""
    candidates: list[ComparisonCandidate]
    """Ranked geometric alternatives, including ones above the acceptance limit."""
    margin: float | None
    """The runner-up's cost minus the best cost, absent without a runner-up."""
    class_disagreement: bool | None
    """Whether assigned classes differ, absent without a match or either class."""
    parent_disagreement: bool | None = None
    """Whether matched parents disagree, absent when that relation cannot be judged."""

    @classmethod
    def compare(
        cls,
        predicted: ComparisonObject,
        ground_truth: list[ComparisonObject],
        settings: ComparisonSettings,
    ) -> BodyComparison:
        """Rank candidates deterministically while retaining tied alternatives."""
        ranked = []
        if predicted.bounds is not None and predicted.size is not None:
            ranked = sorted(
                [
                    ComparisonCandidate.between(predicted, other, settings)
                    for other in ground_truth
                    if other.bounds is not None and other.size is not None
                ],
                key=lambda candidate: (
                    candidate.cost,
                    candidate.ground_truth_identifier,
                ),
            )
        accepted = (
            ranked[0] if ranked and ranked[0].cost <= settings.maximum_cost else None
        )
        other = next(
            (
                other
                for other in ground_truth
                if accepted is not None
                and other.identifier == accepted.ground_truth_identifier
            ),
            None,
        )
        return cls(
            predicted_identifier=predicted.identifier,
            ground_truth_identifier=accepted.ground_truth_identifier
            if accepted is not None
            else None,
            candidates=ranked[: settings.maximum_candidates],
            margin=ranked[1].cost - ranked[0].cost if len(ranked) > 1 else None,
            class_disagreement=(
                not bool(set(predicted.classes) & set(other.classes))
                if other is not None and predicted.classes and other.classes
                else None
            ),
        )


# %% the portable inspection table


@dataclass(frozen=True)
class SpatialComparison(JsonRecord):
    """Provisional many-to-one geometry matches, kept apart from official face scores."""

    settings: ComparisonSettings
    """The policy under which the table was computed."""
    bodies: list[BodyComparison] = field(default_factory=list)
    """Every comparable prediction, including those left unmatched."""
    unmatched_ground_truth: list[int] = field(default_factory=list)
    """GT objects without an assigned prediction; this includes unobserved rooms."""

    @classmethod
    def between(
        cls,
        predicted: list[ComparisonObject],
        ground_truth: list[ComparisonObject],
        settings: ComparisonSettings | None = None,
    ) -> SpatialComparison:
        """Compute geometric matches, then judge classes and parent associations."""
        policy = settings if settings is not None else ComparisonSettings()
        compared = [
            BodyComparison.compare(body, ground_truth, policy) for body in predicted
        ]
        associations = {
            body.predicted_identifier: body.ground_truth_identifier for body in compared
        }
        predicted_by_identifier = {body.identifier: body for body in predicted}
        ground_truth_by_identifier = {body.identifier: body for body in ground_truth}
        judged = []
        for body in compared:
            parent = predicted_by_identifier[
                body.predicted_identifier
            ].parent_identifier
            if body.ground_truth_identifier is None or (
                parent is not None and associations.get(parent) is None
            ):
                judged.append(body)
                continue
            expected = ground_truth_by_identifier[
                body.ground_truth_identifier
            ].parent_identifier
            judged.append(
                replace(body, parent_disagreement=associations.get(parent) != expected)
            )
        assigned = set(associations.values())
        return cls(
            settings=policy,
            bodies=judged,
            unmatched_ground_truth=[
                body.identifier
                for body in ground_truth
                if body.identifier not in assigned
            ],
        )
