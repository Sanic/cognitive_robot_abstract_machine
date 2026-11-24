"""Constrained translation proposals over frozen labelled scan evidence."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import StrEnum
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scan_refiner.comparison import SurfaceComparison
from scan_refiner.coverage import DistanceSummary
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame
from scan_refiner.world import BodyMesh


# %% Proposal vocabulary
class FitOutcome(StrEnum):
    """Applicability of a fitted translation, independent of job completion."""

    PROPOSED = "proposed"
    TOO_FEW_POINTS = "too_few_points"
    UNDERCONSTRAINED = "underconstrained"
    AT_BOUND = "at_bound"
    NO_IMPROVEMENT = "no_improvement"
    NOT_CONVERGED = "not_converged"


# %% Reversible body translation
@dataclass(kw_only=True)
class BodyTranslation:
    """A selected-body displacement relative to the immutable root-frame baseline."""

    body_name: str
    """Qualified body identity; all of its geometry roles move together."""
    translation: list[float]
    """Cumulative XYZ displacement in world-root metres; other bodies stay fixed."""
    source_job: str = ""
    """Reviewed proposal that last changed this override."""

    def validate(self) -> None:
        """Reject incomplete identities and nonfinite displacement vectors."""
        if (
            not self.body_name
            or len(self.translation) != 3
            or not np.isfinite(self.translation).all()
        ):
            raise InvalidEvidence(
                reason="A body override needs a qualified identity and finite XYZ translation."
            )


@dataclass(kw_only=True)
class TranslationConstraints:
    """Explicit movement permissions and evidence gates for a local pose proposal."""

    axes: list[bool] = field(default_factory=lambda: [False, False, False])
    """Unlocked world-root X/Y/Z axes; every disabled component remains exactly zero."""
    maximum_translation: float = 0.05
    """Maximum absolute displacement per unlocked axis, in metres."""
    association_distance: float = 0.1
    """Largest initial scan-to-body distance included in the fixed fitting evidence."""
    robust_distance: float = 0.01
    """Point-to-plane residual scale in metres for the robust least-squares objective."""
    minimum_points: int = 30
    """Required associated original records before fitting is attempted."""
    minimum_normal_ratio: float = 0.001
    """Smallest admitted ratio of weakest to strongest unlocked normal singular value."""
    minimum_improvement: float = 0.0001
    """Required decrease in capped mean triangle distance, in metres."""
    maximum_evaluations: int = 50
    """Maximum bounded optimizer function evaluations."""
    correspondence_iterations: int = 10
    """Maximum surface-correspondence updates over the fixed associated scan records."""
    convergence_distance: float = 0.00001
    """Largest translation update considered numerically converged, in metres."""
    bound_tolerance: float = 0.00001
    """Distance from a translation limit treated as a saturated fit, in metres."""

    @property
    def problem(self) -> str | None:
        """Explain settings that cannot support a bounded translation computation."""
        if (
            len(self.axes) != 3
            or any(type(axis) is not bool for axis in self.axes)
            or not any(self.axes)
        ):
            return "Unlock at least one world X/Y/Z translation axis."
        if (
            not np.isfinite(
                [
                    self.maximum_translation,
                    self.association_distance,
                    self.robust_distance,
                    self.minimum_normal_ratio,
                    self.minimum_improvement,
                    self.bound_tolerance,
                    self.convergence_distance,
                ]
            ).all()
            or min(
                self.maximum_translation,
                self.association_distance,
                self.robust_distance,
                self.minimum_normal_ratio,
                self.minimum_improvement,
                self.bound_tolerance,
                self.convergence_distance,
            )
            <= 0
            or self.minimum_normal_ratio >= 1
            or self.bound_tolerance >= self.maximum_translation
        ):
            return "Use finite positive movement limits and fitting thresholds."
        if (
            not isinstance(self.minimum_points, int)
            or self.minimum_points < 3
            or not isinstance(self.maximum_evaluations, int)
            or not 1 <= self.maximum_evaluations <= 100
            or not isinstance(self.correspondence_iterations, int)
            or not 1 <= self.correspondence_iterations <= 25
        ):
            return "Use at least three matched records and a bounded evaluation count."
        return None


@dataclass(kw_only=True)
class TranslationProposal:
    """A reviewable displacement with metrics over the same initially associated records."""

    body_name: str
    """Qualified body being fitted; attached bodies are not inferred."""
    translation: list[float]
    """Proposed root-frame XYZ displacement relative to the current model revision."""
    constraints: TranslationConstraints
    """Frozen locks, movement bounds and evidence gates."""
    available_points: int
    """Original scan records found inside the requested crop before association."""
    matched_points: int
    """Records admitted by initial point-to-body distance."""
    normal_rank: int
    """Observable unlocked directions in the fixed surface-normal system."""
    before: DistanceSummary
    """Triangle residuals of the associated records before movement."""
    after: DistanceSummary
    """Triangle residuals of those exact records after the proposed movement."""
    outcome: str
    """Proposed or the concrete reason this fit cannot be applied."""

    @property
    def applicable(self) -> bool:
        """Only an improved, unsaturated and constrained solution may become an override."""
        return self.outcome == FitOutcome.PROPOSED


@dataclass(kw_only=True)
class TranslationResult:
    """A proposal paired with its immutable record selection and exact distances."""

    proposal: TranslationProposal
    """Review metrics and frozen constraints."""
    selected: np.ndarray
    """Boolean association mask in original crop-record order."""
    before_distances: np.ndarray
    """Initial triangle distances for the associated records."""
    after_distances: np.ndarray
    """Proposed triangle distances for those same records."""


# %% Numerical translation solution
@dataclass(kw_only=True)
class TranslationSolution:
    """Optimizer displacement and observability before exact residual evaluation."""

    translation: np.ndarray
    """World-root displacement with locked components exactly zero."""
    normal_rank: int
    """Observable directions in the final unlocked surface-normal system."""
    outcome: str
    """Numerical convergence, saturation or observability status."""


# %% Local translation fitting
@dataclass(kw_only=True)
class TranslationFitter:
    """Fit root-frame translation without inferring rotations, shape changes or semantics."""

    batch_size: int = 2000
    """Maximum original records submitted to a triangle proximity query at once."""

    def fit(
        self,
        *,
        body: BodyMesh,
        scan_points: np.ndarray,
        scan_T_root: np.ndarray,
        constraints: TranslationConstraints,
        tolerance: float,
    ) -> TranslationResult:
        """Return a bounded proposal or a rejection without changing geometry or registration."""
        if constraints.problem is not None:
            raise InvalidEvidence(reason=constraints.problem)
        if (
            scan_points.ndim != 2
            or scan_points.shape[1] != 3
            or not np.isfinite(scan_points).all()
            or not np.isfinite(tolerance)
            or tolerance <= 0
            or not CoordinateFrame.accepts(matrix=scan_T_root)
        ):
            raise InvalidEvidence(
                reason="Fitting needs finite XYZ evidence, a rigid placement and positive tolerance."
            )
        query = SurfaceComparison(batch_size=self.batch_size)
        nearest = query.nearest(
            scan_points=scan_points, root_mesh=body.mesh, scan_T_root=scan_T_root
        )
        selected = nearest.distances <= constraints.association_distance
        distances = nearest.distances[selected]
        summary = DistanceSummary.from_distances(
            distances=distances, tolerance=tolerance
        )
        proposal = TranslationProposal(
            body_name=body.body_name,
            translation=[0.0, 0.0, 0.0],
            constraints=constraints,
            available_points=len(scan_points),
            matched_points=int(selected.sum()),
            normal_rank=0,
            before=summary,
            after=summary,
            outcome=FitOutcome.TOO_FEW_POINTS.value,
        )
        result = TranslationResult(
            proposal=proposal,
            selected=selected,
            before_distances=distances,
            after_distances=distances.copy(),
        )
        if len(distances) < constraints.minimum_points:
            return result
        points = trimesh.transform_points(
            scan_points[selected], np.linalg.inv(scan_T_root)
        )
        solution = self.solve(mesh=body.mesh, points=points, constraints=constraints)
        proposal.normal_rank = solution.normal_rank
        proposal.translation = solution.translation.tolist()
        moved = body.mesh.copy()
        moved.apply_translation(solution.translation)
        result.after_distances = query.scan_to_model(
            scan_points=scan_points[selected], root_mesh=moved, scan_T_root=scan_T_root
        )
        proposal.after = DistanceSummary.from_distances(
            distances=result.after_distances, tolerance=tolerance
        )
        if solution.outcome != FitOutcome.PROPOSED:
            proposal.outcome = solution.outcome
            return result
        improvement = float(
            np.mean(np.minimum(distances, tolerance))
            - np.mean(np.minimum(result.after_distances, tolerance))
        )
        proposal.outcome = (
            FitOutcome.PROPOSED.value
            if improvement > constraints.minimum_improvement
            else FitOutcome.NO_IMPROVEMENT.value
        )
        return result

    def solve(
        self,
        *,
        mesh: trimesh.Trimesh,
        points: np.ndarray,
        constraints: TranslationConstraints,
    ) -> TranslationSolution:
        """Update closest surfaces while preserving evidence, axis locks and absolute bounds."""
        axes = np.flatnonzero(constraints.axes)
        delta = np.zeros(3)
        query = SurfaceComparison(batch_size=self.batch_size)
        rank = 0
        for _ in range(constraints.correspondence_iterations):
            moved = mesh.copy()
            moved.apply_translation(delta)
            nearest = query.nearest(
                scan_points=points, root_mesh=moved, scan_T_root=np.eye(4)
            )
            if np.max(nearest.distances) <= constraints.convergence_distance:
                return TranslationSolution(
                    translation=delta,
                    normal_rank=rank,
                    outcome=FitOutcome.PROPOSED.value,
                )
            normals = mesh.face_normals[nearest.triangle_indices]
            system = normals[:, axes]
            singular = np.linalg.svd(system, compute_uv=False)
            rank = (
                int(
                    np.count_nonzero(
                        singular > singular[0] * constraints.minimum_normal_ratio
                    )
                )
                if singular[0] > 0
                else 0
            )
            if rank < len(axes):
                return TranslationSolution(
                    translation=delta,
                    normal_rank=rank,
                    outcome=FitOutcome.UNDERCONSTRAINED.value,
                )
            offsets = (
                np.einsum("ij,ij->i", normals, points - nearest.closest_points)
                + system @ delta[axes]
            )
            fitted = least_squares(
                lambda values: system @ values - offsets,
                delta[axes],
                bounds=(
                    -constraints.maximum_translation,
                    constraints.maximum_translation,
                ),
                loss="soft_l1",
                f_scale=constraints.robust_distance,
                max_nfev=constraints.maximum_evaluations,
            )
            movement = np.linalg.norm(fitted.x - delta[axes])
            delta[axes] = fitted.x
            if not fitted.success:
                return TranslationSolution(
                    translation=delta,
                    normal_rank=rank,
                    outcome=FitOutcome.NOT_CONVERGED.value,
                )
            if np.any(
                np.abs(fitted.x)
                >= constraints.maximum_translation - constraints.bound_tolerance
            ):
                return TranslationSolution(
                    translation=delta,
                    normal_rank=rank,
                    outcome=FitOutcome.AT_BOUND.value,
                )
            if movement <= constraints.convergence_distance:
                return TranslationSolution(
                    translation=delta,
                    normal_rank=rank,
                    outcome=FitOutcome.PROPOSED.value,
                )
        return TranslationSolution(
            translation=delta, normal_rank=rank, outcome=FitOutcome.NOT_CONVERGED.value
        )
