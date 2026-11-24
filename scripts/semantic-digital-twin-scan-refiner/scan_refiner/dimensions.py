"""Constrained dimensions of verified boxes in declared axes, with explicit face anchors."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field, replace
from itertools import permutations, product
from enum import IntEnum, StrEnum
import numpy as np
import trimesh
from scan_refiner.comparison import SurfaceComparison
from scan_refiner.coverage import DistanceSummary
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.frames import CoordinateFrame
from scan_refiner.records import GeometryRole
from scan_refiner.refinement import FitOutcome
from scan_refiner.world import BodyMesh, WorldSnapshot


# %% Box vocabulary
class BoxAxis(IntEnum):
    """Declared box-coordinate directions used by dimension locks."""

    X = 0
    Y = 1
    Z = 2


class DimensionAnchor(StrEnum):
    """Reference held fixed when changing an extent along one declared box axis."""

    CENTER = "center"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class DimensionFrame(StrEnum):
    """Directions in which box extents and face anchors are interpreted."""

    ROOT = "root"
    BOX = "box"


@dataclass(kw_only=True)
class BoxFrame:
    """A verified geometric box frame, independent of an unavailable source-local frame."""

    root_T_box: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())
    """Rigid box-to-model-root placement; columns define the displayed box axes."""

    def __post_init__(self) -> None:
        """Reject nonrigid or nonfinite box-axis placements."""
        if not CoordinateFrame.accepts(matrix=np.array(self.root_T_box)):
            raise InvalidEvidence(
                reason="A box frame must be a finite proper rigid placement."
            )

    @property
    def basis(self) -> np.ndarray:
        """Return box-axis directions expressed in the model root frame."""
        return np.array(self.root_T_box)[:3, :3]

    @property
    def origin(self) -> np.ndarray:
        """Return the box-frame origin in model-root metres."""
        return np.array(self.root_T_box)[:3, 3]

    def to_box(self, *, points: np.ndarray) -> np.ndarray:
        """Express root-frame positions in the declared geometric box frame."""
        return trimesh.transform_points(points, np.linalg.inv(self.root_T_box))

    def to_root(self, *, points: np.ndarray) -> np.ndarray:
        """Express geometric box positions in the model root frame."""
        return trimesh.transform_points(points, np.array(self.root_T_box))


# %% Box eligibility
@dataclass(kw_only=True)
class BoxBounds:
    """Opposing faces of a verified box in its declared coordinate frame, in metres."""

    lower: list[float]
    """Minimum X/Y/Z face positions."""
    upper: list[float]
    """Maximum X/Y/Z face positions."""

    @property
    def dimensions(self) -> np.ndarray:
        """Return the positive X/Y/Z extents independent of the source mesh origin."""
        return np.array(self.upper) - self.lower

    @property
    def center(self) -> np.ndarray:
        """Return the midpoint of each opposing face pair."""
        return (np.array(self.upper) + self.lower) / 2

    @classmethod
    def from_mesh(cls, *, mesh: trimesh.Trimesh) -> BoxBounds:
        """Reject oriented, open or compound geometry rather than fitting its bounding box."""
        inspected = BoxInspection().inspect(mesh=mesh)
        if inspected.bounds is None:
            raise InvalidEvidence(reason=inspected.problem)
        return inspected.bounds


@dataclass(kw_only=True)
class BoxInspectionResult:
    """Eligibility information without converting an unsupported mesh into a box."""

    bounds: BoxBounds | None
    """Actual box faces, absent when eligibility checks fail."""
    problem: str
    """Concrete reason a shape cannot use this dimension fitter."""
    frame: BoxFrame = field(default_factory=BoxFrame)
    """Directions and origin in which the verified bounds are expressed."""


@dataclass(kw_only=True)
class BoxInspection:
    """Verify actual box surfaces in the model root frame."""

    linear_tolerance: float = 1e-6
    """Smallest meaningful positive extent in root-frame metres."""
    relative_tolerance: float = 1e-6
    """Relative agreement required for axis normals, enclosed volume and surface area."""

    def inspect(self, *, mesh: trimesh.Trimesh) -> BoxInspectionResult:
        """Recognize a closed box without changing the supplied geometry."""
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            return BoxInspectionResult(
                bounds=None,
                problem="Dimension fitting requires finite, nonempty box geometry.",
            )
        bounds = BoxBounds(lower=mesh.bounds[0].tolist(), upper=mesh.bounds[1].tolist())
        lengths = bounds.dimensions
        if np.any(lengths <= self.linear_tolerance):
            return BoxInspectionResult(
                bounds=None,
                problem="Dimension fitting requires three positive box extents.",
            )
        checked = mesh.copy()
        checked.merge_vertices()
        if not checked.is_watertight or not checked.is_winding_consistent:
            return BoxInspectionResult(
                bounds=None,
                problem="Dimension fitting currently requires a closed box surface.",
            )
        if not np.allclose(
            np.max(np.abs(checked.face_normals), axis=1),
            1,
            rtol=self.relative_tolerance,
            atol=0,
        ):
            return BoxInspectionResult(
                bounds=None,
                problem="Dimension fitting currently requires a root-aligned box; rotated meshes are unsupported.",
            )
        expected_area = 2 * sum(
            lengths[axis] * lengths[other] for axis, other in [(0, 1), (0, 2), (1, 2)]
        )
        if not np.isclose(
            abs(checked.volume),
            np.prod(lengths),
            rtol=self.relative_tolerance,
            atol=self.linear_tolerance**3,
        ) or not np.isclose(
            checked.area,
            expected_area,
            rtol=self.relative_tolerance,
            atol=self.linear_tolerance**2,
        ):
            return BoxInspectionResult(
                bounds=None,
                problem="This surface is not a single box; its bounding box cannot be used as its shape.",
            )
        return BoxInspectionResult(bounds=bounds, problem="")


@dataclass(kw_only=True)
class OrientedBoxInspection:
    """Verify a cuboid in canonical geometric axes without substituting an enclosing box."""

    normal_tolerance: float = 1e-6
    """Largest orthogonality or off-axis normal component admitted before frame recovery."""
    angle_digits: int = 6
    """Angular precision passed to the existing oriented-bounds search."""
    canonical_digits: int = 12
    """Rounding used solely to resolve equivalent signed-axis permutations consistently."""

    def has_box_faces(self, *, mesh: trimesh.Trimesh) -> bool:
        """Require every triangle normal to follow three orthogonal face directions."""
        first = mesh.face_normals[int(np.argmax(mesh.area_faces))]
        remaining = np.flatnonzero(
            np.abs(mesh.face_normals @ first) < 1 - self.normal_tolerance
        )
        if not len(remaining):
            return False
        second = mesh.face_normals[
            remaining[int(np.argmax(mesh.area_faces[remaining]))]
        ]
        if abs(first @ second) > self.normal_tolerance:
            return False
        third = np.cross(first, second)
        components = np.sort(
            np.abs(mesh.face_normals @ np.column_stack((first, second, third))), axis=1
        )
        return bool(
            np.all(components[:, :2] <= self.normal_tolerance)
            and np.allclose(components[:, 2], 1, atol=self.normal_tolerance, rtol=0)
        )

    def inspect(self, *, mesh: trimesh.Trimesh) -> BoxInspectionResult:
        """Return actual box faces in a proper frame, rejecting skewed or non-box surfaces."""
        root = BoxInspection().inspect(mesh=mesh)
        if root.bounds is not None:
            return root
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            return root
        checked = mesh.copy()
        checked.merge_vertices()
        if not checked.is_watertight or not checked.is_winding_consistent:
            return root
        if not self.has_box_faces(mesh=checked):
            return BoxInspectionResult(
                bounds=None,
                problem="The surface does not have three orthogonal box-face directions; beveled, skewed or compound geometry is unsupported.",
            )
        root_to_bounds, _ = trimesh.bounds.oriented_bounds(
            checked, angle_digits=self.angle_digits
        )
        basis = root_to_bounds[:3, :3].T
        best = np.eye(3)
        best_score = -np.inf
        best_tie = ()
        for permutation in permutations(range(3)):
            for signs in product((-1, 1), repeat=3):
                candidate = basis[:, permutation] * signs
                if np.linalg.det(candidate) < 0:
                    continue
                score = round(float(np.trace(candidate)), self.canonical_digits)
                tie = tuple(np.round(candidate.ravel(), self.canonical_digits))
                if score > best_score or (score == best_score and tie > best_tie):
                    best = candidate
                    best_score = score
                    best_tie = tie
        placement = np.eye(4)
        placement[:3, :3] = best
        placement[:3, 3] = checked.bounds.mean(axis=0)
        frame = BoxFrame(root_T_box=placement.tolist())
        local = checked.copy()
        local.vertices = frame.to_box(points=local.vertices)
        verified = BoxInspection().inspect(mesh=local)
        if verified.bounds is None:
            return BoxInspectionResult(
                bounds=None,
                problem="The surface fails the box volume, area or face checks in its geometric axes.",
                frame=frame,
            )
        return BoxInspectionResult(bounds=verified.bounds, problem="", frame=frame)


@dataclass(kw_only=True)
class BodyDimensionInfo:
    """Current fitting box and eligibility of every present role of the selected body."""

    body_name: str
    """Qualified body identity whose visual/collision relationship must be retained."""
    geometry_role: str
    """Representation whose bounds define the proposed dimension change."""
    bounds: BoxBounds | None
    """Current bounds of the fitting representation when all roles are eligible."""
    problem: str = ""
    """Reason dimensional refinement is unavailable for this body and role."""
    frame: BoxFrame = field(default_factory=BoxFrame)
    """Geometric axes and origin in which the displayed dimensions are measured."""

    @classmethod
    def inspect(
        cls,
        *,
        snapshot: WorldSnapshot,
        body_name: str,
        role: str,
        frame_mode: str = DimensionFrame.ROOT.value,
    ) -> BodyDimensionInfo:
        """Require compatible verified box surfaces in every present geometry role."""
        surfaces = [
            surface for surface in snapshot.meshes if surface.body_name == body_name
        ]
        if not any(surface.role == role for surface in surfaces):
            return cls(
                body_name=body_name,
                geometry_role=role,
                bounds=None,
                problem="The selected body has no geometry in this role.",
            )
        selected_mesh = snapshot.body_mesh(
            body_name=body_name, role=GeometryRole(role)
        ).mesh
        selected = (
            OrientedBoxInspection().inspect(mesh=selected_mesh)
            if frame_mode == DimensionFrame.BOX
            else BoxInspection().inspect(mesh=selected_mesh)
        )
        if selected.bounds is None:
            return cls(
                body_name=body_name,
                geometry_role=role,
                bounds=None,
                problem=f"{role}: {selected.problem}",
                frame=selected.frame,
            )
        for representation in dict.fromkeys(surface.role for surface in surfaces):
            surface = snapshot.body_mesh(
                body_name=body_name, role=GeometryRole(representation)
            )
            local = surface.mesh.copy()
            local.vertices = selected.frame.to_box(points=local.vertices)
            inspected = BoxInspection().inspect(mesh=local)
            if inspected.bounds is None:
                return cls(
                    body_name=body_name,
                    geometry_role=role,
                    bounds=None,
                    problem=f"{representation}: Geometry must be a box with directions compatible with the fitting role.",
                    frame=selected.frame,
                )
        return cls(
            body_name=body_name,
            geometry_role=role,
            bounds=selected.bounds,
            problem="",
            frame=selected.frame,
        )


# %% Effective dimensional override
@dataclass(kw_only=True)
class BodyDimensions:
    """Cumulative positive dimension stretch before the body's separate translation."""

    body_name: str
    """Qualified body identity; both geometry roles receive the same mapping."""
    scale: list[float]
    """Root X/Y/Z multipliers of immutable baseline coordinates."""
    offset: list[float]
    """Root-coordinate offsets applied before any separate translation override."""
    source_job: str = ""
    """Reviewed proposal that last changed the dimensional override."""

    matrix: list[list[float]] = field(default_factory=list)
    """Root-space stretch matrix, or empty for a legacy root-diagonal override."""

    @property
    def linear(self) -> np.ndarray:
        """Return the authoritative root-space stretch without changing legacy scale data."""
        return np.array(self.matrix) if self.matrix else np.diag(self.scale)

    def validate(self) -> None:
        """Reject rotations, reflections, collapsed geometry and invalid stretch mappings."""
        if (
            not self.body_name
            or len(self.scale) != 3
            or len(self.offset) != 3
            or not np.isfinite([self.scale, self.offset]).all()
            or np.any(np.array(self.scale) <= 0)
        ):
            raise InvalidEvidence(
                reason="Dimension overrides require positive finite scales and finite root offsets."
            )
        matrix = self.linear
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T)
            or np.any(np.linalg.eigvalsh(matrix) <= 0)
        ):
            raise InvalidEvidence(
                reason="A dimension mapping must be a positive symmetric stretch, not a rotation, reflection or shear."
            )


# %% Fit records
@dataclass(kw_only=True)
class FaceSupport:
    """Original associated scan records assigned to one opposing box-face pair."""

    axis: str
    """Declared box direction named explicitly so report order carries no hidden meaning."""
    minimum_count: int
    """Records initially associated with the minimum face."""
    maximum_count: int
    """Records initially associated with the maximum face."""


@dataclass(kw_only=True)
class DimensionConstraints:
    """Explicit dimension permissions, anchors and evidence thresholds."""

    axes: list[bool] = field(default_factory=lambda: [False, False, False])
    """Unlocked X/Y/Z extents; locked extents and their face positions stay fixed."""
    anchors: list[str] = field(
        default_factory=lambda: [DimensionAnchor.CENTER.value] * 3
    )
    """Fixed center or fixed minimum/maximum face for each declared box axis."""
    maximum_change: float = 0.05
    """Maximum absolute extent change per unlocked axis, in metres."""
    association_distance: float = 0.1
    """Initial point-to-box gate in metres; associated record identities remain fixed."""
    minimum_points: int = 30
    """Required associated records over the whole fitting box."""
    minimum_face_points: int = 10
    """Required returns on each face used to estimate an unlocked extent."""
    minimum_dimension: float = 0.001
    """Positive extent floor, in metres, for every unlocked dimension."""
    minimum_improvement: float = 0.0001
    """Required decrease in capped mean point-to-triangle distance, in metres."""
    bound_tolerance: float = 0.00001
    """Distance from a requested change limit or positive extent floor treated as saturation."""

    frame_mode: str = DimensionFrame.ROOT.value
    """Root axes or independently verified geometric box axes; never assumed source-local axes."""

    @property
    def problem(self) -> str | None:
        """Explain a dimension request that cannot preserve its declared constraints."""
        if (
            self.frame_mode not in list(DimensionFrame)
            or len(self.axes) != 3
            or any(type(axis) is not bool for axis in self.axes)
            or not any(self.axes)
            or len(self.anchors) != 3
            or any(anchor not in list(DimensionAnchor) for anchor in self.anchors)
        ):
            return (
                "Unlock a box dimension and choose a supported anchor for every axis."
            )
        if (
            not np.isfinite(
                [
                    self.maximum_change,
                    self.association_distance,
                    self.minimum_dimension,
                    self.minimum_improvement,
                    self.bound_tolerance,
                ]
            ).all()
            or min(
                self.maximum_change,
                self.association_distance,
                self.minimum_dimension,
                self.minimum_improvement,
                self.bound_tolerance,
            )
            <= 0
            or self.bound_tolerance >= self.maximum_change
        ):
            return "Use finite positive dimension limits and evidence thresholds."
        if (
            not isinstance(self.minimum_points, int)
            or self.minimum_points < 3
            or not isinstance(self.minimum_face_points, int)
            or self.minimum_face_points < 1
        ):
            return "Declare positive face evidence counts and at least three fitting records."
        return None


@dataclass(kw_only=True)
class DimensionProposal:
    """Reviewable box dimensions over a fixed original record selection."""

    body_name: str
    """Qualified selected body; attached bodies are not resized."""
    before_box: BoxBounds
    """Bounds of the current fitting representation."""
    after_box: BoxBounds
    """Proposed bounds preserving all locked dimensions and declared anchors."""
    scale: list[float]
    """Extent multipliers in the declared box axes; matrix supplies the root mapping when present."""
    offset: list[float]
    """Common root-coordinate offset paired with the scale."""
    constraints: DimensionConstraints
    """Frozen extent locks, face anchors and evidence gates."""
    available_points: int
    """Original crop records before proximity association."""
    matched_points: int
    """Original records admitted by the initial point-to-box gate."""
    face_support: list[FaceSupport]
    """Counts that explain whether the required opposing faces were observed."""
    before: DistanceSummary
    """Current triangle distances on the associated records."""
    after: DistanceSummary
    """Proposed triangle distances on those exact records."""
    outcome: str
    """Applicability or concrete evidence/limit rejection."""

    frame: BoxFrame = field(default_factory=BoxFrame)
    """Geometric axes containing the before/after box bounds and face anchors."""
    matrix: list[list[float]] = field(default_factory=list)
    """Root-space linear mapping, or empty for legacy root-diagonal scales."""

    @property
    def applicable(self) -> bool:
        """Only an improved unsaturated proposal with sufficient required face evidence may apply."""
        return self.outcome == FitOutcome.PROPOSED


@dataclass(kw_only=True)
class DimensionResult:
    """A dimension proposal paired with original record selection and residual arrays."""

    proposal: DimensionProposal
    """Reviewable geometry and evidence diagnostics."""
    selected: np.ndarray
    """Boolean association mask in original crop-record order."""
    before_distances: np.ndarray
    """Current exact triangle distances for selected records."""
    after_distances: np.ndarray
    """Proposed exact triangle distances for those same records."""


# %% Dimension estimation
@dataclass(kw_only=True)
class DimensionFitter:
    """Estimate box faces by medians while preserving declared anchors and axis locks."""

    batch_size: int = 2000
    """Maximum records in one point-to-triangle proximity query."""

    def fit(
        self,
        *,
        body: BodyMesh,
        scan_points: np.ndarray,
        scan_T_root: np.ndarray,
        constraints: DimensionConstraints,
        tolerance: float,
    ) -> DimensionResult:
        """Return a proposal without replacing mesh geometry or interpreting absent returns as faces."""
        if constraints.problem is not None:
            raise InvalidEvidence(reason=constraints.problem)
        if (
            scan_points.ndim != 2
            or scan_points.shape[1] != 3
            or not np.isfinite(scan_points).all()
            or not CoordinateFrame.accepts(matrix=scan_T_root)
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise InvalidEvidence(
                reason="Dimension fitting requires finite XYZ, a rigid placement and positive tolerance."
            )
        if constraints.frame_mode == DimensionFrame.BOX:
            return self.fit_oriented(
                body=body,
                scan_points=scan_points,
                scan_T_root=scan_T_root,
                constraints=constraints,
                tolerance=tolerance,
            )
        box = BoxBounds.from_mesh(mesh=body.mesh)
        if any(
            box.dimensions[axis] <= constraints.minimum_dimension
            for axis in BoxAxis
            if constraints.axes[axis]
        ):
            raise InvalidEvidence(
                reason="The unlocked baseline dimensions must exceed the positive extent floor."
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
        proposal = DimensionProposal(
            body_name=body.body_name,
            before_box=box,
            after_box=BoxBounds(lower=box.lower.copy(), upper=box.upper.copy()),
            scale=[1.0, 1.0, 1.0],
            offset=[0.0, 0.0, 0.0],
            constraints=constraints,
            available_points=len(scan_points),
            matched_points=int(selected.sum()),
            face_support=[],
            before=summary,
            after=summary,
            outcome=FitOutcome.TOO_FEW_POINTS.value,
        )
        result = DimensionResult(
            proposal=proposal,
            selected=selected,
            before_distances=distances,
            after_distances=distances.copy(),
        )
        if selected.sum() < constraints.minimum_points:
            return result
        points = trimesh.transform_points(
            scan_points[selected], np.linalg.inv(scan_T_root)
        )
        closest = nearest.closest_points[selected]
        face_axes = np.argmax(
            np.abs(body.mesh.face_normals[nearest.triangle_indices[selected]]), axis=1
        )
        requested = box.dimensions.copy()
        for axis in BoxAxis:
            minimum = (face_axes == axis) & (
                np.abs(closest[:, axis] - box.lower[axis])
                <= np.abs(closest[:, axis] - box.upper[axis])
            )
            maximum = (face_axes == axis) & ~minimum
            support = FaceSupport(
                axis=axis.name,
                minimum_count=int(minimum.sum()),
                maximum_count=int(maximum.sum()),
            )
            proposal.face_support.append(support)
            if not constraints.axes[axis]:
                continue
            anchor = constraints.anchors[axis]
            if (
                anchor != DimensionAnchor.MINIMUM
                and support.minimum_count < constraints.minimum_face_points
            ) or (
                anchor != DimensionAnchor.MAXIMUM
                and support.maximum_count < constraints.minimum_face_points
            ):
                proposal.outcome = FitOutcome.UNDERCONSTRAINED.value
                return result
            measured_lower = (
                box.lower[axis]
                if anchor == DimensionAnchor.MINIMUM
                else float(np.median(points[minimum, axis]))
            )
            measured_upper = (
                box.upper[axis]
                if anchor == DimensionAnchor.MAXIMUM
                else float(np.median(points[maximum, axis]))
            )
            requested[axis] = measured_upper - measured_lower
        saturated = False
        for axis in BoxAxis:
            if not constraints.axes[axis]:
                continue
            lower_limit = max(
                constraints.minimum_dimension,
                box.dimensions[axis] - constraints.maximum_change,
            )
            upper_limit = box.dimensions[axis] + constraints.maximum_change
            length = float(np.clip(requested[axis], lower_limit, upper_limit))
            saturated |= (
                length <= lower_limit + constraints.bound_tolerance
                or length >= upper_limit - constraints.bound_tolerance
            )
            anchor = constraints.anchors[axis]
            if anchor == DimensionAnchor.MINIMUM:
                proposal.after_box.upper[axis] = box.lower[axis] + length
            elif anchor == DimensionAnchor.MAXIMUM:
                proposal.after_box.lower[axis] = box.upper[axis] - length
            else:
                proposal.after_box.lower[axis] = float(box.center[axis] - length / 2)
                proposal.after_box.upper[axis] = float(box.center[axis] + length / 2)
        proposal.scale = (proposal.after_box.dimensions / box.dimensions).tolist()
        proposal.offset = (
            np.array(proposal.after_box.lower) - np.array(box.lower) * proposal.scale
        ).tolist()
        moved = body.mesh.copy()
        moved.vertices = moved.vertices * np.array(proposal.scale) + proposal.offset
        result.after_distances = query.scan_to_model(
            scan_points=scan_points[selected], root_mesh=moved, scan_T_root=scan_T_root
        )
        proposal.after = DistanceSummary.from_distances(
            distances=result.after_distances, tolerance=tolerance
        )
        improvement = float(
            np.mean(np.minimum(distances, tolerance))
            - np.mean(np.minimum(result.after_distances, tolerance))
        )
        proposal.outcome = (
            FitOutcome.AT_BOUND.value
            if saturated
            else FitOutcome.PROPOSED.value
            if improvement > constraints.minimum_improvement
            else FitOutcome.NO_IMPROVEMENT.value
        )
        return result

    def fit_oriented(
        self,
        *,
        body: BodyMesh,
        scan_points: np.ndarray,
        scan_T_root: np.ndarray,
        constraints: DimensionConstraints,
        tolerance: float,
    ) -> DimensionResult:
        """Fit verified local box faces and lift their deformation into the model root frame."""
        inspected = OrientedBoxInspection().inspect(mesh=body.mesh)
        if inspected.bounds is None:
            raise InvalidEvidence(reason=inspected.problem)
        frame = inspected.frame
        local_mesh = body.mesh.copy()
        local_mesh.vertices = frame.to_box(points=local_mesh.vertices)
        root_points = trimesh.transform_points(scan_points, np.linalg.inv(scan_T_root))
        result = self.fit(
            body=BodyMesh(body_name=body.body_name, role=body.role, mesh=local_mesh),
            scan_points=frame.to_box(points=root_points),
            scan_T_root=np.eye(4),
            constraints=replace(constraints, frame_mode=DimensionFrame.ROOT.value),
            tolerance=tolerance,
        )
        proposal = result.proposal
        linear = frame.basis @ np.diag(proposal.scale) @ frame.basis.T
        proposal.offset = (
            frame.to_root(points=np.array([proposal.offset]))[0] - linear @ frame.origin
        ).tolist()
        proposal.matrix = (
            linear.tolist()
            if not np.allclose(frame.basis, np.eye(3), atol=1e-12, rtol=0)
            else []
        )
        proposal.frame = frame
        proposal.constraints = constraints
        return result
