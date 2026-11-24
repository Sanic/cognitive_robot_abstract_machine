"""
Constrained pose proposals preserve locks and reject unsupported motion.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
import trimesh
from scan_refiner.refinement import (
    TranslationConstraints,
    TranslationFitter,
    FitOutcome,
)
from scan_refiner.world import BodyMesh
from scan_refiner.records import GeometryRole


# %% Flat evidence
@pytest.fixture
def horizontal_surface() -> BodyMesh:
    """
    A single plane constrains height but cannot constrain in-plane translation.
    """
    mesh = trimesh.Trimesh(
        vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    return BodyMesh(body_name="test/plane", role=GeometryRole.VISUAL, mesh=mesh)


@pytest.fixture
def surface_returns() -> np.ndarray:
    """
    Interior returns avoid edge correspondences in the plane's unconstrained axes.
    """
    x, y = np.meshgrid(np.linspace(-0.8, 0.8, 20), np.linspace(-0.8, 0.8, 20))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.02)))


# %% Locked axes
def test_height_fit_preserves_locked_horizontal_axes(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray
) -> None:
    """
    A height-only proposal recovers the plane offset without horizontal motion.
    """
    constraints = TranslationConstraints(axes=[False, False, True])
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=surface_returns,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.005,
    )
    assert result.proposal.outcome == FitOutcome.PROPOSED
    np.testing.assert_allclose(result.proposal.translation, [0, 0, 0.02], atol=1e-7)
    assert result.proposal.translation[:2] == [0, 0]
    assert result.proposal.before.within_tolerance == 0
    assert result.proposal.after.within_tolerance == 1
    assert result.proposal.matched_points == len(surface_returns)


def test_plane_cannot_support_three_unlocked_axes(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray
) -> None:
    """
    The fitter rejects underdetermined motion instead of inventing lateral offsets.
    """
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=surface_returns,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(axes=[True, True, True]),
        tolerance=0.005,
    )
    assert result.proposal.outcome == FitOutcome.UNDERCONSTRAINED
    assert result.proposal.translation == [0, 0, 0]
    assert not result.proposal.applicable


# %% Bounds and evidence gates
def test_fit_at_motion_bound_cannot_be_applied(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray
) -> None:
    """
    A clipped solution is retained for inspection but cannot become an override.
    """
    constraints = TranslationConstraints(
        axes=[False, False, True], maximum_translation=0.01
    )
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=surface_returns,
        scan_T_root=np.eye(4),
        constraints=constraints,
        tolerance=0.005,
    )
    assert result.proposal.outcome == FitOutcome.AT_BOUND
    assert not result.proposal.applicable


def test_distant_returns_are_excluded_from_fit(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray
) -> None:
    """
    Proximity association is frozen before optimization and retains source indices.
    """
    points = np.vstack((surface_returns, [[10, 10, 10]]))
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(axes=[False, False, True]),
        tolerance=0.005,
    )
    assert result.proposal.available_points == len(points)
    np.testing.assert_array_equal(
        np.flatnonzero(result.selected), np.arange(len(surface_returns))
    )
    assert result.proposal.matched_points == len(surface_returns)


def test_already_aligned_surface_has_no_applicable_proposal(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray
) -> None:
    """
    An unchanged fit must not create a model revision just because optimization ran.
    """
    points = surface_returns.copy()
    points[:, 2] = 0
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(axes=[False, False, True]),
        tolerance=0.005,
    )
    assert result.proposal.outcome == FitOutcome.NO_IMPROVEMENT
    assert not result.proposal.applicable


# %% Bounded proximity
def test_fitting_batches_every_triangle_query(
    horizontal_surface: BodyMesh,
    surface_returns: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Fitting cannot bypass the configured triangle-query memory bound.
    """
    original = trimesh.proximity.closest_point
    sizes = []

    def measure(
        mesh: trimesh.Trimesh, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track the external proximity call without changing its geometry results.
        """
        sizes.append(len(points))
        return original(mesh, points)

    monkeypatch.setattr(trimesh.proximity, "closest_point", measure)
    fitter = TranslationFitter(batch_size=11)
    fitter.fit(
        body=horizontal_surface,
        scan_points=surface_returns,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(axes=[False, False, True]),
        tolerance=0.005,
    )
    assert max(sizes) <= fitter.batch_size


# %% Optimizer completion
@pytest.mark.parametrize("evaluations", [2, 3])
def test_unfinished_optimizer_cannot_produce_applicable_proposal(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray, evaluations: int
) -> None:
    """
    An intermediate improvement is not an accepted numerical solution.
    """
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=surface_returns,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(
            axes=[False, False, True], maximum_evaluations=evaluations
        ),
        tolerance=0.005,
    )
    assert not result.proposal.applicable


def test_partial_optimizer_improvement_does_not_allow_application(
    horizontal_surface: BodyMesh, surface_returns: np.ndarray
) -> None:
    """
    A large metric threshold must not hide the optimizer's unfinished state.
    """
    result = TranslationFitter().fit(
        body=horizontal_surface,
        scan_points=surface_returns,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(
            axes=[False, False, True], maximum_evaluations=2
        ),
        tolerance=0.05,
    )
    assert not result.proposal.applicable


# %% Correspondence updates
def test_thin_box_height_fit_updates_surface_correspondences() -> None:
    """
    Inner-face ambiguity in an initial thin-box offset must not freeze a biased
    solution.
    """
    measured = trimesh.creation.box(extents=[0.8, 0.5, 0.04])
    points, _ = trimesh.sample.sample_surface(measured, 2000, seed=42)
    rough = measured.copy()
    rough.apply_translation([0, 0, 0.02])
    result = TranslationFitter().fit(
        body=BodyMesh(body_name="box", role=GeometryRole.VISUAL, mesh=rough),
        scan_points=points,
        scan_T_root=np.eye(4),
        constraints=TranslationConstraints(axes=[False, False, True]),
        tolerance=0.005,
    )
    assert result.proposal.applicable
    np.testing.assert_allclose(result.proposal.translation, [0, 0, -0.02], atol=1e-6)
