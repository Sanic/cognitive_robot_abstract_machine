"""
Exercise automatic registration without apartment-specific names or priors.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
import trimesh

from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.registration import RegistrationSearch


# %% Registration behaviour
def test_automatic_registration_recovers_unknown_upright_pose(
    asymmetric_points: np.ndarray,
) -> None:
    """
    Search the complete yaw range and recover translations in all three axes.
    """
    scan_T_root = trimesh.transformations.rotation_matrix(np.radians(74), (0, 0, 1))
    scan_T_root[:3, 3] = (8.2, -4.7, 1.3)
    scan = trimesh.transform_points(asymmetric_points, scan_T_root)
    result = RegistrationSearch(yaw_step_degrees=10).search(
        model_points=asymmetric_points, scan_points=scan
    )
    np.testing.assert_allclose(result[0].scan_T_root, scan_T_root, atol=0.015)
    assert result[0].support_fraction == 1.0


def test_registration_retains_separated_alternatives(
    asymmetric_points: np.ndarray,
) -> None:
    """
    Keep multiple hypotheses available for review in repetitive environments.
    """
    result = RegistrationSearch(candidate_count=3).search(
        model_points=asymmetric_points, scan_points=asymmetric_points
    )
    assert len(result) == 3
    assert result[0].support_fraction >= result[-1].support_fraction


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.array([[np.nan, 0, 0]])])
def test_registration_rejects_invalid_evidence(points: np.ndarray) -> None:
    """
    Invalid observations fail before constructing spatial search structures.
    """
    with pytest.raises(InvalidEvidence):
        RegistrationSearch().search(model_points=points, scan_points=points)
