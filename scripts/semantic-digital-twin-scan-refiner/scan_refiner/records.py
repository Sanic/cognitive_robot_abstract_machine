"""Versioned project records and artifact vocabulary."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from enum import StrEnum

from krrood.patterns.path_enumeration import PathEnumeration


# %% Artifact vocabulary
class ProjectFile(PathEnumeration):
    """Durable artifacts relative to an individual project directory."""

    RECORD = "project.json"
    HISTORY = "history.json"
    JOBS = "jobs"
    SCAN = "scan.npz"
    WORLD = "world.npz"
    CANDIDATES = "candidates.json"
    TRANSFORM = "scan_T_root.npy"
    VIEWER = "index.html"
    SCAN_INDEX = "scan-index"


class ArrayField(StrEnum):
    """Named arrays in portable numeric snapshots."""

    POINTS = "points"
    COLORS = "colors"
    COLOURS = "colours"
    SOURCE_INDICES = "source_indices"
    VERTICES = "vertices"
    FACES = "faces"
    BODY_NAMES = "body_names"
    ASSEMBLIES = "assemblies"
    KINEMATIC_NODES = "kinematic_nodes"
    GEOMETRY_ROLES = "geometry_roles"
    VERTEX_OFFSETS = "vertex_offsets"
    FACE_OFFSETS = "face_offsets"


class GeometryRole(StrEnum):
    """Geometry intent retained independently through inspection."""

    VISUAL = "visual"
    COLLISION = "collision"


# %% Persistent records
@dataclass(kw_only=True)
class ProjectRecord:
    """Input identities and explicit registration acceptance for one baseline."""

    scan_source: str
    """Original preview path, retained for provenance rather than future reads."""
    scan_sha256: str
    """Digest of the imported scan file before coordinate conversion."""
    model_source: str
    """Provider identifier or source URDF path."""
    world_sha256: str
    """Digest of the frozen geometry snapshot."""
    scan_scale: float
    """Multiplier converting source coordinates to metres."""
    scan_count: int
    """Number of original records in this imported preview."""
    schema_version: int = 1
    """Version of the project's persisted record contract."""
    accepted_candidate: int | None = None
    """Reviewed candidate index; absent until explicitly accepted."""
    registration_geometry: str = GeometryRole.VISUAL
    """Geometry representation selected for candidate generation."""


@dataclass(kw_only=True)
class RegistrationCandidate:
    """A proposed root-to-scan transform and sampled-surface diagnostics."""

    scan_T_root: list[list[float]]
    """Homogeneous transform mapping root-frame coordinates into scan coordinates."""
    support_fraction: float
    """Fraction of model samples within the configured support distance."""
    median_distance: float
    """Median model-sample to scan-sample distance in metres."""


@dataclass(kw_only=True)
class RegistrationResult:
    """Candidate transforms and settings retained for reproducible review."""

    candidates: list[RegistrationCandidate] = field(default_factory=list)
    """Alternatives ranked by sampled model support and residual."""
    settings: dict[str, str | float | int] = field(default_factory=dict)
    """Search settings that produced these candidates."""
