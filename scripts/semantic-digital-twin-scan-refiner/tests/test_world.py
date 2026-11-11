"""
Verify real URDF imports with distinct origins and geometry representations.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

import numpy as np

from scan_refiner.records import GeometryRole
from scan_refiner.world import WorldSnapshot
from semantic_digital_twin.adapters.urdf import URDFParser


# %% Geometry interpretation
def test_snapshot_preserves_visual_origin_and_collision_size(tmp_path: Path) -> None:
    """
    Body kinematics and shape offsets contribute once to frozen root-frame bounds.
    """
    source = Path(__file__).parent / "dataset" / "offset_box.urdf"
    world = URDFParser.from_file(str(source)).create_world()
    snapshot = WorldSnapshot.from_world(world)
    [visual] = snapshot.select(GeometryRole.VISUAL)
    [collision] = snapshot.select(GeometryRole.COLLISION)
    np.testing.assert_allclose(visual.mesh.bounds, ((0.2, 1.5, 0.1), (2.2, 2.5, 0.5)))
    np.testing.assert_allclose(
        collision.mesh.bounds, ((0.2, 1.6, 0.2), (1.8, 2.4, 0.4))
    )
    path = tmp_path / "world.npz"
    snapshot.save(path)
    restored = WorldSnapshot.read(path)
    assert [item.body_name for item in restored.meshes] == [
        item.body_name for item in snapshot.meshes
    ]
    for original, frozen in zip(snapshot.meshes, restored.meshes, strict=True):
        np.testing.assert_array_equal(original.mesh.vertices, frozen.mesh.vertices)
        np.testing.assert_array_equal(original.mesh.faces, frozen.mesh.faces)
