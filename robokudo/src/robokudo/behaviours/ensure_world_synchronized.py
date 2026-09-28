"""
Gate sensor collection on a synchronized served world.
"""

from py_trees.common import Status

from py_trees.behaviour import Behaviour
from robokudo.world_sync import WorldSyncManager


class EnsureWorldSynchronized(Behaviour):
    """
    Make the served robot world available before reading any camera.
    """

    def __init__(self) -> None:
        """
        Create a camera-independent world readiness step.
        """
        super().__init__(name="EnsureWorldSynchronized")

    def update(self) -> Status:
        """
        Fetch the world on first use and retain its update subscription.
        """
        WorldSyncManager.current().ensure_ready()
        return Status.SUCCESS
