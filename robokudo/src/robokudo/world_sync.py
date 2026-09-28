"""
Lifecycle of a world fetched from another ROS process.
"""

from __future__ import annotations

from rclpy.node import Node

import robokudo.world as rk_world
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer


class WorldSyncManager:
    """
    Install one served world and keep its updates alive for this process.
    """

    def __init__(self, node: Node) -> None:
        """
        Borrow the application node without fetching a world yet.
        """
        self.node = node
        """
        Application-owned ROS node used for fetching and updates.
        """
        self._synchronizer: WorldSynchronizer | None = None
        """
        Update subscription retained until application shutdown.
        """

    def bind_to_node(self) -> None:
        """
        Expose this manager through its application-owned ROS node.
        """
        self.node._robokudo_world_sync_manager = self

    @classmethod
    def current(cls) -> WorldSyncManager:
        """
        Return the manager attached to RoboKudo's registered ROS node.
        """
        node = ROSNodeRegistry().get()
        manager = getattr(node, "_robokudo_world_sync_manager", None)
        if not isinstance(manager, cls):
            raise RuntimeError("RoboKudo world sync manager is not initialized")
        return manager

    def ensure_ready(self) -> None:
        """
        Fetch the served model once and subscribe to later world updates.
        """
        if self._synchronizer is not None:
            return
        world = fetch_world_from_service(node=self.node)
        synchronizer = WorldSynchronizer(_world=world, node=self.node)
        rk_world.set_world(world)
        rk_world.init_world_entity_tracker_from_world(world)
        self._synchronizer = synchronizer

    def close(self) -> None:
        """
        Release update transport and detach from the application node.
        """
        if self._synchronizer is not None:
            self._synchronizer.close()
            self._synchronizer = None
        if getattr(self.node, "_robokudo_world_sync_manager", None) is self:
            del self.node._robokudo_world_sync_manager
