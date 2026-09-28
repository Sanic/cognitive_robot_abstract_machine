"""
Served-world synchronization independent of camera input.
"""

from unittest.mock import Mock

from py_trees.behaviour import Behaviour
from py_trees.common import Status

import robokudo.world as rk_world
import robokudo.world_sync as world_sync_module
from robokudo.annotators.core import BaseAnnotator
from robokudo.behaviours.ensure_world_synchronized import EnsureWorldSynchronized
from robokudo.world_sync import WorldSyncManager


def test_manager_fetches_once_and_keeps_synchronizer_until_close(monkeypatch):
    """
    The application owns one fetched world and its live update subscriber.
    """
    node = Mock()
    world = Mock()
    fetch = Mock(return_value=world)
    synchronizer = Mock()
    create_synchronizer = Mock(return_value=synchronizer)
    install_world = Mock()
    init_tracker = Mock()
    monkeypatch.setattr(world_sync_module, "fetch_world_from_service", fetch)
    monkeypatch.setattr(world_sync_module, "WorldSynchronizer", create_synchronizer)
    monkeypatch.setattr(rk_world, "set_world", install_world)
    monkeypatch.setattr(rk_world, "init_world_entity_tracker_from_world", init_tracker)
    manager = WorldSyncManager(node)
    fetch.assert_not_called()

    manager.ensure_ready()
    manager.ensure_ready()

    fetch.assert_called_once_with(node=node)
    create_synchronizer.assert_called_once_with(_world=world, node=node)
    install_world.assert_called_once_with(world)
    init_tracker.assert_called_once_with(world)
    synchronizer.close.assert_not_called()

    manager.close()
    synchronizer.close.assert_called_once_with()


def test_manager_is_found_through_registered_node_until_closed(monkeypatch):
    """
    Any camera pipeline can locate the same application-owned manager.
    """
    node = Mock()
    monkeypatch.setattr(
        world_sync_module, "ROSNodeRegistry", lambda: Mock(get=lambda: node)
    )
    manager = WorldSyncManager(node)

    manager.bind_to_node()
    assert WorldSyncManager.current() is manager

    manager.close()
    assert "_robokudo_world_sync_manager" not in node.__dict__


def test_ensure_world_synchronized_calls_process_manager(monkeypatch):
    """
    The readiness step delegates to the camera-independent manager.
    """
    manager = Mock()
    monkeypatch.setattr(WorldSyncManager, "current", lambda: manager)

    assert EnsureWorldSynchronized().update() == Status.SUCCESS
    manager.ensure_ready.assert_called_once_with()


def test_ensure_world_synchronized_is_a_plain_behaviour():
    """
    World synchronization does not require annotator access to the CAS.
    """
    behaviour = EnsureWorldSynchronized()

    assert isinstance(behaviour, Behaviour)
    assert not isinstance(behaviour, BaseAnnotator)
