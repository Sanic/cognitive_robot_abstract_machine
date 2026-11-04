"""
Keeping a run's worlds in a database between its steps.

A step hands the next one a world by its id, so what matters is that a world written
comes back as the world it was, and that an id from another run's schema is reported
rather than answered with nothing.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import InvalidRequestError

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.warsaw.exceptions import (
    DatabaseNotConfiguredError,
    WorldNotInDatabaseError,
)
from experiments.warsaw.pipeline.database.world_store import WorldStore

# %% a database of this test's own


@pytest.fixture
def store(monkeypatch, tmp_path) -> WorldStore:
    """
    :return: A store pointed at a database that lives only for this test.
    """
    # The generated interface declares its tables when it is imported, and a process
    # that already holds a declaration of them refuses a second one. Whether this test
    # landed in such a process is a property of the run, not of the store.
    try:
        WorldStore().mappings()
    except InvalidRequestError as already_declared:
        pytest.skip(f"the ORM interface cannot be loaded here: {already_declared}")

    monkeypatch.setenv(
        "SEMANTIC_DIGITAL_TWIN_DATABASE_URI", f"sqlite:///{tmp_path / 'worlds.db'}"
    )
    store = WorldStore()
    store.create_tables()
    return store


@pytest.fixture
def two_body_world() -> World:
    """
    :return: A world of one body hanging from another.
    """
    world = World()
    root = Body(name=PrefixedName("root", "warsaw_test"))
    child = Body(name=PrefixedName("child", "warsaw_test"))
    with world.modify_world():
        world.add_body(root)
        world.add_body(child)
        world.add_connection(
            FixedConnection(root, child, HomogeneousTransformationMatrix())
        )
    return world


# %% writing one and reading it back


def test_making_the_tables_declares_the_ones_the_orm_asks_for(store):
    """
    A generated class has no table until one is made for it, and a world holding one
    cannot be written.
    """
    assert store.create_tables() > 0


def test_a_world_read_back_holds_the_bodies_it_was_written_with(store, two_body_world):
    """
    The next step is handed an id, so what it reads has to be the world that was
    written.
    """
    world_id = store.write(two_body_world)

    read_back = store.read(world_id)

    assert {body.name.name for body in read_back.bodies} == {
        body.name.name for body in two_body_world.bodies
    }


def test_an_id_nothing_was_written_under_is_reported(store):
    """
    From a run's own schema this also means the id was written by a different run.
    """
    with pytest.raises(WorldNotInDatabaseError):
        store.read(4242)


# %% the database a store reaches


@pytest.fixture
def session_opened_on_another_database(monkeypatch, tmp_path):
    """
    Open the semantic digital twin's process-wide session on a database of its own, and
    forget it again afterwards so no later test reaches that database.
    """
    monkeypatch.setenv(WorldStore().variable, f"sqlite:///{tmp_path / 'earlier.db'}")
    semantic_digital_twin_sessionmaker()
    yield
    semantic_digital_twin_sessionmaker.cache_clear()


def test_a_store_reaches_the_database_the_environment_names_when_it_is_made(
    session_opened_on_another_database, monkeypatch, tmp_path
):
    """
    A step is pointed at its run's schema through the environment, and an earlier step
    in the same process may already have opened a session on another one.
    """
    later = f"sqlite:///{tmp_path / 'later.db'}"
    monkeypatch.setenv(WorldStore().variable, later)

    assert WorldStore().engine.url == make_url(later)


def test_a_store_without_a_database_to_reach_is_refused(monkeypatch):
    """
    Without a database named, a world would have nowhere to go between steps.
    """
    store = WorldStore()
    monkeypatch.delenv(store.variable, raising=False)

    with pytest.raises(DatabaseNotConfiguredError):
        store.engine
