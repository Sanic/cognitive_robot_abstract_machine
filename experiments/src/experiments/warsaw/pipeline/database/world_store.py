"""
Reading and writing worlds, without holding the ORM open across a rebuild.

The pipeline regenerates the ORM twice: once when it puts the ontology back as it is
committed, and once when it builds the classes a scene needed that the ontology does not
have. An interpreter that imported the ORM before either of those is holding the version
from before it, and every class it hands out is the wrong one.

So this is the only place the generated interface is named, and it is named inside the
methods rather than at the top of the file. A step reaches worlds through this and
imports nothing itself, which is what lets the steps run in one process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType

from krrood.ormatic.utils import create_engine
from typing_extensions import TYPE_CHECKING

from experiments.warsaw.exceptions import (
    DatabaseNotConfiguredError,
    WorldNotInDatabaseError,
)

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from semantic_digital_twin.world import World


@dataclass
class WorldStore:
    """
    The database a run's worlds are kept in.
    """

    variable: str = "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
    """
    The environment variable the database is read from, the one a run points at its own
    schema.
    """

    def mappings(self) -> ModuleType:
        """
        Load the generated interface that says how a world is stored.

        Importing it is what registers a data access object for every mapped class, so
        anything converting a world has to have done it -- ``to_dao`` looks the mapping
        up by class and reports the world as unmapped when nothing has.

        :return: The interface module.
        """
        import semantic_digital_twin.orm.ormatic_interface as ormatic_interface

        return ormatic_interface

    @cached_property
    def engine(self) -> Engine:
        """
        Built once per store, from the database the environment names when it is first
        asked for, whichever database other code in the process has opened before.

        :return: A connection to that database.
        :raises DatabaseNotConfiguredError: If the environment names none.
        """
        uri = os.environ.get(self.variable)
        if uri is None:
            raise DatabaseNotConfiguredError(variable=self.variable)
        return create_engine(uri)

    def create_tables(self) -> int:
        """
        Build the tables the ORM as it now stands asks for.

        A generated class has no table until one is made for it, and a world holding one
        cannot be written.

        :return: How many tables the ORM declares.
        """
        base = self.mappings().Base
        base.metadata.create_all(bind=self.engine)
        return len(base.metadata.tables)

    def write(self, world: World) -> int:
        """
        Write a world to the database.

        :param world: The world to write.
        :return: The id it was written under, which is what a later step takes.
        """
        from krrood.ormatic.data_access_objects.helper import to_dao
        from sqlalchemy.orm import Session

        self.mappings()
        with Session(self.engine) as session:
            stored = to_dao(world)
            session.add(stored)
            session.commit()
            return stored.database_id

    def read(self, world_id: int) -> World:
        """
        Read a world back out of the database.

        :param world_id: The id it was written under.
        :return: The world.
        :raises WorldNotInDatabaseError: If nothing is stored under that id, which from
            a run's own schema also means it was written by a different run.
        """
        from sqlalchemy.orm import Session

        with Session(self.engine) as session:
            stored = session.get(self.mappings().WorldMappingDAO, world_id)
            if stored is None:
                raise WorldNotInDatabaseError(world_id=world_id)
            return stored.from_dao()
