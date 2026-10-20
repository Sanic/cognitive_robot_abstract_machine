#!/usr/bin/env python3
"""
What this package needs installed, and which of it this environment is missing.

The declaration is ``pyproject.toml``'s ``[project]`` table: the package's own name and its
``dependencies``, which is where every package in this repository states them, so there is
one list rather than a metadata table and a requirements file that can disagree. An
environment runs the package only with the package itself installed in it, so the package
counts as a requirement of its own.

Kept as a module rather than as a snippet inside the shell that calls it: the bash entry
points ask this question before anything is installed, and a question with parsing in it
is real, testable code wherever it is written.

Usage:
    python3 -m basstler.dependencies [--declaration <pyproject.toml>]

Prints one requirement specifier per missing requirement, and nothing at all when the
environment already has them all.

..note:: Imports nothing outside the standard library. It runs before any install, so a
    dependency of its own would be the one thing it could never report.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from basstler.locations import PackageLocation
from basstler.standard_streams import StandardStreamHandler

logger = StandardStreamHandler.logger_for(__name__)
"""
This module's logger, which is also what its command prints through.
"""


class PyprojectKey(StrEnum):
    """
    The keys of ``pyproject.toml`` this module reads.
    """

    PROJECT = "project"
    """
    The table a package's own metadata lives in.
    """

    NAME = "name"
    """
    The field of that table naming the package's own distribution.
    """

    DEPENDENCIES = "dependencies"
    """
    The field of that table listing the requirement specifiers.
    """


@dataclass
class UnreadableDependencyDeclarationError(Exception):
    """
    Raised when the declaration cannot be read, so nothing can be said about what is
    missing.

    Distinct from *nothing is missing*, which is what a caller would otherwise conclude
    from an empty answer and act on by installing nothing.
    """

    declaration: Path
    """
    The file that was to be read.
    """

    def __str__(self) -> str:
        """:return: What to tell a reader who asked what is missing and cannot be told."""
        return (
            f"{self.declaration} does not exist, so this package's dependencies cannot "
            f"be read"
        )


@dataclass(frozen=True)
class Dependency:
    """
    One requirement this package declares.
    """

    specifier: str
    """
    The requirement as ``pyproject.toml`` writes it, version bounds and all.
    """

    constraint_start: re.Pattern[str] = field(
        default=re.compile(r"[<>=!~;\[ ]"), repr=False, compare=False
    )
    """
    The first character that ends a distribution's name and begins a version bound, an extra
    or an environment marker, in a PEP 508 specifier.
    """

    @property
    def distribution_name(self) -> str:
        """:return: The distribution this requirement names, without its constraints."""
        return self.constraint_start.split(self.specifier, maxsplit=1)[0].strip()

    @property
    def is_missing(self) -> bool:
        """
        Presence rather than version: an installed distribution is left alone, which is
        what lets a session start run this on every start and install nothing. Names are
        compared as PEP 503 does, so a declaration need not match a distribution's own
        spelling.

        :return: Whether this environment has no distribution of that name.
        """
        return next(iter(distributions(name=self.distribution_name)), None) is None


@dataclass(frozen=True)
class DependencyDeclaration:
    """
    A ``pyproject.toml`` and the dependencies it declares.
    """

    path: Path
    """
    The ``pyproject.toml`` to read.
    """

    @classmethod
    def of_this_package(cls) -> DependencyDeclaration:
        """
        :return: This package's own metadata, found beside the modules it declares the
            dependencies of.
        """
        return cls(PackageLocation.DEPENDENCY_DECLARATION.value)

    def project(self) -> Dependency:
        """
        :return: The package's own distribution, as a requirement of an environment that
            runs it.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        return Dependency(self._project_table()[PyprojectKey.NAME])

    def dependencies(self) -> tuple[Dependency, ...]:
        """
        :return: Every dependency it declares, in the order it declares them.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        return tuple(
            Dependency(specifier)
            for specifier in self._project_table().get(PyprojectKey.DEPENDENCIES, [])
        )

    def requirements(self) -> tuple[Dependency, ...]:
        """
        :return: Everything an environment needs to run the package: the package itself,
            then its dependencies.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        return (self.project(), *self.dependencies())

    def missing(self) -> tuple[Dependency, ...]:
        """
        :return: The requirements this environment does not have.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        return tuple(
            requirement for requirement in self.requirements() if requirement.is_missing
        )

    def _project_table(self) -> dict[str, Any]:
        """
        :return: The ``[project]`` table, the package's own metadata.
        :raises UnreadableDependencyDeclarationError: If the file is absent.
        """
        if not self.path.is_file():
            raise UnreadableDependencyDeclarationError(self.path)
        return tomllib.loads(self.path.read_text(encoding="utf-8"))[
            PyprojectKey.PROJECT
        ]


class ExitCode(IntEnum):
    """
    How the command ended, as its caller reads it.
    """

    SUCCESS = 0
    """
    The missing requirements, if any, were printed.
    """

    UNREADABLE_DECLARATION = 1
    """
    The declaration could not be read, so nothing was printed.
    """


def main() -> ExitCode:
    """
    Print one specifier per missing requirement, for a caller to report.

    :return: How the command ended.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DependencyDeclaration.of_this_package().path,
        help="The pyproject.toml to read the dependencies from",
    )
    declaration = DependencyDeclaration(parser.parse_args().declaration)
    if not declaration.path.is_file():
        logger.error(str(UnreadableDependencyDeclarationError(declaration.path)))
        return ExitCode.UNREADABLE_DECLARATION
    for requirement in declaration.missing():
        logger.info(requirement.specifier)
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
