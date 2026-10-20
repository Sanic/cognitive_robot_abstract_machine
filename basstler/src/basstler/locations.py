"""
Every file and directory this package names, each written once.

..note:: Imports nothing outside the standard library, since :mod:`basstler.dependencies`
    reads its declaration's location from here before anything is installed.
"""

from __future__ import annotations

import os
from enum import ReprEnum, StrEnum
from pathlib import Path


class PathEnumeration(Path, ReprEnum):
    """
    An enumeration whose members are paths.

    A member is the path it names - joined onto another with ``/``, opened, passed to a
    subprocess, formatted as its text - and every path derived from one is a plain
    :class:`~pathlib.Path`.

    ..warning:: ``Path.name`` shadows the enumeration's own ``name``: ``member.name`` is
        the path's last component, and the member's name is ``member._name_``.
    """

    def __hash__(self) -> int:
        """
        Hash as the path does, so a member and its path are one key; the enumeration's own
        hash would tell them apart.
        """
        return hash(self.value)

    def __eq__(self, value: object, /) -> bool:
        """
        The value of the Enum should be equal to an equivalent Path instance.

        :param value: A Path instance to compare.
        :return: True if the value is equal to an equivalent Path instance, False otherwise.
        """
        return self.value == value


    def with_segments(self, *path_segments: str | os.PathLike[str]) -> Path:
        """
        :param path_segments: The segments of a path derived from this one.
        :return: That path, as a plain path rather than a lookup of a member by value.
        """
        return Path(*path_segments)


class PackageLocation(PathEnumeration):
    """
    The package's own directory and the files shipped in it, as absolute paths.
    """

    DIRECTORY = Path(__file__).parent
    """
    This package's own directory, under its source tree's ``src`` directory.
    """

    SOURCE_TREE = DIRECTORY.parent.parent
    """
    The directory holding the package's ``pyproject.toml``, ``README.md`` and ``src``.

    Found from the modules, so it is the clone's own source tree wherever the package is
    imported from that source - through an editable install or ``PYTHONPATH``.
    """

    REPOSITORY_ROOT = SOURCE_TREE.parent
    """
    The repository root, which holds the source tree.
    """

    STACK_CONFIGURATION = DIRECTORY / "stack.toml"
    """
    The checked-in stack configuration every run starts from, before any per-user
    override.
    """

    BOARD = DIRECTORY / "board.json"
    """
    The exported snapshot of the fork's open pull requests - scratch state, never committed.
    """

    DEPENDENCY_DECLARATION = SOURCE_TREE / "pyproject.toml"
    """
    The package metadata, whose ``[project] dependencies`` this package installs.
    """

    TEMPLATES = DIRECTORY / "templates"
    """
    The page templates dashboards are rendered from.
    """

    QUERIES = DIRECTORY / "queries"
    """
    The GraphQL documents the upstream review reader sends.
    """


class ProjectLocation(PathEnumeration):
    """
    The files this package reads, runs or writes in a project, relative to its root.

    Those on the personal-notes branch are relative to that branch's root, which is the
    same tree.
    """

    CLAUDE_CODE_DIRECTORY = Path(".claude")
    """
    The directory Claude Code reads a project's settings, hooks and skills from.
    """

    HOOKS = CLAUDE_CODE_DIRECTORY / "hooks"
    """
    The shell entry points that read and write personal-notes data.
    """

    PERSONAL_NOTES = CLAUDE_CODE_DIRECTORY / "personal"
    """
    Where the personal-notes branch keeps everything it holds.
    """

    PACKAGE_SOURCE_TREE = PackageLocation.SOURCE_TREE.value.relative_to(
        PackageLocation.REPOSITORY_ROOT.value
    )
    """
    The directory holding the package's ``pyproject.toml``, ``README.md`` and ``src``.
    """

    PACKAGE_IMPORT_DIRECTORY = PACKAGE_SOURCE_TREE / "src"
    """
    The directory a caller puts on the import path to import this clone's own package.
    """

    PACKAGE = PACKAGE_IMPORT_DIRECTORY / PackageLocation.DIRECTORY.value.name
    """
    This package's own directory.
    """

    PACKAGE_ENVIRONMENT = PACKAGE_SOURCE_TREE / ".venv"
    """
    The package's own virtual environment, which a session start creates and installs
    the package into.

    Mirrors ``BASSTLER_ENVIRONMENT_DIRECTORY`` in ``resolve-personal-notes-config.sh``;
    a test holds the two equal.
    """

    PACKAGE_ENVIRONMENT_INTERPRETER = PACKAGE_ENVIRONMENT / "bin" / "python"
    """
    The interpreter of :attr:`PACKAGE_ENVIRONMENT`, which every caller runs the package
    with once it exists.
    """

    PERSONAL_NOTES_CONFIGURATION_SCRIPT = HOOKS / "resolve-personal-notes-config.sh"
    """
    The shell configuration that resolves the personal-notes remote and branch, and
    fetches it.
    """

    PERSONAL_NOTES_WRITER_SCRIPT = HOOKS / "write-personal-notes-file.sh"
    """
    The helper that commits one file to the personal-notes branch and pushes it.
    """

    PERSONAL_NOTES_BRANCH_CREATION_SCRIPT = HOOKS / "create-personal-notes-branch.sh"
    """
    The helper that creates the personal-notes branch where it does not exist yet.
    """

    PERSONAL_NOTES_DOCUMENT = PERSONAL_NOTES / "cram-notes.md"
    """
    The notes file itself, unless a clone's settings put it elsewhere.
    """

    PLANS = PERSONAL_NOTES / "plans"
    """
    Where plans live on the personal-notes branch.

    Mirrors ``PLANS_DIR`` in ``resolve-personal-notes-config.sh``, which is the shell
    half of the same tooling; a test holds the two equal so the mirror cannot drift.
    """

    PERSONAL_STACK_CONFIGURATION = PERSONAL_NOTES / "stack.toml"
    """
    The per-user override of :attr:`PackageLocation.STACK_CONFIGURATION`.
    """

    PLAN_ITEM_MODE_DEFAULTS = PACKAGE / "plan-item-modes.toml"
    """
    The shipped plan-item execution modes.
    """

    PERSONAL_PLAN_ITEM_MODES = PERSONAL_NOTES / "plan-item-modes.toml"
    """
    The per-user override of :attr:`PLAN_ITEM_MODE_DEFAULTS`.
    """


class ClaudeCodeVariable(StrEnum):
    """
    The environment variables Claude Code sets for the commands it runs.
    """

    PROJECT_DIRECTORY = "CLAUDE_PROJECT_DIR"
    """
    The root of the project the session runs in.
    """
