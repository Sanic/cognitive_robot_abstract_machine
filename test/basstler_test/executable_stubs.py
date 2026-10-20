"""
Standing an executable in for the real one, on a test subprocess's ``PATH``.

Two suites need the same two moves - put a stub where a script will find it, and take a
real executable out of reach - so both live here rather than in whichever suite happened
to need them first.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .constants import DatasetLocation, EnvironmentVariable, FileMode


@dataclass(frozen=True)
class ExecutableStubDirectory:
    """
    A directory of stubbed executables, meant to be placed first on a subprocess's
    ``PATH`` so the script under test finds the stub instead of the real thing.
    """

    path: Path
    """
    The directory itself, which is what goes on ``PATH``.
    """

    @classmethod
    def create(cls, parent: Path) -> ExecutableStubDirectory:
        """
        :param parent: Where to create the directory, typically pytest's ``tmp_path``.
        :return: An empty stub directory, ready for :meth:`install`.
        """
        path = parent / "stub-bin"
        path.mkdir()
        return cls(path)

    @staticmethod
    def stub_for(executable_name: str) -> Path:
        """
        :param executable_name: The executable to stand in for.
        :return: The script standing in for it, ``dataset/stubs/<name>.sh``.
        """
        return DatasetLocation.STUBS / f"{executable_name}.sh"

    def install(self, executable_name: str) -> None:
        """
        Copy the stub backing *executable_name* into this directory.

        :param executable_name: The executable to stand in for.
        """
        destination = self.path / executable_name
        shutil.copy(self.stub_for(executable_name), destination)
        destination.chmod(FileMode.EXECUTABLE)

    def ahead_of(self, search_path: str) -> str:
        """
        :param search_path: A ``PATH`` string to fall back to.
        :return: *search_path* with this directory first, so its stubs win.
        """
        return os.pathsep.join([str(self.path), search_path])

    def ahead_of_inherited_path(self) -> str:
        """
        :return: This process's own ``PATH`` with this directory first.
        """
        return self.ahead_of(
            os.environ.get(EnvironmentVariable.EXECUTABLE_SEARCH_PATH, "")
        )


def path_hiding_executables(mirror_parent: Path, *executable_names: str) -> str:
    """
    Build a ``PATH`` string equivalent to the current one but with every one of
    *executable_names* unfindable through it.

    Mirrors (via symlinks) any directory that provides one of them into a copy missing
    just those files, rather than dropping the whole directory from `PATH` - the
    directory providing them (typically ``/usr/bin``) also provides ``bash``, ``git`` and
    ``python3``, which the script under test still needs to run at all.

    :param mirror_parent: Where to create mirror directories.
    :param executable_names: The executables to hide.
    :return: The adjusted ``PATH`` string.
    """
    hidden = frozenset(executable_names)
    entries = []
    for mirror_index, entry in enumerate(
        os.environ.get(EnvironmentVariable.EXECUTABLE_SEARCH_PATH, "").split(os.pathsep)
    ):
        directory = Path(entry)
        if not directory.is_dir() or not any(
            (directory / name).exists() for name in hidden
        ):
            entries.append(entry)
            continue
        mirror = mirror_parent / f"hidden-executables-{mirror_index}"
        mirror.mkdir()
        for item in directory.iterdir():
            if item.name in hidden:
                continue
            try:
                (mirror / item.name).symlink_to(item)
            except OSError:
                continue
        entries.append(str(mirror))
    return os.pathsep.join(entries)
