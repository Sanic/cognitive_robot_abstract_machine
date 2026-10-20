"""
The contract :mod:`basstler` has to keep: every caller importing the clone's own source,
every module importable on its own, every entry point reachable through
``python3 -m``, every workflow that runs a module installing its dependencies first, and
nothing left behind under ``.claude/``.

:mod:`basstler.package_layout` discovers the modules and the entry points from the package
itself, so nothing here is a second list of them, and the workflows are read from the
directory holding them rather than named.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from basstler.locations import ClaudeCodeVariable, PackageLocation, ProjectLocation

from basstler import _version
from basstler.package_layout import (
    PackageModule,
    command_line_entry_points,
    package_modules,
)

from .constants import (
    ConfigurationVariable,
    EnvironmentVariable,
)
from .project_declaration import ProjectDeclaration
from .script_runner import ScriptRunner, configuration_value

CLAUDE_DIRECTORY = (
    PackageLocation.REPOSITORY_ROOT / ProjectLocation.CLAUDE_CODE_DIRECTORY
)
"""
The directory the migration emptied of Python.

Its SKILL.md files, settings.json and bash entry points stay; not one ``.py`` file does.
"""

WORKFLOWS_DIRECTORY = PackageLocation.REPOSITORY_ROOT / ".github" / "workflows"
"""
Where this repository's Actions workflows live.

Read as a directory rather than named one by one, so a workflow added later is checked
without anyone recording that it exists.
"""


SHELL_CONFIGURATION = (
    PackageLocation.REPOSITORY_ROOT
    / ProjectLocation.HOOKS
    / "resolve-personal-notes-config.sh"
)
"""
The file the bash callers source, which is where a module gets a shell name and the
interpreter its import path.
"""


@dataclass(frozen=True, kw_only=True)
class ConfiguredInterpreterRunner(ScriptRunner):
    """
    Runs this interpreter the way a bash caller does: after sourcing the shell
    configuration, and so with whatever import path it sets up.
    """

    @property
    def command(self) -> tuple[str, ...]:
        """:return: bash sourcing the configuration, then handing over to this interpreter."""
        return (
            "bash",
            "-c",
            'source "$0" && exec "$@"',
            str(SHELL_CONFIGURATION),
            sys.executable,
        )


def run_as_a_caller(*arguments: str) -> subprocess.CompletedProcess[str]:
    """
    Run this interpreter as a bash caller does, with an environment that cannot help it
    find the package.

    ``PYTHONPATH`` is removed so a pass proves the import path comes from the shell
    configuration, rather than from whatever the caller's shell happened to export.

    :param arguments: Arguments to the interpreter, e.g. ``("-c", "import basstler")``.
    :return: The completed process, with output captured as text.
    """
    return ConfiguredInterpreterRunner(
        project_root=PackageLocation.REPOSITORY_ROOT.value,
        removed_variable_prefixes=(EnvironmentVariable.IMPORT_PATH,),
    ).run(*arguments)


MODULE_VARIABLE = re.compile(r'^([A-Z_]+)="(basstler\.[a-z_]+)"', re.MULTILINE)
"""
One ``NAME="basstler.module"`` assignment in the shell configuration.
"""

DEPENDENCY_INSTALL = re.compile(
    r"(pip install|uv pip install|uv sync)[^\n]*" r"(BASSTLER_SOURCE_TREE|basstler)"
)
"""
A step that installs this package, and with it the dependencies its metadata declares,
however the caller spells the directory.
"""


def names_of(module: PackageModule) -> frozenset[str]:
    """
    Every spelling a caller can invoke a module by.

    Bash callers go through a shell variable rather than the dotted path, so the variable
    names are read from the configuration that assigns them instead of written out here.

    :param module: The module to name.
    :return: Its import path, plus any shell variable holding that path.
    """
    assignments = MODULE_VARIABLE.findall(SHELL_CONFIGURATION.read_text())
    return frozenset({module.import_path}) | {
        variable for variable, path in assignments if path == module.import_path
    }


def installs_the_dependencies(caller_source: str) -> bool:
    """
    :param caller_source: A caller's own file.
    :return: Whether it installs this package's dependencies before running anything.
    """
    return DEPENDENCY_INSTALL.search(caller_source) is not None


# %% a caller imports the clone's own source


def test_a_caller_imports_the_clones_own_source_with_no_install():
    """
    What a bash caller imports is this clone's source tree, even where nothing is
    installed: ``-S`` leaves site-packages, and with it any install, out of the search.
    """
    result = run_as_a_caller("-S", "-c", "import basstler; print(basstler.__file__)")

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == PackageLocation.DIRECTORY / "__init__.py"


def test_the_package_version_is_the_repository_version():
    """
    ``_version.py`` is written by ``scripts/sync_version.py`` from the root ``VERSION``
    file, the same as every other package here, so one bump moves them all together.
    """
    assert (
        _version.__version__
        == (PackageLocation.REPOSITORY_ROOT / "VERSION").read_text().strip()
    )


# %% each module stands on its own


@pytest.mark.parametrize("module", package_modules(), ids=lambda module: module.name)
def test_every_module_imports_on_its_own(module: PackageModule):
    """
    Each module imports in an interpreter that has imported nothing else.

    One subprocess per module rather than one for all of them: an import cycle only bites
    whichever module a caller reaches first, so a suite that imports them together can
    stay green while a single-module entry point is broken.
    """
    result = run_as_a_caller("-c", f"import {module.import_path}")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module", command_line_entry_points(), ids=lambda module: module.name
)
def test_every_entry_point_answers_help_through_the_module_runner(
    module: PackageModule,
):
    """
    Each entry point is reachable as ``python -m basstler.<name>``.

    That form rather than a path to the file: a module run by path puts its own directory
    on ``sys.path`` instead of the repository root, so its absolute imports of its
    siblings would not resolve. Every bash caller invokes them this way.
    """
    result = run_as_a_caller("-m", module.import_path, "--help")

    assert result.returncode == 0, result.stderr


# %% what a workflow runs, and what it installs first


def workflows_that_run_a_module() -> tuple[tuple[Path, PackageModule], ...]:
    """
    Every Actions workflow that runs a module of this package, paired with that module.

    Discovered from the workflow directory and the package itself rather than declared:
    an Actions runner is the one caller no session hook reaches, so a workflow added
    later has to be found rather than remembered.

    :return: Each workflow's path and the module it runs, in path then name order.
    """
    return tuple(
        (workflow_path, module)
        for workflow_path in sorted(WORKFLOWS_DIRECTORY.glob("*.yml"))
        for module in package_modules()
        if any(name in workflow_path.read_text() for name in names_of(module))
    )


def test_every_workflow_that_runs_a_module_installs_the_dependencies_first():
    """
    A runner starts with nothing installed and no hook runs on it, so a workflow that
    invokes a module installs this package's dependencies itself.

    The emptiness assertion is the scan's own coverage: a discovery that finds no
    workflow checks nothing, and a parametrization over it would pass while doing so.
    """
    running = workflows_that_run_a_module()

    assert running != ()
    assert [
        workflow_path.name
        for workflow_path, _ in running
        if not installs_the_dependencies(workflow_path.read_text())
    ] == []


# %% nothing is left behind


def test_no_python_module_remains_under_the_claude_directory():
    """
    The migration's own completeness assertion from the other side.

    ``.claude/`` keeps its SKILL.md files, settings.json and bash entry points - Claude
    Code discovers those by path - and not one Python file.
    """
    remaining_module_paths = sorted(
        str(path.relative_to(PackageLocation.REPOSITORY_ROOT.value))
        for path in CLAUDE_DIRECTORY.rglob("*.py")
    )

    assert remaining_module_paths == []


def test_no_module_sits_directly_in_the_source_tree():
    """
    The package's modules live under ``src/``, and nothing beside ``pyproject.toml`` is
    one.

    A module a branch adds where the package used to be merges with no conflict and is
    then never imported, so the move's own completeness has to be asserted rather than
    trusted to the merge.
    """
    stray_module_paths = sorted(
        str(path.relative_to(PackageLocation.REPOSITORY_ROOT.value))
        for path in PackageLocation.SOURCE_TREE.value.glob("*.py")
    )

    assert stray_module_paths == []


# %% the shell's copies of what the package states


def test_the_shell_creates_the_environment_the_package_names():
    """
    The shell creates the environment; the package names it for every Python reader.
    """
    assert configuration_value(ConfigurationVariable.ENVIRONMENT_DIRECTORY) == str(
        ProjectLocation.PACKAGE_ENVIRONMENT
    )


def test_the_shell_asks_for_the_interpreters_the_package_supports():
    """
    The interpreter an environment is created from satisfies the package's own
    ``requires-python``, so the two cannot disagree about which versions run it.
    """
    assert (
        configuration_value(ConfigurationVariable.PYTHON_REQUIREMENT)
        == ProjectDeclaration.of_this_package().requires_python
    )


# %% what the skill documents name


SHELL_VARIABLE_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
"""
A ``${NAME}`` reference to an upper-case shell variable, as a skill document writes one.
"""


def variables_the_configuration_defines() -> frozenset[str]:
    """
    :return: Every variable set once the shell configuration is sourced, in an environment
        carrying nothing but a search path, so an inherited variable cannot pass for one
        the configuration defines.
    """
    return frozenset(
        subprocess.run(
            ["bash", "-c", 'source "$0" && compgen -v', SHELL_CONFIGURATION],
            cwd=PackageLocation.REPOSITORY_ROOT.value,
            capture_output=True,
            text=True,
            check=True,
            env={
                EnvironmentVariable.EXECUTABLE_SEARCH_PATH: os.environ[
                    EnvironmentVariable.EXECUTABLE_SEARCH_PATH
                ]
            },
        ).stdout.split()
    )


def test_every_variable_a_skill_document_names_is_defined():
    """
    A document naming a variable nothing defines hands its reader an empty string, and a
    command built from it fails far from the cause - a rename or a move leaves exactly
    this behind.
    """
    claude_code_directory = (
        PackageLocation.REPOSITORY_ROOT / ProjectLocation.CLAUDE_CODE_DIRECTORY
    )
    personal_notes = PackageLocation.REPOSITORY_ROOT / ProjectLocation.PERSONAL_NOTES
    named = {
        variable_name
        for document in claude_code_directory.rglob("*.md")
        if not document.is_relative_to(personal_notes)
        for variable_name in SHELL_VARIABLE_REFERENCE.findall(document.read_text())
    }

    assert named != set()
    assert (
        sorted(named - variables_the_configuration_defines() - set(ClaudeCodeVariable))
        == []
    )
