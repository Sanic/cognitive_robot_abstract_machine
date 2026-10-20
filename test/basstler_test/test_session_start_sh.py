"""
Integration tests for session-start.sh's summary report.

Cover the three guards the report exists for: naming which situation a branch with no
plan item is actually in, surfacing check-setup.sh's verdict rather than leaving it to be
remembered, and installing the package's dependencies so nothing downstream has to run
without them. All three stay invisible to anyone who uses neither plans nor personal
notes.

Run against a scratch project root with a local bare repository standing in for the
personal-notes remote - no network access or real personal-notes branch involved.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from basstler.locations import ProjectLocation

from .constants import (
    ConfigurationVariable,
    DatasetLocation,
    EnvironmentVariable,
    FileMode,
    PackageManager,
    PersonalNotesPath,
    ScratchBranch,
    StubVariable,
)
from .executable_stubs import ExecutableStubDirectory, path_hiding_executables
from .project_declaration import ProjectDeclaration
from .scratch_repository import SCRATCH_IDENTITY, ScratchRepository
from .script_runner import configuration_value
from .session_start_summary import (
    SummaryLine,
    SummaryMessage,
    summary_message,
    summary_value,
)

PLAN_MANIFEST = (DatasetLocation.DIRECTORY / "plan.yaml").read_text()

PLAN_MANIFEST_WITH_TRACKING_ISSUE = (
    DatasetLocation.DIRECTORY / "plan-with-tracking-issue.yaml"
).read_text()

TRACKING_ISSUE = "55"

PLAN_IDENTIFIER = "test-plan"


MANIFEST_PATH = ProjectLocation.PLANS / PLAN_IDENTIFIER / "plan.yaml"

CLAUDE_LOCAL_MD = "CLAUDE.local.md"


def branch_index(plan_identifier_by_branch: Mapping[str, str]) -> str:
    """
    Build a branch index mapping each branch to the plan that tracks it.

    :param plan_identifier_by_branch: Plan ids, keyed by the branch each one tracks.
    :return: The index's tab-separated content.
    """
    return "".join(
        f"{branch}\t{plan_identifier}\n"
        for branch, plan_identifier in plan_identifier_by_branch.items()
    )


# %% the scratch layout


@pytest.fixture
def session_start_repository(
    scratch_repository: ScratchRepository,
) -> ScratchRepository:
    """
    A scratch repository carrying the real session-start.sh and everything else a set up
    clone has, with nothing published to the notes remote yet.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to publish a notes branch and run the hook.
    """
    scratch_repository.install_hook_scripts(
        "resolve-personal-notes-config.sh",
        "session-start-messages.sh",
        "session-start.sh",
        "check-setup.sh",
    )
    scratch_repository.install_package()
    scratch_repository.install_package_environment()
    scratch_repository.write_setup_prerequisites()
    scratch_repository.commit_everything("initial commit")
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def run_session_start(
    repository: ScratchRepository, environment: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's session-start.sh.

    :param repository: A fixture-built scratch repository.
    :param environment: Variables to set for this run.
    :return: The finished subprocess.
    """
    return repository.run_hook_script("session-start.sh", **(environment or {}))


def publish_and_run(
    repository: ScratchRepository,
    notes_branch_files: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Publish everything a set up notes branch carries, plus *notes_branch_files*, then
    run session-start.sh against it.

    The recorded git identity is the one this repository already commits with, so the
    baseline these tests assert against is a clone with nothing left to set up - what
    the git identity round trip itself does is
    ``test_git_identity_sync.py``'s subject, not this module's.

    :param repository: The fixture-built scratch repository.
    :param notes_branch_files: Extra file contents, keyed by path relative to the
        project root.
    :param environment: Variables to set for the hook's run.
    :return: The finished session-start.sh process.
    """
    repository.publish_notes_branch(
        {
            ProjectLocation.PERSONAL_NOTES_DOCUMENT: "personal notes\n",
            PersonalNotesPath.GIT_IDENTITY: SCRATCH_IDENTITY.as_git_config_file(),
            **(notes_branch_files or {}),
        }
    )
    return run_session_start(repository, environment)


# %% someone who uses neither plans nor personal notes


def test_reports_nothing_when_no_notes_branch_exists(
    session_start_repository: ScratchRepository,
):
    session_start_repository.run_git("checkout", "--quiet", "-b", ScratchBranch.WORK)

    result = run_session_start(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


# %% the plan line


def test_reports_no_plans_when_none_are_tracked(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.NO_PLANS_TRACKED, ScratchBranch.PERSONAL_NOTES
    )


def test_names_the_missing_item_when_other_plans_are_tracked(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            PersonalNotesPath.BRANCH_INDEX: branch_index(
                {
                    "some-other-branch": PLAN_IDENTIFIER,
                    "a-third-branch": "another-plan",
                }
            ),
            MANIFEST_PATH: PLAN_MANIFEST,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.NO_PLAN_ITEM_TRACKS_BRANCH, ScratchBranch.WORK, "2"
    )


def test_reports_the_plan_that_tracks_this_branch(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            PersonalNotesPath.BRANCH_INDEX: branch_index(
                {ScratchBranch.WORK: PLAN_IDENTIFIER}
            ),
            MANIFEST_PATH: PLAN_MANIFEST_WITH_TRACKING_ISSUE,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.BRANCH_TRACKED_IN_PLAN, PLAN_IDENTIFIER, TRACKING_ISSUE
    )


def test_reports_a_tracked_plan_that_has_no_tracking_issue(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            PersonalNotesPath.BRANCH_INDEX: branch_index(
                {ScratchBranch.WORK: PLAN_IDENTIFIER}
            ),
            MANIFEST_PATH: PLAN_MANIFEST,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.BRANCH_TRACKED_IN_PLAN, PLAN_IDENTIFIER, "none"
    )


def test_reports_a_tracked_branch_whose_manifest_is_missing(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            PersonalNotesPath.BRANCH_INDEX: branch_index(
                {ScratchBranch.WORK: PLAN_IDENTIFIER}
            )
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.PLAN_MANIFEST_MISSING,
        PLAN_IDENTIFIER,
        str(MANIFEST_PATH),
        ScratchBranch.PERSONAL_NOTES,
    )


# %% branches no plan item can ever track


def test_reports_plan_as_not_applicable_on_the_default_branch(
    session_start_repository: ScratchRepository,
):
    session_start_repository.publish_notes_branch(
        {
            ProjectLocation.PERSONAL_NOTES_DOCUMENT: "personal notes\n",
            PersonalNotesPath.BRANCH_INDEX: branch_index(
                {ScratchBranch.WORK: PLAN_IDENTIFIER}
            ),
            MANIFEST_PATH: PLAN_MANIFEST,
        }
    )
    session_start_repository.run_git("checkout", "--quiet", "-b", "main")

    result = run_session_start(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.PLAN_NOT_APPLICABLE
    )


def test_reports_plan_as_not_applicable_on_the_notes_branch(
    session_start_repository: ScratchRepository,
):
    session_start_repository.publish_notes_branch(
        {
            ProjectLocation.PERSONAL_NOTES_DOCUMENT: "personal notes\n",
            PersonalNotesPath.BRANCH_INDEX: branch_index(
                {ScratchBranch.WORK: PLAN_IDENTIFIER}
            ),
            MANIFEST_PATH: PLAN_MANIFEST,
        }
    )
    session_start_repository.run_git(
        "checkout", "--quiet", ScratchBranch.PERSONAL_NOTES
    )

    result = run_session_start(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.PLAN) == summary_message(
        SummaryMessage.PLAN_NOT_APPLICABLE
    )


# %% the setup line


def test_reports_setup_as_ok_when_every_check_passes(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.SETUP) == summary_message(
        SummaryMessage.SETUP_OK
    )


def test_names_every_check_that_needs_setup(
    session_start_repository: ScratchRepository,
):
    (
        session_start_repository.project_root
        / ProjectLocation.CLAUDE_CODE_DIRECTORY
        / "settings.json"
    ).unlink()
    session_start_repository.commit_everything("unregister the SessionStart hook")

    result = publish_and_run(session_start_repository)

    failing_check = "session_start_hook"
    detail = next(
        row.split("\t")[2]
        for row in session_start_repository.run_hook_script(
            "check-setup.sh"
        ).stdout.splitlines()
        if row.split("\t")[0] == failing_check
    )
    assert summary_value(result.stdout, SummaryLine.SETUP) == summary_message(
        SummaryMessage.CHECKS_NEED_SETUP, "1"
    )
    assert f"    {failing_check}: {detail}" in result.stdout


def test_a_failing_setup_check_does_not_fail_the_hook(
    session_start_repository: ScratchRepository,
):
    (
        session_start_repository.project_root
        / ProjectLocation.CLAUDE_CODE_DIRECTORY
        / "settings.json"
    ).unlink()
    session_start_repository.commit_everything("unregister the SessionStart hook")

    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


# %% the dependencies line


DECLARATION_FILE = str(ProjectLocation.PACKAGE_SOURCE_TREE / "pyproject.toml")
"""
Where the scratch clone declares its requirements.
"""

ENVIRONMENT_DIRECTORY = str(ProjectLocation.PACKAGE_ENVIRONMENT)
"""
The package's own environment, as the summary line names it.
"""

SOURCE_TREE_ARGUMENT = f"./{ProjectLocation.PACKAGE_SOURCE_TREE}"
"""
What the installer is handed: the source tree, installed editable.
"""

SUPPORTED_PYTHON = ProjectDeclaration.of_this_package().requires_python
"""
The interpreters the package supports, as its metadata states them and so as the
environment has to be asked for.
"""

UNINSTALLABLE_REQUIREMENT = "basstler-no-such-distribution>=1"
"""
A distribution nothing can have installed, so the run has something to install.

Its own name says why it is here, which matters because a real name that happened to be
installed on the machine running the suite would make the test pass without exercising
anything. The version bound is part of it: what the summary names is the specifier, not
the bare name.
"""


def require_the_uninstallable(repository: ScratchRepository) -> None:
    """
    Leave the scratch clone declaring a dependency that is certainly not installed,
    beside the package's own name, which is.

    :param repository: The fixture-built scratch repository.
    """
    repository.write(
        DECLARATION_FILE,
        ProjectDeclaration(
            name=ProjectDeclaration.of_this_package().name,
            dependencies=(UNINSTALLABLE_REQUIREMENT,),
        ).as_toml(),
    )
    repository.commit_everything("declare something that is not installed")


def environment_interpreter(repository: ScratchRepository) -> Path:
    """
    :param repository: The fixture-built scratch repository.
    :return: Where the scratch clone's package environment keeps its interpreter.
    """
    return repository.project_root / ProjectLocation.PACKAGE_ENVIRONMENT_INTERPRETER


def pip_installer_call(repository: ScratchRepository) -> str:
    """
    :param repository: The fixture-built scratch repository.
    :return: The one invocation ``pip`` is expected to receive, where there is no ``uv``.
    """
    return (
        f"--python {environment_interpreter(repository)} install --editable "
        f"{SOURCE_TREE_ARGUMENT}"
    )


def uv_installer_call(repository: ScratchRepository) -> str:
    """
    :param repository: The fixture-built scratch repository.
    :return: The install invocation ``uv`` is expected to receive, where there is one.
    """
    return (
        f"{PackageManager.PIP} install --python {environment_interpreter(repository)} "
        f"--editable "
        f"{SOURCE_TREE_ARGUMENT}"
    )


def environment_call(repository: ScratchRepository) -> str:
    """
    :param repository: The fixture-built scratch repository.
    :return: The invocation ``uv`` is expected to create the environment with.
    """
    return (
        f"venv --no-project --python {SUPPORTED_PYTHON} "
        f"{repository.project_root / ProjectLocation.PACKAGE_ENVIRONMENT}"
    )


def test_installs_nothing_when_every_requirement_is_already_installed(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    The common case, and the one that decides whether installing on every start is
    affordable: nothing is missing, so no installer runs at all.
    """
    stub_bin.install(PackageManager.PIP)
    call_log = tmp_path / "pip-calls"

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_hiding_executables(tmp_path, PackageManager.UV)
            ),
            StubVariable.PIP_CALL_LOG: str(call_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_ALREADY_INSTALLED, ENVIRONMENT_DIRECTORY
    )
    assert not call_log.exists()


def test_installs_the_package_with_pip_when_a_requirement_is_missing(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    A missing requirement is installed without anyone being asked, by installing the
    package editable into its own environment - with ``pip`` where there is no ``uv`` -
    and the run says so.
    """
    require_the_uninstallable(session_start_repository)
    stub_bin.install(PackageManager.PIP)
    call_log = tmp_path / "pip-calls"

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_hiding_executables(tmp_path, PackageManager.UV)
            ),
            StubVariable.PIP_CALL_LOG: str(call_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_INSTALLED,
        UNINSTALLABLE_REQUIREMENT,
        ENVIRONMENT_DIRECTORY,
    )
    assert call_log.read_text().splitlines() == [
        pip_installer_call(session_start_repository)
    ]


def test_installs_the_package_with_uv_where_there_is_one(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    ``uv`` installs where there is one, since it runs no code of its own inside the
    environment's interpreter, however new that interpreter is.
    """
    require_the_uninstallable(session_start_repository)
    stub_bin.install(PackageManager.UV)
    stub_bin.install(PackageManager.PIP)
    install_log = tmp_path / "uv-calls"
    pip_log = tmp_path / "pip-calls"

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of_inherited_path(),
            StubVariable.UV_CALL_LOG: str(install_log),
            StubVariable.PIP_CALL_LOG: str(pip_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_INSTALLED,
        UNINSTALLABLE_REQUIREMENT,
        ENVIRONMENT_DIRECTORY,
    )
    assert install_log.read_text().splitlines() == [
        uv_installer_call(session_start_repository)
    ]
    assert not pip_log.exists()


def test_reports_a_failed_install_and_finishes_the_run(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    A failing install is reported and the rest of the run still happens.

    The guard the whole design turns on: an install that dies inside the hook takes
    everything after it down with it, and a session then starts with no notes, no plan
    and no explanation.
    """
    require_the_uninstallable(session_start_repository)
    stub_bin.install(PackageManager.PIP)

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_hiding_executables(tmp_path, PackageManager.UV)
            ),
            StubVariable.PIP_STATUS: "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES).startswith(
        summary_message(
            SummaryMessage.DEPENDENCIES_INSTALL_FAILED,
            UNINSTALLABLE_REQUIREMENT,
            ENVIRONMENT_DIRECTORY,
            "",
            "",
        ).split(" - ")[0]
    )
    assert (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()
    assert summary_value(result.stdout, SummaryLine.PERSONAL_NOTES) != ""


def test_reports_that_it_could_not_check_when_the_declaration_will_not_parse(
    session_start_repository: ScratchRepository,
):
    """
    A declaration that cannot be read is reported as such rather than as an environment
    with nothing missing - which is what an empty answer means, and would install
    nothing while saying everything was already there.
    """
    session_start_repository.write(DECLARATION_FILE, "this is not toml\n")
    session_start_repository.commit_everything("break the declaration")

    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_NOT_CHECKED
    )


def test_reports_a_missing_installer_without_failing(
    session_start_repository: ScratchRepository,
    tmp_path: Path,
):
    """
    No installer on PATH at all is the same case as one that fails: reported, and the
    run finishes.
    """
    require_the_uninstallable(session_start_repository)

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: path_hiding_executables(
                tmp_path, PackageManager.PIP, PackageManager.UV
            )
        },
    )

    assert result.returncode == 0, result.stderr
    assert UNINSTALLABLE_REQUIREMENT in summary_value(
        result.stdout, SummaryLine.DEPENDENCIES
    )
    assert (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


# %% creating the package's environment


def without_an_environment(repository: ScratchRepository) -> ScratchRepository:
    """
    Take the package's environment away, as on a clone no session start has set up.

    :param repository: The fixture-built scratch repository.
    :return: The same repository.
    """
    shutil.rmtree(repository.project_root / ProjectLocation.PACKAGE_ENVIRONMENT)
    return repository


def test_creates_the_environment_with_uv_and_installs_the_package_into_it(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    A clone with no environment gets one from ``uv``, asked for an interpreter the
    package supports, with the package installed into it by ``uv`` too.
    """
    without_an_environment(session_start_repository)
    stub_bin.install(PackageManager.UV)
    stub_bin.install(PackageManager.PIP)
    environment_log = tmp_path / "uv-calls"
    install_log = tmp_path / "pip-calls"

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of_inherited_path(),
            StubVariable.UV_CALL_LOG: str(environment_log),
            StubVariable.UV_PYTHON: sys.executable,
            StubVariable.PIP_CALL_LOG: str(install_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_ENVIRONMENT_CREATED,
        ENVIRONMENT_DIRECTORY,
        SOURCE_TREE_ARGUMENT,
    )
    assert environment_log.read_text().splitlines() == [
        environment_call(session_start_repository),
        uv_installer_call(session_start_repository),
    ]
    assert not install_log.exists()


def test_creates_the_environment_from_an_interpreter_found_by_name_without_uv(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    Without ``uv``, the environment is made from an interpreter on the path that the
    package supports - a real one, so the environment is a real one too.
    """
    without_an_environment(session_start_repository)
    stub_bin.install(PackageManager.PIP)

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_hiding_executables(tmp_path, PackageManager.UV)
            ),
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_ENVIRONMENT_CREATED,
        ENVIRONMENT_DIRECTORY,
        SOURCE_TREE_ARGUMENT,
    )
    version = subprocess.run(
        [
            environment_interpreter(session_start_repository),
            "-c",
            "import sys; print(sys.version_info >= (3, 12))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert version.stdout.strip() == str(True)


def path_without_a_supported_interpreter(mirror_parent: Path) -> str:
    """
    :param mirror_parent: Where to build the mirror directories.
    :return: A ``PATH`` offering neither ``uv`` nor any interpreter the environment could
        be created from, as on a machine whose only interpreter is too old.
    """
    return path_hiding_executables(
        mirror_parent,
        PackageManager.UV,
        *configuration_value(ConfigurationVariable.INTERPRETER_CANDIDATES).split(),
    )


def installed_uv_directory(repository: ScratchRepository) -> Path:
    """
    :param repository: The fixture-built scratch repository.
    :return: Where the session start installs ``uv`` for itself.
    """
    return repository.project_root / configuration_value(
        ConfigurationVariable.UV_DIRECTORY
    )


def test_installs_uv_when_no_supported_interpreter_is_on_the_path(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    A machine with no ``uv`` and only an interpreter too old for the package still gets
    an environment: ``uv`` is installed with the ``pip`` there is, and fetches one.
    """
    without_an_environment(session_start_repository)
    stub_bin.install(PackageManager.PIP)
    install_log = tmp_path / "pip-calls"
    environment_log = tmp_path / "uv-calls"

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_without_a_supported_interpreter(tmp_path)
            ),
            StubVariable.PIP_CALL_LOG: str(install_log),
            StubVariable.PIP_INSTALLED_UV: str(
                ExecutableStubDirectory.stub_for(PackageManager.UV)
            ),
            StubVariable.UV_CALL_LOG: str(environment_log),
            StubVariable.UV_PYTHON: sys.executable,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_ENVIRONMENT_CREATED,
        ENVIRONMENT_DIRECTORY,
        SOURCE_TREE_ARGUMENT,
    )
    assert install_log.read_text().splitlines() == [
        f"install --target {installed_uv_directory(session_start_repository)} "
        f"{PackageManager.UV}"
    ]
    assert environment_log.read_text().splitlines() == [
        environment_call(session_start_repository),
        uv_installer_call(session_start_repository),
    ]


def test_reuses_the_uv_it_installed_before(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    The ``uv`` a session start installed is used again rather than installed again.
    """
    without_an_environment(session_start_repository)
    stub_bin.install(PackageManager.PIP)
    installed_uv = (
        installed_uv_directory(session_start_repository) / "bin" / PackageManager.UV
    )
    installed_uv.parent.mkdir(parents=True)
    shutil.copy(ExecutableStubDirectory.stub_for(PackageManager.UV), installed_uv)
    installed_uv.chmod(FileMode.EXECUTABLE)
    pip_log = tmp_path / "pip-calls"
    uv_log = tmp_path / "uv-calls"

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_without_a_supported_interpreter(tmp_path)
            ),
            StubVariable.PIP_CALL_LOG: str(pip_log),
            StubVariable.UV_CALL_LOG: str(uv_log),
            StubVariable.UV_PYTHON: sys.executable,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES) == summary_message(
        SummaryMessage.DEPENDENCIES_ENVIRONMENT_CREATED,
        ENVIRONMENT_DIRECTORY,
        SOURCE_TREE_ARGUMENT,
    )
    assert uv_log.read_text().splitlines() == [
        environment_call(session_start_repository),
        uv_installer_call(session_start_repository),
    ]
    assert not pip_log.exists()


def test_reports_uv_it_could_not_install_and_finishes_the_run(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
    tmp_path: Path,
):
    """
    With no supported interpreter and no way to install ``uv`` - no network, no ``pip``
    - the environment is reported as not created, and the run carries on.
    """
    without_an_environment(session_start_repository)
    stub_bin.install(PackageManager.PIP)

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of(
                path_without_a_supported_interpreter(tmp_path)
            ),
            StubVariable.PIP_STATUS: "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES).startswith(
        summary_message(
            SummaryMessage.DEPENDENCIES_ENVIRONMENT_NOT_CREATED,
            ENVIRONMENT_DIRECTORY,
            "",
        ).split(" - ")[0]
    )
    assert not environment_interpreter(session_start_repository).exists()
    assert (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


def test_reports_an_environment_it_could_not_create_and_finishes_the_run(
    session_start_repository: ScratchRepository,
    stub_bin: ExecutableStubDirectory,
):
    """
    An environment that cannot be made - no supported interpreter, no network for
    ``uv`` to fetch one - is reported, nothing is installed, and the run carries on.
    """
    without_an_environment(session_start_repository)
    stub_bin.install(PackageManager.UV)

    result = publish_and_run(
        session_start_repository,
        environment={
            EnvironmentVariable.EXECUTABLE_SEARCH_PATH: stub_bin.ahead_of_inherited_path(),
            StubVariable.UV_STATUS: "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, SummaryLine.DEPENDENCIES).startswith(
        summary_message(
            SummaryMessage.DEPENDENCIES_ENVIRONMENT_NOT_CREATED,
            ENVIRONMENT_DIRECTORY,
            "",
        ).split(" - ")[0]
    )
    assert not environment_interpreter(session_start_repository).exists()
    assert (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


# %% every message renders


def test_every_summary_message_renders_something():
    """
    Check that every message this module can name resolves to a shell function that
    prints something.

    Deliberately not an assertion about the wording: every other assertion here renders
    its expectation from session-start-messages.sh, so a reword moves both sides at once
    and none of them fail. What is still worth catching is a member naming a function
    that does not exist, or one that prints nothing - neither of which survives here.

    Three placeholder arguments cover the widest message; a shell function ignores the
    ones it does not read.
    """
    for message in SummaryMessage:
        assert summary_message(message, "first", "second", "third").strip() != ""
