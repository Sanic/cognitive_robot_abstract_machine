"""
Fixtures shared by every suite in this directory.

basstler is imported from its editable install, which is this clone's own source under
basstler/src, as CI installs it and as a session start installs it into basstler/.venv.

This suite runs in the lightweight test_basstler CI job with --confcutdir pointed
here, so the repository-root test/conftest.py - which imports the robotics stack that
job does not install - is never loaded for it.
"""

from pathlib import Path

import pytest


from .executable_stubs import ExecutableStubDirectory
from .scratch_repository import ScratchRepository
from .upstream_reviews_replay import RecordedResponse, ReplayingClient


@pytest.fixture
def scratch_repository(tmp_path: Path) -> ScratchRepository:
    """
    An initialized scratch project root and its bare notes remote, with nothing
    committed and no personal-notes branch published yet.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch repository.
    """
    return ScratchRepository.create(tmp_path)


@pytest.fixture
def stub_bin(tmp_path: Path) -> ExecutableStubDirectory:
    """
    An empty directory meant to be placed first on a test subprocess's PATH, into which
    a test installs whichever stubbed executable it needs.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The stub directory.
    """
    return ExecutableStubDirectory.create(tmp_path)


@pytest.fixture
def paginated_client() -> ReplayingClient:
    """:return: A client replaying both pages of the recorded review threads."""
    return ReplayingClient(
        [
            RecordedResponse.PULL_REQUEST_PAGE_ONE.load(),
            RecordedResponse.PULL_REQUEST_PAGE_TWO.load(),
        ]
    )
