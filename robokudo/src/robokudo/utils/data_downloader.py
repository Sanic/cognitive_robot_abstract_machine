from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from pathlib import Path

import pooch
from filelock import FileLock


# %% Test dataset
@dataclass(frozen=True)
class TestDataset:
    """
    Identify the downloadable RoboKudo test dataset.
    """

    package_name: str = "robokudo_test_data"
    """
    Dataset archive and cache name.
    """

    revision: str = "52fedaf2f2e56f52a78eb9ea0a64a15d53a7294b"
    """
    Repository revision packaged in the archive.
    """

    checksum: str | None = (
        "sha256:336c328d324f8c0023f924cb6fd0808aaabce8b11270597db427ebde333b3dd6"
    )
    """
    Expected archive checksum.
    """


def test_data_path() -> Path:
    """
    Retrieve the extracted dataset directory, downloading it when absent.
    """
    dataset = TestDataset()
    cache_directory = Path(pooch.os_cache(dataset.package_name))
    cache_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(cache_directory / f"{dataset.package_name}.zip.lock"):
        extracted_files = pooch.retrieve(
            url=(
                "https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_test_data/-/jobs/artifacts/"
                f"{dataset.revision}/raw/{dataset.package_name}-{dataset.revision}.zip?job=package_zip"
            ),
            known_hash=dataset.checksum,
            path=cache_directory,
            fname=f"{dataset.package_name}.zip",
            downloader=pooch.HTTPDownloader(),
            processor=pooch.Unzip(),
        )
    return Path(extracted_files[0]).parent
