from pathlib import Path
import pooch

# ---- Configuration ----
DATA_PACKAGE_NAME: str = "robokudo_test_data"
DATA_VERSION: str = "52fedaf2f2e56f52a78eb9ea0a64a15d53a7294b"
KNOWN_HASH: str = (
    "sha256:336c328d324f8c0023f924cb6fd0808aaabce8b11270597db427ebde333b3dd6"
)

URL: str = (
    "https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_test_data/-/jobs/artifacts/"
    f"{DATA_VERSION}/raw/robokudo_test_data-{DATA_VERSION}.zip?job=package_zip"
)

FILENAME: str = f"{DATA_PACKAGE_NAME}.zip"


def test_data_path() -> Path:
    """
    Retrieve Robokudo test data, downloading and unpacking if needed.

    :return: Path to the root extracted dataset directory in the local cache.

    .. warning::
       This function performs network I/O on first use.
    """
    downloader = pooch.HTTPDownloader()

    # Download + verify + unzip
    extracted_files = pooch.retrieve(
        url=f"{URL}",
        known_hash=KNOWN_HASH,
        path=pooch.os_cache(DATA_PACKAGE_NAME),  # ~/.cache/DATA_PACKAGE_NAME
        fname=FILENAME,
        downloader=downloader,
        processor=pooch.Unzip(),
    )

    # Pooch returns list of extracted file paths
    # We usually want the root extracted directory:
    extracted_dir = Path(extracted_files[0]).parents[0]

    return extracted_dir
