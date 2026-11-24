"""
Browser state follows shared agent changes without writing another history action.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path
import subprocess


# %% Revision notifications
def test_browser_refreshes_agent_changes_after_local_operations_finish(
    javascript_runtime: Path,
) -> None:
    """
    A busy local operation defers the notification without losing the pending revision.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/agent_sync.cjs"),
            str(root / "scan_refiner/web/jobs.js"),
        ],
        check=True,
    )


def test_clipboard_handoff_uses_the_displayed_revision_and_oriented_region(
    javascript_runtime: Path,
) -> None:
    """
    The clipboard and its manual fallback describe the same scoped evidence.
    """
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            str(javascript_runtime),
            str(root / "tests/web/investigation.cjs"),
            str(root / "scan_refiner/web/investigation.js"),
        ],
        check=True,
    )
