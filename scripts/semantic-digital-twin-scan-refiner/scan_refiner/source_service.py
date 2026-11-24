"""
Local reviewed source application with bounded reload and export processes.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from http import HTTPStatus
from urllib.parse import urlsplit
import os
import subprocess
import sys
import json
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.source_application import SourceReview, SourceResult
from scan_refiner.exceptions import InvalidEvidence
from scan_refiner.job_service import JobRequestHandler


# %% HTTP vocabulary
class SourceRoute(StrEnum):
    """
    Explicit review and publication actions beneath the loopback service.
    """

    REVIEW = "/api/source/review"
    PUBLISH = "/api/source/publish"


# %% Bounded source reload
@dataclass(kw_only=True)
class SourceProcess:
    """
    Run source validation and mesh conversion within the standalone task's memory limit.
    """

    directory: Path
    """
    Project containing the accepted geometry and immutable source reviews.
    """

    timeout_seconds: float = 240
    """
    Maximum elapsed time for a single source operation.
    """

    def run_source_action(
        self, *, publish: bool, identifier: str = ""
    ) -> subprocess.CompletedProcess[str]:
        """
        Return the task's typed JSON output or its diagnostic process result.
        """
        command = [
            sys.executable,
            str(Path(__file__).parents[1] / "source_task.py"),
            "--project",
            str(self.directory),
            "--action",
            "publish" if publish else "review",
        ]
        if identifier:
            command += ["--review", identifier]
        environment = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            MPLCONFIGDIR="/tmp/scan-refiner-matplotlib",
        )
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=self.timeout_seconds,
            check=False,
        )

    def request_source_action(
        self, *, publish: bool, identifier: str = ""
    ) -> SourceReview | SourceResult:
        """
        Restore the task's declared result, rejecting a failed source validation.
        """
        completed = self.run_source_action(publish=publish, identifier=identifier)
        if completed.returncode:
            raise InvalidEvidence(
                reason="Source validation failed. " + completed.stderr[-2000:]
            )
        return DataclassJSONSerializer.from_json(
            data=json.loads(completed.stdout),
            clazz=SourceResult if publish else SourceReview,
        )


@dataclass(kw_only=True, init=False)
class SourceRequestHandler(JobRequestHandler):
    """
    Add source-copy actions without changing the existing inspection/job routes.
    """

    def serve_source_post(self) -> bool:
        """
        Run a bounded source action only for a same-origin, project-local request.
        """
        route = urlsplit(self.path).path
        if route not in list(SourceRoute):
            return False
        origin = self.headers.get("Origin")
        length = self.headers.get("Content-Length", "0")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_error(HTTPStatus.FORBIDDEN)
            return True
        if not length.isdecimal() or int(length) > 1024:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return True
        identifier = ""
        if route == SourceRoute.PUBLISH:
            identifier = self.rfile.read(int(length)).decode().strip()
            if len(identifier) != 32 or any(
                character not in "0123456789abcdef" for character in identifier
            ):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return True
        with self.server.action_lock:
            completed = SourceProcess(directory=Path(self.directory)).run_source_action(
                publish=route == SourceRoute.PUBLISH, identifier=identifier
            )
        if completed.returncode:
            self.send_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                completed.stderr.splitlines()[-2]
                if len(completed.stderr.splitlines()) > 1
                else "Source validation failed. See the source and reviewed geometry.",
            )
            return True
        self.send_payload(
            payload=completed.stdout.encode(), content_type="application/json"
        )
        return True
