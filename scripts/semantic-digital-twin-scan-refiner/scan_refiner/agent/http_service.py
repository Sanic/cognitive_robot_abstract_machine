"""
Agent requests dispatched through the existing UI server and project action lock.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from json import JSONDecodeError
from pydantic import ValidationError
from urllib.parse import urlsplit
from scan_refiner.agent.models import AgentFile, AgentError
from scan_refiner.agent.registry import AgentRegistry
from scan_refiner.agent.codec import AgentCodec
from scan_refiner.source_service import SourceRequestHandler
from scan_refiner.exceptions import InvalidEvidence


# %% Agent resources
class AgentRoute(StrEnum):
    """
    Project-local tool calls and bounded render artifacts.
    """

    OPERATIONS = "/api/agent/"
    REVISION = "/api/agent/revision"
    IMAGES = "/api/agent/images/"


@dataclass(kw_only=True, init=False)
class AgentRequestHandler(SourceRequestHandler):
    """
    Use the same worker owner and journal lock as human UI actions.
    """

    def serve_agent_get(self) -> bool:
        """
        Expose revision notifications and fixed-name PNGs, without arbitrary path reads.
        """
        route = urlsplit(self.path).path
        if route == AgentRoute.REVISION:
            self.send_payload(
                payload=AgentCodec.encode_record(
                    record=self.server.agent.revision_state()
                ),
                content_type="application/json",
            )
            return True
        if not route.startswith(AgentRoute.IMAGES):
            return False
        identifier = route[len(AgentRoute.IMAGES) :]
        if len(identifier) != 32 or any(
            character not in "0123456789abcdef" for character in identifier
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        path = (
            self.server.agent.artifact_directory(
                kind=AgentFile.RENDERS, identifier=identifier
            )
            / AgentFile.IMAGE
        )
        if not path.is_file() or path.stat().st_size > 10000000:
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        self.send_payload(payload=path.read_bytes(), content_type="image/png")
        return True

    def serve_agent_post(self) -> bool:
        """
        Decode only a declared DTO before invoking a revision-checked operation.
        """
        route = urlsplit(self.path).path
        if not route.startswith(AgentRoute.OPERATIONS):
            return False
        operation = route[len(AgentRoute.OPERATIONS) :]
        binding = next(
            (
                entry
                for entry in AgentRegistry(implementation=self.server.agent).bindings
                if entry.operation == operation
            ),
            None,
        )
        if binding is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        origin = self.headers.get("Origin")
        length = self.headers.get("Content-Length", "0")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.send_error(HTTPStatus.FORBIDDEN)
            return True
        if not length.isdecimal() or int(length) > 131072:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return True
        payload = self.rfile.read(int(length))
        with self.server.action_lock:
            try:
                request = AgentCodec().decode_request(
                    payload=payload, request_type=binding.request_type
                )
                result = binding.handler(request=request)
            except (
                JSONDecodeError,
                UnicodeDecodeError,
                ValidationError,
                InvalidEvidence,
            ) as error:
                status = (
                    HTTPStatus.CONFLICT
                    if isinstance(error, InvalidEvidence)
                    else HTTPStatus.BAD_REQUEST
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                response = AgentCodec.encode_record(
                    record=AgentError(
                        message=str(error),
                        error_type=type(error).__name__,
                        current_revision=self.server.agent.revision,
                    )
                )
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                return True
        self.send_payload(
            payload=AgentCodec.encode_record(record=result),
            content_type="application/json",
        )
        return True
