"""Strict transport records using the repository serializer without arbitrary type imports."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from typing_extensions import TypeVar
from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import DataclassJSONSerializer
from scan_refiner.agent import models
from scan_refiner.dimensions import DimensionConstraints
from scan_refiner.refinement import TranslationConstraints
from scan_refiner.registration import RegistrationSearch
from scan_refiner.exceptions import InvalidEvidence


# %% Typed records
RecordType = TypeVar("RecordType")
"""Expected operation input selected by the server, never by a request type tag."""


@dataclass(kw_only=True)
class AgentCodec:
    """Whitelist DTOs and fields before invoking the existing dataclass decoder."""

    @staticmethod
    def encode_record(*, record: object) -> bytes:
        """Serialize repository-owned records through their shared serializer."""
        return json.dumps(DataclassJSONSerializer.to_json(instance=record)).encode()

    def decode_request(
        self, *, payload: bytes, request_type: type[RecordType]
    ) -> RecordType:
        """Reject unknown record/collection types and unsupported fields before deserialization."""
        data = json.loads(payload)
        if (
            not isinstance(data, dict)
            or data.get(JSONField.TYPE)
            != request_type.__module__ + "." + request_type.__qualname__
        ):
            raise InvalidEvidence(
                reason="Use the typed agent client for this operation request."
            )
        allowed = [
            kind
            for kind in vars(models).values()
            if isinstance(kind, type)
            and kind.__module__ == models.__name__
            and (is_dataclass(kind) or issubclass(kind, Enum))
        ]
        allowed.extend(
            [DimensionConstraints, TranslationConstraints, RegistrationSearch]
        )
        known = {kind.__module__ + "." + kind.__qualname__: kind for kind in allowed}
        self.validate_type_tags(value=data, known=known)
        return DataclassJSONSerializer.from_json(data=data, clazz=request_type)

    def validate_type_tags(self, *, value: object, known: dict[str, type]) -> None:
        """Validate only serializer metadata; the existing serializer restores actual records."""
        if isinstance(value, list):
            for item in value:
                self.validate_type_tags(value=item, known=known)
            return
        if not isinstance(value, dict):
            return
        if JSONField.IS_CLASS in value:
            raise InvalidEvidence(reason="Class objects are not agent request data.")
        if JSONField.COLLECTION_TYPE in value:
            if value[JSONField.COLLECTION_TYPE] != "builtins.list" or set(value) != {
                JSONField.COLLECTION_TYPE,
                JSONField.ITEMS,
            }:
                raise InvalidEvidence(
                    reason="Only declared list collections are accepted in agent requests."
                )
        if JSONField.TYPE in value:
            identity = value[JSONField.TYPE]
            if identity not in known:
                raise InvalidEvidence(reason="Unknown serialized agent request type.")
            kind = known[identity]
            if issubclass(kind, Enum) and value.get("name") not in kind.__members__:
                raise InvalidEvidence(reason="Unknown agent request enum member.")
            permitted = (
                {item.name for item in fields(kind)} | {str(JSONField.TYPE)}
                if is_dataclass(kind)
                else {str(JSONField.TYPE), "name"}
            )
            if set(value) - permitted:
                raise InvalidEvidence(
                    reason="Unsupported request or constraint fields must not be ignored."
                )
        for item in value.values():
            self.validate_type_tags(value=item, known=known)
