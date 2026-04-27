"""Validation helpers for RoboKudo benchmark specs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.resources as resources


SCHEMA_RESOURCE = "schemas/benchmark_spec.v1.schema.json"


@dataclass(frozen=True)
class SpecValidationError:
    """Structured validation error for machine-readable feedback."""

    path: str
    message: str
    expected: str | None = None
    actual: str | None = None


class BenchmarkSpecError(ValueError):
    """Raised when a benchmark spec is invalid."""

    def __init__(self, errors: list[SpecValidationError]):
        self.errors = errors
        super().__init__("\n".join(_format_error(e) for e in errors))


def _format_error(error: SpecValidationError) -> str:
    base = f"{error.path}: {error.message}"
    if error.expected is None and error.actual is None:
        return base
    return (
        f"{base} "
        f"(expected={error.expected or 'n/a'}, actual={error.actual or 'n/a'})"
    )


def _load_schema() -> dict[str, Any]:
    schema_path = resources.files("robokudo.benchmarking").joinpath(SCHEMA_RESOURCE)
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_json_schema(spec: dict[str, Any]) -> list[SpecValidationError]:
    try:
        import jsonschema
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency expected in benchmark env
        raise RuntimeError(
            "jsonschema is required for benchmark spec validation. "
            "Install it in your environment."
        ) from exc

    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema=schema)

    errors: list[SpecValidationError] = []
    for error in sorted(validator.iter_errors(spec), key=lambda x: str(x.path)):
        json_path = "$"
        for segment in error.path:
            json_path += f"[{segment!r}]"
        errors.append(
            SpecValidationError(
                path=json_path,
                message=error.message,
                expected=str(error.validator_value),
                actual=_json_type_name(error.instance),
            )
        )
    return errors


def _validate_semantics(spec: dict[str, Any]) -> list[SpecValidationError]:
    errors: list[SpecValidationError] = []

    samples = spec.get("samples", [])
    if samples:
        seen: set[int] = set()
        for index, sample in enumerate(samples):
            sample_index = sample.get("sample_index")
            if sample_index in seen:
                errors.append(
                    SpecValidationError(
                        path=f"$.samples[{index}].sample_index",
                        message=f"duplicate sample_index={sample_index}",
                    )
                )
            else:
                seen.add(sample_index)

    return errors


def validate_spec_dict(spec: dict[str, Any]) -> None:
    """Validate a benchmark spec dictionary.

    :raises BenchmarkSpecError: when validation fails
    """
    errors = []
    errors.extend(_validate_json_schema(spec))
    errors.extend(_validate_semantics(spec))
    if errors:
        raise BenchmarkSpecError(errors)


def load_spec_file(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_spec_file(path: str | Path) -> None:
    """Validate a benchmark spec file."""
    spec = load_spec_file(path)
    validate_spec_dict(spec)
