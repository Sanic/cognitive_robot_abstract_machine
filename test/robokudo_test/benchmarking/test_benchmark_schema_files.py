from pathlib import Path

import pytest

from robokudo.benchmarking.validator import validate_spec_file


def test_example_specs_validate() -> None:
    pytest.importorskip("jsonschema")

    repo_root = Path(__file__).resolve().parents[3]
    benchmark_root = repo_root / "robokudo" / "benchmarks" / "suites"

    specs = sorted(benchmark_root.glob("*/spec.json"))
    assert specs, "No example benchmark specs found."

    for spec_path in specs:
        validate_spec_file(spec_path)
