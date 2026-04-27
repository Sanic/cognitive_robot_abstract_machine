"""CLI entrypoints for RoboKudo benchmarking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robokudo.benchmarking.validator import BenchmarkSpecError, validate_spec_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a benchmark spec file."
    )
    validate_parser.add_argument("spec_path", type=Path)

    subparsers.add_parser(
        "run",
        help=("Run a benchmark suite. " "Not implemented yet in this scaffold."),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            validate_spec_file(args.spec_path)
        except BenchmarkSpecError as exc:
            print("SPEC_INVALID")
            for error in exc.errors:
                print(
                    f"- path={error.path} message={error.message} "
                    f"expected={error.expected} actual={error.actual}"
                )
            return 2
        except Exception as exc:  # pragma: no cover - operational failures
            print(f"VALIDATION_ERROR: {exc}")
            return 1

        print("SPEC_VALID")
        return 0

    if args.command == "run":
        print(
            "RUN_NOT_IMPLEMENTED: this scaffold currently supports schema validation only."
        )
        return 3

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
