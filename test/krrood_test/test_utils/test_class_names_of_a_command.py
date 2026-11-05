"""
Reading back what a module run with ``python -m`` wrote.

Such a module is imported as ``__main__``, and a class name taken from there names a
module nothing can import, so whatever was written under it can never be read again.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import from_json

from ..dataset import enum_writing_command
from ..dataset.enum_writing_command import WrittenByACommand

# %% a command that writes an enum member


def written_by_the_command() -> dict:
    """
    Run the command and read what it wrote.
    """
    finished = subprocess.run(
        [sys.executable, "-m", enum_writing_command.__name__],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(finished.stdout)


def test_a_class_is_named_after_the_module_the_command_was_run_as():
    assert written_by_the_command()[JSONField.TYPE] == (
        f"{enum_writing_command.__name__}.{WrittenByACommand.__name__}"
    )


def test_what_a_command_wrote_can_be_read_back():
    assert from_json(written_by_the_command()) is WrittenByACommand.WRITTEN
