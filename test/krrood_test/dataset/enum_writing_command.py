"""
A command that writes one enum member, to show what name the member is written under.

A module run with ``python -m`` is imported as ``__main__``, so an enum declared beside
the command that writes it takes ``__main__`` as its module.
"""

from __future__ import annotations

import json
from enum import StrEnum

from krrood.adapters.json_serializer import to_json

# %% the command


class WrittenByACommand(StrEnum):
    """
    An enum declared in the module that writes it while being run.
    """

    WRITTEN = "written"
    """
    The one member, so that there is something to write.
    """


def main() -> int:
    """
    Write the member to standard output.

    :return: Zero once it is written.
    """
    print(json.dumps(to_json(WrittenByACommand.WRITTEN)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
