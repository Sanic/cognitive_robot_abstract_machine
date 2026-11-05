"""
What anything in the Warsaw pipeline can be, over and above what it does.

Two things are wanted almost everywhere and belong to no one step: saying what is being
done, and being written to a run's files. Each is a handful of lines, and a module
apiece made the two look like separate concerns rather than the small shared ones they
are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from krrood.adapters.json_field import JSONField
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)
from krrood.utils import get_full_class_name
from typing_extensions import Any, Dict, Self

# %% saying what is being done


@dataclass
class HasLogger:
    """
    Something that says what it is doing.

    A run is minutes of work over seven steps, and what it says as it goes is the only
    account of it until the report is written at the end. That account goes to logging
    rather than to standard output, so a caller decides where it lands and a step does
    not.
    """

    @property
    def logger(self) -> logging.Logger:
        """
        :return: Where this says what it is doing, named for the module it is declared in
            so a caller can quieten one part of a run without quietening the rest.
        """
        return logging.getLogger(type(self).__module__)


# %% being written to a run's files


class JsonRecord(SubclassJSONSerializer):
    """
    Something a step writes into a run's directory, or reads out of a model's reply.

    Every step reads what the step before it wrote, so each field name is written in one
    module and read in another. Mirroring each file in a dataclass writes those names
    once: the reader and the writer are the same declaration, and a field that moves
    moves for both. What reads and writes it is krrood's serializer, working from the
    fields themselves, so the mapping is not spelled out anywhere.

    A record reaches this class from two sides. A file the pipeline wrote carries the name
    of the class that wrote it, so reading it back needs nothing but the file. A model's
    reply carries no such name -- it was written by something that has never heard of these
    classes -- so the class it is read on is taken to be the class it is.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The record as JSON-ready data, with the name of its class.
        """
        # SubclassJSONSerializer.to_json writes the class name and nothing else; the
        # fields are what DataclassJSONSerializer knows how to walk.
        written = DataclassJSONSerializer.to_json(self)
        written[JSONField.TYPE] = get_full_class_name(type(self))
        return written

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Build the record once the class to build has been settled.

        :param data: The record as JSON-ready data.
        :param kwargs: Passed on to the records held inside this one.
        :return: The record.
        """
        return DataclassJSONSerializer.from_json(data, clazz=cls, **kwargs)

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Read a record, from a file the pipeline wrote or from what a model answered.

        :param data: The record as JSON-ready data.
        :param kwargs: Passed on to the records held inside this one.
        :return: The record.
        """
        if isinstance(data, dict) and JSONField.TYPE not in data:
            data = {**data, JSONField.TYPE: get_full_class_name(cls)}
        return super().from_json(data, **kwargs)
