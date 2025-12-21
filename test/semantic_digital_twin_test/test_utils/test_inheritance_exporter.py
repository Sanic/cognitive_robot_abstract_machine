import json
from pathlib import Path
from typing_extensions import Any
from abc import ABC

import pytest

from semantic_digital_twin.utils import InheritanceStructureExporter


# --- Test-only class hierarchy -------------------------------------------------


class Root:
    pass


class ExternalBase:
    def __init__(self, x: float, y: int = 1, _z: int = 2, *args, **kwargs):
        pass


class AbstractChild(Root, ABC):
    # No required public fields
    pass


class ConcreteChild(Root, ExternalBase):
    def __init__(
        self,
        name: str,
        value: int,
        flag: bool = False,
        _hidden: str = "x",
        *args,
        **kwargs,
    ):
        # Intentionally do nothing
        pass


# --- Helper --------------------------------------------------------------------


def make_exporter(tmp_path: Path | None = None) -> InheritanceStructureExporter:
    if tmp_path is None:
        return InheritanceStructureExporter(root_class=Root)
    return InheritanceStructureExporter(
        root_class=Root, output_path=tmp_path / "out.json"
    )


# --- Unit tests for individual methods -----------------------------------------


def test_is_inheriting_from_abc():
    exp = make_exporter()
    assert exp._is_inheriting_from_abc(AbstractChild) is True
    assert exp._is_inheriting_from_abc(ConcreteChild) is False


def test_walk_related_classes_subclasses_and_bases():
    exp = make_exporter()

    # Only immediate subclasses are returned
    subs = list(exp._walk_related_classes(Root, relation="subclasses"))
    assert set(sub.__name__ for sub in subs) == {"AbstractChild", "ConcreteChild"}

    # For ConcreteChild the only non-root, non-ABC base is ExternalBase
    bases = list(exp._walk_related_classes(ConcreteChild, relation="bases"))
    assert bases == [ExternalBase]

    # Unsupported relation should raise
    with pytest.raises(ValueError):
        list(exp._walk_related_classes(Root, relation="unknown"))


def test_type_to_string_variants():
    exp = make_exporter()

    # None annotation becomes Any object (not a string)
    assert exp._type_to_string(None) is Any

    # Forward reference string
    assert exp._type_to_string("World") == "World"

    # Builtins and generics
    assert exp._type_to_string(int) == "int"
    assert exp._type_to_string(list[int]) == "list[int]"
    assert exp._type_to_string(dict[str, int]) == "dict[str, int]"

    # Custom class
    assert exp._type_to_string(ExternalBase) == "ExternalBase"


def test_get_only_required_public_fields_filters_correctly():
    import inspect

    exp = make_exporter()
    sig = inspect.signature(ConcreteChild.__init__)
    init_ann = getattr(ConcreteChild.__init__, "__annotations__", {}) or {}
    class_ann = getattr(ConcreteChild, "__annotations__", {}) or {}

    # Collect by invoking the private helper directly
    results = []
    for name, param in sig.parameters.items():
        item = exp._get_only_required_public_fields(name, param, init_ann, class_ann)
        if item:
            results.append(item)

    # Only required, public, non-vararg parameters make it through
    assert results == [
        {"name": "name", "type": "str"},
        {"name": "value", "type": "int"},
    ]


def testcollect_required_public_fields_aggregates_from_init():
    exp = make_exporter()
    fields_concrete = exp.collect_required_public_fields(ConcreteChild)
    assert fields_concrete == [
        {"name": "name", "type": "str"},
        {"name": "value", "type": "int"},
    ]

    fields_abstract = exp.collect_required_public_fields(AbstractChild)
    assert fields_abstract == []

    # ExternalBase has one required public param (x), others filtered out
    fields_external = exp.collect_required_public_fields(ExternalBase)
    assert fields_external == [{"name": "x", "type": "float"}]


def test_build_node_structure_with_and_without_subclasses():
    exp = make_exporter()

    # Without subclasses key
    node_abs = exp._build_node(AbstractChild, include_subclasses=False)
    assert node_abs["name"] == "AbstractChild"
    assert node_abs["is_abstract"] is True
    assert node_abs["other_superclasses"] == []  # Root filtered out, ABC excluded
    assert node_abs["fields"] == []
    assert "subclasses" not in node_abs

    # With subclasses key and external base rendered as a node (without its own subclasses)
    node_conc = exp._build_node(ConcreteChild, include_subclasses=True)
    assert node_conc["name"] == "ConcreteChild"
    assert node_conc["is_abstract"] is False
    assert node_conc["fields"] == [
        {"name": "name", "type": "str"},
        {"name": "value", "type": "int"},
    ]

    # Validate the external base is included properly
    other_bases = node_conc["other_superclasses"]
    assert len(other_bases) == 1
    base_node = other_bases[0]
    assert base_node["name"] == "ExternalBase"
    assert base_node["is_abstract"] is False
    assert base_node["other_superclasses"] == []
    assert base_node["fields"] == [{"name": "x", "type": "float"}]
    assert "subclasses" not in base_node


def test_build_inheritance_structure_from_root():
    exp = make_exporter()
    data = exp._build_inheritance_structure()

    assert data["root_name"] == "Root"

    # Compare subclasses by name to avoid ordering sensitivity
    names = sorted(node["name"] for node in data["subclasses"])
    assert names == ["AbstractChild", "ConcreteChild"]


def test_export_writes_json(tmp_path):
    exp = make_exporter(tmp_path)
    exp.export()

    out_path = tmp_path / "out.json"
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload["root_name"] == "Root"
    names = sorted(node["name"] for node in payload["subclasses"])
    assert names == ["AbstractChild", "ConcreteChild"]
