import io
from pathlib import Path

import pytest

from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
    SpecialFieldTypes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


def _builder(name: str = "MyAnnotation") -> SemanticAnnotationClassBuilder:
    return SemanticAnnotationClassBuilder(
        name=name,
        template_name="dataclass_template.py.jinja",
    )


def test_add_field_appends_and_handles_no_default():
    b = _builder().add_base(SemanticAnnotation)

    # add one required and one with explicit default (including None allowed)
    b.add_field("age", int)
    b.add_field("label", str, default="foo")
    b.add_field("note", str, default=None)

    assert [f[0] for f in b.fields] == ["age", "label", "note"]
    assert b.fields[0][2] is SpecialFieldTypes.NO_DEFAULT
    assert b.fields[1][2] == "foo"
    assert b.fields[2][2] is None


def test_add_method_adds_to_namespace_and_built_class_has_method():
    def greet(self):
        return f"hello {self.__class__.__name__}"

    b = _builder().add_base(SemanticAnnotation)
    b.add_method("greet", greet)

    assert "greet" in b.namespace

    Cls = b.build()
    # required kw-only fields are none yet, so instantiation works directly
    obj = Cls()
    assert obj.greet() == f"hello {Cls.__name__}"


def test_add_base_accumulates_bases():
    b = _builder()
    b.add_base(SemanticAnnotation)
    b.add_base(object)
    assert SemanticAnnotation in b.bases
    assert object in b.bases
    assert isinstance(b.bases, tuple)


def test_build_requires_semantic_annotation_base():
    b = _builder()
    b.add_field("name", str, default="x")
    with pytest.raises(TypeError):
        _ = b.build()


def test_build_creates_dataclass_with_kwonly_required_fields():
    b = _builder("BuiltAnno").add_base(SemanticAnnotation)
    b.add_field("age", int)  # required
    b.add_field("label", str, default="foo")

    Cls = b.build()

    # dataclass settings
    assert getattr(Cls, "__dataclass_params__").eq is False

    # must pass required field as keyword-only
    with pytest.raises(TypeError):
        _ = Cls(10)  # positional not allowed due to kw_only

    with pytest.raises(TypeError):
        _ = Cls()  # missing required field

    instance = Cls(age=10)
    assert instance.age == 10
    assert instance.label == "foo"


def test_fields_for_template_representation():
    b = _builder("TmplAnno").add_base(SemanticAnnotation)
    b.add_field("age", int)
    b.add_field("label", str, default="foo")
    b.add_field("maybe", object, default=None)

    payload = b._fields_for_template()
    # Maintain order and mapping
    assert [d["name"] for d in payload] == ["age", "label", "maybe"]
    assert payload[0]["type_hint"] == "int"
    assert payload[0]["default"] is None  # NO_DEFAULT becomes None for template
    assert payload[1]["type_hint"] == "str" and payload[1]["default"] == "'foo'"
    # object has a __name__ we expect to be 'object'
    assert payload[2]["type_hint"] == "object" and payload[2]["default"] == "None"


def test_render_source_contains_expected_content():
    b = _builder("RenderAnno").add_base(SemanticAnnotation)
    b.add_field("age", int)
    b.add_field("label", str, default="foo")

    src_no_imports = b.render_source(include_imports=False)
    src_with_imports = b.render_source(include_imports=True)

    # Current template ignores include_imports, but both should be non-empty strings
    assert isinstance(src_no_imports, str) and src_no_imports.strip() != ""
    assert isinstance(src_with_imports, str) and src_with_imports.strip() != ""

    # Check class header and fields rendered with defaults
    assert "@dataclass" in src_no_imports
    assert "class RenderAnno(" in src_no_imports and "SemanticAnnotation" in src_no_imports
    assert "age: int" in src_no_imports
    assert "label: str = 'foo'" in src_no_imports


def test_append_to_file_appends_source(tmp_path: Path):
    b1 = _builder("A1").add_base(SemanticAnnotation)
    b1.add_field("x", int, default=1)
    b2 = _builder("A2").add_base(SemanticAnnotation)
    b2.add_field("y", int)

    filepath = tmp_path / "out.py"
    b1.append_to_file(filepath)
    b2.append_to_file(filepath)

    content = filepath.read_text(encoding="utf-8")
    # Each append prefixes with two newlines and ends with a single newline
    assert content.startswith("\n\n@dataclass")
    assert "class A1(" in content and "x: int = 1" in content
    assert "class A2(" in content and "y: int" in content
