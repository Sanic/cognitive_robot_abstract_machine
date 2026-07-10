import os.path
from xml.etree import ElementTree as ET

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    _category_from_class_name,
    _mjcf_document_from_element,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaFixtureResolver,
    RoboCasaObjectResolver,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Cabinet,
    Microwave,
)

ROBOCASA_RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "robocasa",
)


def test_fixture_resolver_resolves_known_category():
    resolver = RoboCasaFixtureResolver()
    assert resolver.resolve("cabinet") is Cabinet
    assert resolver.resolve("MICROWAVE") is Microwave


def test_fixture_resolver_returns_none_for_unknown_category():
    resolver = RoboCasaFixtureResolver()
    assert resolver.resolve("paper_towel_holder") is None


def test_fixture_resolver_resolves_compound_class_name():
    resolver = RoboCasaFixtureResolver()
    assert resolver.resolve("HingeCabinet") is Cabinet
    assert resolver.resolve("hinge_cabinet") is Cabinet


def test_object_resolver_resolves_known_category():
    resolver = RoboCasaObjectResolver()
    assert resolver.resolve("apple") is Apple


def test_object_resolver_returns_none_for_unknown_category():
    resolver = RoboCasaObjectResolver()
    assert resolver.resolve("blender") is None


def test_category_from_class_name_converts_upper_camel_case():
    assert _category_from_class_name("HingeCabinet") == "hinge_cabinet"
    assert _category_from_class_name("Sink") == "sink"


def test_mjcf_document_from_element_wraps_element_in_worldbody():
    fixture_body = ET.fromstring(
        '<body name="hinge_cabinet_main">'
        '<geom type="box" size="0.3 0.3 0.5"/>'
        "</body>"
    )

    document = _mjcf_document_from_element(fixture_body)

    root = ET.fromstring(document)
    assert root.tag == "mujoco"
    assert root.find("worldbody/body").attrib["name"] == "hinge_cabinet_main"


def test_mjcf_document_from_element_is_parseable_by_mjcf_parser():
    fixture_body = ET.fromstring(
        '<body name="hinge_cabinet_main">'
        '<inertial mass="5.0" pos="0 0 0" diaginertia="1 1 1"/>'
        '<geom type="box" size="0.3 0.3 0.5" contype="1" conaffinity="1" group="3"/>'
        "</body>"
    )

    world = MJCFParser.from_xml_string(
        _mjcf_document_from_element(fixture_body)
    ).parse()

    assert world.get_body_by_name("hinge_cabinet_main") is not None


def test_attach_semantic_annotation_uses_fixture_resolver():
    world = MJCFParser(
        os.path.join(ROBOCASA_RESOURCES_DIR, "cabinet_fixture.xml")
    ).parse()
    body = world.get_body_by_name("hinge_cabinet_main")

    loader = RoboCasaDatasetLoader()
    loader._attach_semantic_annotation(world, body, "cabinet")

    annotations = world.get_semantic_annotations_by_type(Cabinet)
    assert len(annotations) == 1
    assert annotations[0].root == body


def test_attach_semantic_annotation_falls_back_to_natural_language():
    world = MJCFParser(
        os.path.join(ROBOCASA_RESOURCES_DIR, "cabinet_fixture.xml")
    ).parse()
    body = world.get_body_by_name("hinge_cabinet_main")

    loader = RoboCasaDatasetLoader()
    loader._attach_semantic_annotation(world, body, "paper_towel_holder")

    annotations = world.get_semantic_annotations_by_type(
        NaturalLanguageWithTypeDescription
    )
    assert len(annotations) == 1
    assert annotations[0].type_description == "paper_towel_holder"


def test_find_body_returns_none_for_missing_body():
    world = MJCFParser(
        os.path.join(ROBOCASA_RESOURCES_DIR, "cabinet_fixture.xml")
    ).parse()
    assert RoboCasaDatasetLoader._find_body(world, "does_not_exist") is None


def test_find_body_falls_back_to_prefix_match():
    world = MJCFParser(
        os.path.join(ROBOCASA_RESOURCES_DIR, "cabinet_fixture.xml")
    ).parse()

    body = RoboCasaDatasetLoader._find_body(world, "hinge_cabinet")

    assert body is not None
    assert body.name.name == "hinge_cabinet_main"


def test_apply_object_semantics_annotates_body_with_collision_not_root():
    world = MJCFParser(
        os.path.join(ROBOCASA_RESOURCES_DIR, "cabinet_fixture.xml")
    ).parse()

    loader = RoboCasaDatasetLoader()
    loader._apply_object_semantics(world, "cabinet")

    annotations = world.get_semantic_annotations_by_type(Cabinet)
    assert len(annotations) == 1
    assert annotations[0].root.name.name == "hinge_cabinet_main"
    assert annotations[0].root != world.root
