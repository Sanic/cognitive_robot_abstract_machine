"""
Building the classes a scene needs that the ontology does not have.

Two steps proposed compositions and only one of them was asked to. The vocabulary step
answers what a *label* means and names the mixins to compose it from, having been shown
what objects of that label were measured to meet; the classification step answers which
class each *object* is, from a picture, and its schema carries a superclass as well.
Read from the classification alone, a Faucet the vocabulary had composed with HasHandle
comes out with no way to hold a handle, and the pairing measured for it cannot be
mounted.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from dataclasses import replace

import pytest

from experiments.warsaw.pipeline.records import (
    BodyAnswer,
    Classifications,
    LabelAnswer,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.templates import PipelineTemplates
from experiments.warsaw.pipeline.run_classes import GeneratedClasses
from experiments.warsaw.pipeline.settings import PipelineSettings
from experiments.warsaw.pipeline.steps.annotate import AnnotateAndMount
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    compose_class,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@pytest.fixture
def known(taxonomy):
    """
    The ontology as a run finds it, which is the ontology as it was read out.

    Not every subclass this interpreter happens to hold: composing a proposed class
    registers it as a subclass for the rest of the process, so a step asked what the
    ontology has would be told about classes another run proposed. A run meets a fresh
    interpreter; a test has to say so.

    :return: The ontology's classes by name.
    """
    declared = {node["name"] for node in taxonomy["classes"]}
    return {
        name: annotation_class
        for name, annotation_class in annotation_classes(SemanticAnnotation).items()
        if name in declared
    }


@pytest.fixture
def step(tmp_path):
    """
    :return: The annotating step, for the classes it builds.
    """
    return AnnotateAndMount(settings=PipelineSettings(), run=Run(directory=tmp_path))


def classified(**bodies) -> Classifications:
    """
    :param bodies: What each body was answered to be.
    :return: Those answers, as the classification step wrote them.
    """
    return Classifications(
        model="",
        scene="",
        bodies=[replace(answer, name=name) for name, answer in bodies.items()],
    )


def answered(**labels) -> Vocabulary:
    """
    :param labels: What each label was answered to mean.
    :return: Those answers, as the vocabulary step wrote them.
    """
    return Vocabulary(
        model="",
        scene="",
        labels=[replace(answer, label=label) for label, answer in labels.items()],
    )


# %% what a class is built from


def test_a_class_the_vocabulary_composed_keeps_its_composition(step):
    """
    The mixins are what let the class hold the parts measured for it, and the
    classification's schema carries no mixins at all.
    """
    wanted = step.wanted_classes(
        classified(
            faucet_1=BodyAnswer(
                class_name="Faucet", is_new_class=True, superclass="Furniture"
            )
        ),
        answered(
            faucet=LabelAnswer(
                class_name="Faucet",
                is_new_class=True,
                superclass="Fixture",
                mixins=["HasHandle"],
            )
        ),
    )
    assert wanted["Faucet"] == ["Fixture", "HasHandle"]


def test_a_class_only_the_classification_proposed_is_built_from_its_superclass(step):
    """
    A body the split left that is no longer what its label said still needs a class.
    """
    wanted = step.wanted_classes(
        classified(
            ceiling_1=BodyAnswer(
                class_name="Ceiling", is_new_class=True, superclass="Floor"
            )
        ),
        answered(),
    )
    assert wanted["Ceiling"] == ["Floor"]


def test_a_class_proposed_with_no_superclass_falls_back_to_the_root(step):
    """
    Every annotation is a SemanticAnnotation, so a proposal that names nothing else is
    still buildable.
    """
    wanted = step.wanted_classes(
        classified(thing_1=BodyAnswer(class_name="Thing", is_new_class=True)),
        answered(),
    )
    assert wanted["Thing"] == ["SemanticAnnotation"]


def test_a_body_with_no_class_asks_for_nothing(step):
    """
    A body nobody named gets no annotation, and no class has to exist for it.
    """
    assert step.wanted_classes(classified(thing_1=BodyAnswer()), answered()) == {}


# %% which of them are actually generated


def test_a_class_the_ontology_already_has_is_not_generated(step, known):
    """
    Generating it again would put a second Cabinet in front of the ORM.
    """
    assert step.generate_missing({"Cabinet": ["Furniture"]}, known) == []
    assert not GeneratedClasses(directory=step.run.directory).were_generated


def test_a_missing_class_is_written_into_the_run_that_proposed_it(step, known):
    """
    A class one scene needed is not a class the next one starts with, and nothing
    outside this run should ever import it.
    """
    generated = step.generate_missing({"KitchenIsland": ["Table", "HasDrawers"]}, known)
    assert generated == ["KitchenIsland(Table, HasDrawers)"]
    written = GeneratedClasses(directory=step.run.directory)
    assert written.were_generated
    assert "KitchenIsland" in written.path.read_text()


def test_the_bases_each_generated_class_declares_are_read_back(step, known):
    """
    Read from the file, so a finished run says what its classes derive from without
    being imported.
    """
    wanted = {"KitchenIsland": ["Table", "HasDrawers"], "Ceiling": ["Furniture"]}
    step.generate_missing(wanted, known)

    assert GeneratedClasses(directory=step.run.directory).declared_bases == wanted


def test_a_run_that_generated_nothing_declares_no_bases(step):
    assert GeneratedClasses(directory=step.run.directory).declared_bases == {}


def test_bases_are_ordered_as_a_class_must_declare_them(step, known):
    """
    A proposal naming HasRootBody beside IsStorageSpace -- which derives from it --
    names them the wrong way round for Python.
    """
    generated = step.generate_missing(
        {"Container": ["HasRootBody", "IsStorageSpace"]}, known
    )
    assert generated == ["Container(IsStorageSpace, HasRootBody)"]


def test_a_base_the_ontology_lacks_is_left_out_rather_than_guessed_at(step, known):
    """
    A class built from the bases that exist is still a class the bodies can be given.
    """
    generated = step.generate_missing({"Gizmo": ["Table", "NotAClass"]}, known)
    assert generated == ["Gizmo(Table)"]


def test_a_class_naming_no_base_the_ontology_has_derives_from_the_root(step, known):
    """
    Every annotation is a SemanticAnnotation, so there is always something to build
    from.
    """
    generated = step.generate_missing({"Gizmo": ["NotAClass"]}, known)
    assert generated == ["Gizmo(SemanticAnnotation)"]


# %% against what a real run generated


def test_a_real_run_asks_for_exactly_the_classes_it_generated(
    step, classifications, vocabulary, known, finished_run
):
    """
    Read against the file a finished run left behind: the classes wanted beyond the
    ontology are the ones it actually wrote, no more and no fewer.
    """
    wanted = step.wanted_classes(classifications, vocabulary)
    assert set(wanted) == {
        one.class_name for one in classifications.bodies if one.class_name
    }
    declared = {
        node.name
        for node in ast.parse(
            (finished_run.directory / "generated_classes.py").read_text()
        ).body
        if isinstance(node, ast.ClassDef)
    }
    assert set(wanted) - set(known) == declared


def test_a_real_run_builds_each_composed_class_the_way_the_vocabulary_composed_it(
    step, classifications, vocabulary
):
    """
    Whatever superclass the classification went on to name for the same object, the
    class keeps the mixins that let it hold the parts measured for it.
    """
    wanted = step.wanted_classes(classifications, vocabulary)
    for answer in vocabulary.proposals:
        assert wanted[answer.class_name] == [answer.superclass] + answer.mixins


# %% what a step is told the ontology holds


def test_a_class_composed_earlier_in_the_run_is_not_mistaken_for_the_ontology_s(
    step, taxonomy
):
    """
    Composing a proposed class registers it as a subclass of the annotation root for the
    rest of the process, and the steps of a run share one. Asked the live interpreter,
    the step that generates classes is told a class an earlier step invented for this
    scene is already part of the ontology, so it generates nothing -- and the
    interpreter that writes the world, which invented nothing, cannot find the class and
    leaves those bodies with no annotation at all.

    One real run named its island ``KitchenIsland`` and the class was never written.
    """
    step.run.write_json(RunFile.TAXONOMY, taxonomy)
    compose_class("InventedMidRun", annotation_classes(SemanticAnnotation)["Table"], [])

    assert "InventedMidRun" in annotation_classes(SemanticAnnotation)
    assert "InventedMidRun" not in step.ontology_classes()


def test_a_class_invented_for_this_scene_is_generated_even_after_it_was_composed(
    step, taxonomy
):
    """
    The class has to reach the run's own file, or the world cannot hold it.
    """
    step.run.write_json(RunFile.TAXONOMY, taxonomy)
    compose_class("InventedMidRun", annotation_classes(SemanticAnnotation)["Table"], [])

    generated = step.generate_missing(
        {"InventedMidRun": ["Table"]}, step.ontology_classes()
    )
    assert generated == ["InventedMidRun(Table)"]


def test_the_ontology_a_step_reads_is_the_one_the_run_wrote_down(step, taxonomy):
    """
    Every step and every interpreter of one run has to mean the same thing by it.
    """
    step.run.write_json(RunFile.TAXONOMY, taxonomy)
    declared = {node["name"] for node in taxonomy["classes"]}
    assert set(step.ontology_classes()) <= declared
    assert "Cabinet" in step.ontology_classes()


# %% the file a run starts from


def test_the_empty_stub_is_what_is_committed():
    """
    Every run rewrites the ontology's generated classes to an empty stub before it
    starts and puts them back when it ends.

    Written from a template, the stub has to come out byte for byte as the committed
    file, or every run leaves the source tree dirty.
    """
    committed = subprocess.run(
        [
            "git",
            "show",
            "HEAD:semantic_digital_twin/src/semantic_digital_twin/"
            "semantic_annotations/generated_classes.py",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    ).stdout
    assert PipelineTemplates().render("empty_generated_classes.py") == committed
