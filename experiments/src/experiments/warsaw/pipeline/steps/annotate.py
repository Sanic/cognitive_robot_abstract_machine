"""
Annotate a split scene's bodies and mount the parts into their wholes.

This is where everything the pipeline decided becomes a world: each body gets an
annotation of the class it was named as, and every pairing the adjudication settled is
carried out with the method the ontology says mounts it.

The mount is done with ``add()`` rather than by filling constructor fields, because a
whole holds *several* drawers and a constructor slot takes one value: ``add`` routes a
part to the field its type matches and appends where the field holds many.

The classes a scene needs that the ontology does not have are generated first, into the
run's own directory, and the ORM is rebuilt so the database knows them. That has to
happen before anything imports the classes, so the world is annotated in a second
interpreter.

Nothing here decides anything. Which body is which came from the split, what each one is
came from the classification, and which whole each part belongs to came from the
adjudication; if any of those is wrong, this writes it faithfully into the world.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from semantic_digital_twin.exceptions import UsageError
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    in_base_order,
    names_a_category,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasRootRegion,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from typing_extensions import Dict, List, Type

from experiments.warsaw.exceptions import NoWorldRecordedError
from experiments.warsaw.evaluation.graph import EvaluationGraph
from experiments.warsaw.pipeline.records import (
    Classifications,
    RefusedMount,
    SplitRecord,
    Vocabulary,
)
from experiments.warsaw.pipeline.report import RunReport
from experiments.warsaw.bases import HasLogger
from experiments.warsaw.pipeline.database.orm_rebuild import OrmRebuild
from experiments.warsaw.pipeline.database.run_schema import RunSchema
from experiments.warsaw.pipeline.database.world_store import WorldStore
from experiments.warsaw.pipeline.run import Run, RunFile
from experiments.warsaw.pipeline.run_classes import GeneratedClasses
from experiments.warsaw.pipeline.steps.compose.step import SuperclassChoice
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.scene_split import Pairing

# %% what an annotation needs beyond the body it is about

ROOT = "root"
"""
The field an annotation holds the body it is about in, and the one thing the step gives
every annotation it makes.
"""


def fields_beyond_the_body(annotation_class: Type) -> List[str]:
    """
    Name what an annotation class needs that the body it is about does not supply.

    Every annotation is made from that body and nothing else, so a class needing more
    cannot be made here. Most need nothing: a cabinet is made with no drawers and has
    them mounted into it afterwards, which is the only order the run could work in,
    since what belongs in which cabinet is not known when the cabinet is made.

    Two kinds cannot be. Some are constituted by other annotations -- a room is its
    floor, a double door is its two doors -- and are annotations over annotations rather
    than over a body. Others are rooted on a region rather than a body: an aperture is a
    hole, and a hole is not a thing. Nothing refuses the body in either case, because a
    field takes what it is given, so the second kind stays wrong and quiet until
    something reads the area a body does not have.

    :param annotation_class: The class an answer named.
    :return: The fields it requires besides the body, with ``root`` among them where a
        body is not the root it takes. Empty when a body is all it needs.
    """
    wanted = [
        one.name
        for one in dataclasses.fields(annotation_class)
        if one.init
        and one.name != ROOT
        and one.default is dataclasses.MISSING
        and one.default_factory is dataclasses.MISSING
    ]
    if not issubclass(annotation_class, HasRootBody):
        wanted.append(ROOT)
    return sorted(wanted)


# %% turning a body into the annotation it was answered as


@dataclass(frozen=True)
class AnnotationFromBody:
    """
    How a run makes an annotation of the class a body was answered as, out of that body.

    A run may be allowed to build a region from the body for a class that takes one.
    Both what it refuses and what it builds read that permission here, so the two cannot
    disagree about which classes a run can make.
    """

    may_make_a_region: bool = False
    """
    Whether a class rooted on a region is given one the size and pose of the body,
    rather than being refused for taking no body.
    """

    def cannot_supply(self, annotation_class: Type) -> List[str]:
        """
        Name what this run cannot give an annotation of that class.

        :param annotation_class: The class an answer named.
        :return: The fields it needs that the run has no way to fill. Empty when it can
            be made.
        """
        wanted = fields_beyond_the_body(annotation_class)
        if self.may_make_a_region and issubclass(annotation_class, HasRootRegion):
            return [one for one in wanted if one != ROOT]
        return wanted

    def make(
        self, annotation_class: Type, name: str, body: Body, world: World
    ) -> SemanticAnnotation:
        """
        Make one annotation of that class about that body.

        :param annotation_class: The class the answer named.
        :param name: What to call the annotation.
        :param body: The body it is about.
        :param world: The world the body is in.
        :return: The annotation, already in the world where making it put it there.
        """
        if self.may_make_a_region and issubclass(annotation_class, HasRootRegion):
            return annotation_class.create_with_new_region_in_world_from_body(
                name=name, world=world, body=body
            )
        return annotation_class(**{ROOT: body}, _world=world)


# %% what the mounting came to


@dataclass
class Mounted:
    """
    What carrying out a run's pairings produced.
    """

    carried_out: int = 0
    """
    How many mounts the world accepted.
    """

    refused: List[RefusedMount] = field(default_factory=list)
    """
    The ones it would not carry out, and why.
    """

    @property
    def attempted(self) -> int:
        """
        :return: How many mounts were tried.
        """
        return self.carried_out + len(self.refused)


# %% annotating and mounting, in a new interpreter


@dataclass
class MountAnnotations(HasLogger):
    """
    The annotating and mounting half, done in an interpreter that knows the run's
    classes.

    Constructed with a run directory and nothing else: which body is which, what each one
    is and which whole each belongs to are all in the run's own files, and the world to
    annotate is the one the split recorded.
    """

    directory: Path
    """
    The run's directory.
    """

    @property
    def run(self) -> Run:
        """
        :return: The run, as a directory of files.
        """
        return Run(directory=self.directory)

    @property
    def skip_classes_a_body_cannot_make(self) -> bool:
        """
        :return: Whether this run was told to leave a body alone when its class cannot
            be made from one. A run that says nothing was not.
        """
        written = self.run.read_json_if_written(RunFile.PROVENANCE)
        return bool(
            written.get("settings", {}).get("skip_classes_a_body_cannot_make", False)
        )

    @property
    def skip_classes_that_name_a_category(self) -> bool:
        """
        :return: Whether this run was told to refuse a class the ontology declares a
            category. A run that says nothing was not.
        """
        written = self.run.read_json_if_written(RunFile.PROVENANCE)
        return bool(
            written.get("settings", {}).get("skip_classes_that_name_a_category", False)
        )

    @property
    def make_a_region_where_a_class_needs_one(self) -> bool:
        """
        :return: Whether this run was told to build a region for a class that takes one.
            A run that says nothing was not.
        """
        written = self.run.read_json_if_written(RunFile.PROVENANCE)
        return bool(
            written.get("settings", {}).get(
                "make_a_region_where_a_class_needs_one", False
            )
        )

    @property
    def annotation_from_body(self) -> AnnotationFromBody:
        """
        :return: How this run turns a body into the annotation it was answered as.
        """
        return AnnotationFromBody(
            may_make_a_region=self.make_a_region_where_a_class_needs_one
        )

    def carry_out(self) -> None:
        """
        Read the split world back, annotate it, mount its parts, and write it afresh.
        """
        # Before the ORM is reached, so that the classes it names are this run's.
        GeneratedClasses(directory=self.directory).use()
        RunSchema.for_run(self.directory).use()

        split = self.run.read_record(RunFile.SPLIT, SplitRecord)
        classifications = self.run.read_record(RunFile.CLASSIFICATIONS, Classifications)
        if split.world_id is None:
            raise NoWorldRecordedError(step="split")

        store = WorldStore()
        # The ORM was regenerated with the classes this scene needed, but a generated
        # class has no table until one is made for it, and a world holding one cannot be
        # written.
        store.create_tables()
        world = store.read(split.world_id)
        self.logger.info(
            "world %s read back, %s bodies", split.world_id, len(world.bodies)
        )

        annotations = self.annotate(world, classifications)
        mounted = self.mount(world, annotations, split.pairings)

        self.logger.info(
            "%s of %s pairings mounted", mounted.carried_out, mounted.attempted
        )
        for refusal in mounted.refused:
            self.logger.warning(
                "  %s <- %s: %s",
                refusal.pairing.whole,
                refusal.pairing.part,
                refusal.reason,
            )

        # Written as a world of its own rather than merged onto the one it was read from.
        # Merging serialises a world afresh and grafts it onto the stored graph, and what
        # came back could not be read at all: replaying its modifications reached an entity
        # that was not there. Two rows also keep the split world as it was, which is what
        # the classification and the pairings were decided against.
        split.annotated_world_id = store.write(world)
        split.refused = mounted.refused
        self.logger.info(
            "written as world %s, annotated, beside the split world %s",
            split.annotated_world_id,
            split.world_id,
        )
        self.run.write_record(RunFile.SPLIT, split)
        graph = EvaluationGraph.from_run_products(
            split=split,
            classifications=classifications,
            annotated_names=set(annotations),
        ).with_fields_resolved(annotation_classes(SemanticAnnotation))
        self.run.write_record(RunFile.EVALUATION_GRAPH, graph)

        report = RunReport(run=self.run)
        report.write_inspector()
        report.write_publisher()
        report.write()
        self.logger.info(
            "written to %s, %s and %s",
            self.run.path(RunFile.REPORT),
            self.run.path(RunFile.INSPECTOR),
            self.run.path(RunFile.PUBLISHER),
        )

        # Last, because it paints the world on the way out and because a run whose every
        # other product is already written loses nothing if this fails.
        self.logger.info("written as a scene to %s", report.write_world_mesh(world))

    def annotate(
        self, world: World, classifications: Classifications
    ) -> Dict[str, SemanticAnnotation]:
        """
        Give every body an annotation of the class it was named as.

        :param world: The world the bodies are in.
        :param classifications: What each body was answered to be.
        :return: The annotation made for each body, by name.
        """
        # Every subclass this interpreter holds, and deliberately: it composed nothing, so
        # nothing here is another step's invention, and the classes this run generated are
        # exactly what it has to be able to find.
        known = annotation_classes(SemanticAnnotation)
        bodies = {str(body.name.name): body for body in world.bodies}

        annotations: Dict[str, SemanticAnnotation] = {}
        left_alone: Counter = Counter()
        with world.modify_world():
            for answer in classifications.bodies:
                name = answer.name
                body = bodies.get(name)
                if body is None:
                    left_alone["no such body"] += 1
                    continue
                if answer.class_name not in known:
                    left_alone[answer.class_name] += 1
                    continue
                # A category names no object, and the taxonomy told the model so. One
                # answer naming one should cost that body rather than every body after it,
                # and it is refused here rather than instantiated, so that what the model
                # was told and what the run asserts are the same thing.
                if names_a_category(
                    known[answer.class_name],
                    categories_are_answers=not self.skip_classes_that_name_a_category,
                ):
                    left_alone[f"{answer.class_name} (abstract)"] += 1
                    continue
                making = self.annotation_from_body
                wanted = (
                    making.cannot_supply(known[answer.class_name])
                    if self.skip_classes_a_body_cannot_make
                    else []
                )
                if wanted:
                    left_alone[f"{answer.class_name} (needs {', '.join(wanted)})"] += 1
                    continue
                annotation = making.make(known[answer.class_name], name, body, world)
                # Registered as it is made, before anything is mounted into it. A mount
                # records an attribute update against the annotation it changes, and a
                # world replaying its modifications has to have that annotation already:
                # registering them all afterwards puts every update before its own subject
                # and the world cannot be read back at all.
                if annotation not in world.semantic_annotations:
                    world.add_semantic_annotation(annotation)
                annotations[name] = annotation

        self.logger.info("%s bodies annotated", len(annotations))
        for missing, count in left_alone.most_common():
            self.logger.info("  %s left alone: %s", count, missing)
        return annotations

    def mount(
        self,
        world: World,
        annotations: Dict[str, SemanticAnnotation],
        pairings: List[Pairing],
    ) -> Mounted:
        """
        Mount every pairing's part into its whole, in the field the pairing names.

        :param world: The world the annotations are in.
        :param annotations: The annotation made for each body, by name.
        :param pairings: The mounts the adjudication settled.
        :return: What was mounted, and what was refused.
        """
        mounted = Mounted()
        # A mount moves the part's branch under the whole, so it modifies the world model
        # and has to be told so.
        with world.modify_world():
            for pairing in pairings:
                whole = annotations.get(pairing.whole)
                part = annotations.get(pairing.part)
                if whole is None or part is None:
                    mounted.refused.append(
                        RefusedMount(
                            pairing=pairing, reason="one end has no annotation"
                        )
                    )
                    continue
                try:
                    whole.add(part, field_name=pairing.field_name)
                except UsageError as refusal:
                    # The world refusing a mount is an answer about this pairing, not a
                    # state the run should not have reached: the class a body was given
                    # may simply not admit the part another step said it holds.
                    mounted.refused.append(
                        RefusedMount(
                            pairing=pairing,
                            reason=f"{type(refusal).__name__}: "
                            f"{str(refusal).splitlines()[0]}",
                        )
                    )
                else:
                    mounted.carried_out += 1
        return mounted


# %% generating the classes the scene needs


@dataclass
class AnnotateAndMount(PipelineStep):
    """
    The classes a scene needed, the ORM that knows them, and the world they annotate.
    """

    @property
    def name(self) -> str:
        return "annotate the bodies and mount the parts"

    @property
    def class_template(self) -> str:
        """
        :return: What a generated class is written from.
        """
        return "dataclass_template.py.jinja"

    def carry_out(self) -> None:
        """
        Generate what the ontology lacks, rebuild the ORM, then annotate in a new one.
        """
        classifications = self.run.read_record(RunFile.CLASSIFICATIONS, Classifications)
        vocabulary = self.run.read_record(RunFile.VOCABULARY, Vocabulary)
        wanted = self.wanted_classes(classifications, vocabulary)
        known = self.ontology_classes()
        self.logger.info(
            "%s classes over %s bodies", len(wanted), len(classifications.bodies)
        )
        if self.settings.settle_the_superclass:
            wanted = self.settled_superclasses(wanted, known, classifications)

        generated = self.generate_missing(wanted, known)
        if generated:
            self.logger.info("generated %s: %s", len(generated), ", ".join(generated))
            self.logger.info("regenerating the ORM ...")
            OrmRebuild(directory=self.run.directory).run_in_new_interpreter()
        else:
            self.logger.info("every class the scene needs is already in the ontology")

        # The classes were written after this process read the ontology, so the world is
        # annotated in an interpreter that starts after they exist.
        self.logger.info("annotating in a new interpreter ...")
        printed = self.in_new_interpreter(
            MountAnnotations, what="annotating the world and mounting its parts"
        )
        for line in printed.splitlines():
            self.logger.info(line)

    def settled_superclasses(
        self,
        wanted: Dict[str, List[str]],
        known: Dict[str, Type],
        classifications: Classifications,
    ) -> Dict[str, List[str]]:
        """
        Ask what each class this run wants is a kind of, and build it from that answer.

        Asked once per class rather than once per body, and only for the classes the
        ontology does not already have: placing a class it has is settled already.

        :param wanted: Per class name, what the step that named it proposed.
        :param known: The ontology's classes by name.
        :param classifications: What each body was answered to be, for the labels a class
            was proposed from.
        :return: The same classes, built from what was settled where an answer was usable
            and from what was proposed where none was.
        """
        taxonomy = self.run.read_json(RunFile.TAXONOMY)
        labels = defaultdict(set)
        for answer in classifications.bodies:
            if answer.class_name and answer.label:
                labels[answer.class_name].add(answer.label)

        questioner = self.questioner(RunFile.SUPERCLASS_ANSWERS)
        settled = dict(wanted)
        for name, proposed in sorted(wanted.items()):
            if name in known:
                continue
            question = SuperclassChoice(
                taxonomy=taxonomy,
                known=known,
                class_name=name,
                proposed_bases=proposed,
                labels=sorted(labels.get(name, ())),
            )
            bases = question.bases(questioner.answer(question).answer)
            if bases is None:
                self.logger.info("  %s: kept %s", name, ", ".join(proposed))
                continue
            if bases != proposed:
                self.logger.info(
                    "  %s: %s -> %s", name, ", ".join(proposed), ", ".join(bases)
                )
            settled[name] = bases
        return settled

    @staticmethod
    def wanted_classes(
        classifications: Classifications, vocabulary: Vocabulary
    ) -> Dict[str, List[str]]:
        """
        Say what each class a body was given should be built from.

        Two steps proposed compositions and only one of them was asked to. The
        vocabulary step answers what a *label* means and names a superclass and the
        mixins to compose it from, having been shown what objects of that label were
        measured to meet; the classification step answers which class each *object* is,
        from a picture, and its schema carries a superclass as well. Read from the
        classification alone, a Faucet that the vocabulary had composed with HasHandle
        comes out with no way to hold a handle at all, and the pairing measured for it
        cannot be mounted.

        So the bases come from the vocabulary where it composed that class, and from the
        classification only where it did not.

        :param classifications: What the classification step wrote.
        :param vocabulary: What the vocabulary step wrote.
        :return: Per class name, the names of the classes to derive it from.
        """
        composed = {
            answer.class_name: [answer.superclass] + list(answer.mixins)
            for answer in vocabulary.labels
            if answer.class_name and answer.is_new_class and answer.superclass
        }

        wanted: Dict[str, List[str]] = {}
        for answer in classifications.bodies:
            if not answer.class_name or answer.class_name in wanted:
                continue
            wanted[answer.class_name] = composed.get(
                answer.class_name, [answer.superclass or SemanticAnnotation.__name__]
            )
        return wanted

    def generate_missing(
        self, wanted: Dict[str, List[str]], known: Dict[str, Type]
    ) -> List[str]:
        """
        Write the classes a scene needs that the ontology does not have.

        The file is written whole rather than appended to: a class one scene needed is not
        a class the next one starts with, and nothing outside this run should ever import
        it.

        :param wanted: Per class name, the names of the classes to derive it from.
        :param known: The ontology's classes by name.
        :return: The names that were generated, with what each was built from.
        """
        builders, generated = [], []
        for name, base_names in sorted(wanted.items()):
            if name in known:
                continue
            bases = [known[one] for one in base_names if one in known]
            for missing in [one for one in base_names if one not in known]:
                self.logger.warning(
                    "  %s: %s is not in the ontology, leaving it out", name, missing
                )
            if not bases:
                bases = [SemanticAnnotation]

            # Ordered as a class must declare them, since a proposal naming HasRootBody
            # beside IsStorageSpace -- which derives from it -- names them the wrong way
            # round for Python.
            ordered = in_base_order(bases)
            builder = SemanticAnnotationClassBuilder(
                name, template_name=self.class_template
            )
            for base in ordered:
                builder.add_base(base)
            builders.append(builder)
            generated.append(f"{name}({', '.join(base.__name__ for base in ordered)})")
            # A stubbed class can be made and annotated, which is the point, and it then
            # looks complete to everything that only asks whether it can be. Said aloud
            # so that what a run could not know is not left for a robot to discover.
            for stubbed in builder.stubbed_methods:
                self.logger.warning(
                    "  %s: %s was stubbed, since no run can know what it answers",
                    name,
                    stubbed,
                )

        if builders:
            generated_classes = GeneratedClasses(directory=self.run.directory)
            generated_classes.searched_directory.mkdir(parents=True, exist_ok=True)
            SemanticAnnotationClassBuilder.write_classes_to_file(
                builders, generated_classes.path
            )
        return generated


if __name__ == "__main__":
    # What the annotating says goes to standard output, which the step that started this
    # interpreter reads and says again.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    MountAnnotations(directory=Path(sys.argv[1])).carry_out()
