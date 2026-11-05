"""
Measure how a scene's labelled objects meet, and say what the ontology makes of it.

Splitting a scene into bodies means deciding what to do where its labels overlap: the same
faces can be a cabinet and its door, a drawer and the handle on it. This gathers what those
decisions need and decides nothing itself:

- the geometry, measured (shared faces, edges touched along, distance apart),
- the ontology, in the form that shows what each class can be composed of,
- per pair, which part-whole relations the ontology admits between their classes.

Labels are not classes -- a scene labelling something ``kitchen_island`` says nothing about
which class that is, and some labels have no class at all. That mapping is a question for a
model, and the vocabulary request is what it is asked with. Run again once the answer is
back, and the same measurement says what the ontology admits for every pair.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path

import numpy as np
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    admissible_mounts,
    describe_class,
)
from typing_extensions import Dict, List, Optional, Sequence, Tuple, Type

from semantic_digital_twin.world_description.geometry import Color

from experiments.warsaw.painting import PaintColor
from experiments.warsaw.pipeline.camera_poses import CameraPoses
from experiments.warsaw.pipeline.templates import PipelineTemplates
from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import (
    AdmissibleMount,
    CountedClaimants,
    ContestedShare,
    ForcedMembership,
    MeasuredSegment,
    MembershipCandidate,
    MembershipQuestion,
    OntologySlice,
    OntologyView,
    OpenQuestions,
    OwnershipQuestion,
    PairRecord,
    Relations,
    RelationStatus,
    Vocabulary,
    VocabularyRequest,
    LabelRequest,
)
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.segment_relations import (
    SegmentRelations,
    claimant_groups,
    segment_evidence,
)
from experiments.warsaw.world_loader.loader import (
    RenderedPictures,
    WarsawWorldLoader,
)
from experiments.warsaw.world_loader.scene import LabelSegment
from experiments.warsaw.world_loader.viewpoints import ViewpointChoice

# %% what the measurements leave open


@dataclass
class WhatIsOpen:
    """
    The questions a scene's overlaps raise, and the faces each is rendered around.
    """

    questions: OpenQuestions
    """
    What is left to decide.
    """

    settled: List[CountedClaimants] = field(default_factory=list)
    """
    The sets of contested faces the ontology decides on its own.
    """

    forced: List[ForcedMembership] = field(default_factory=list)
    """
    The memberships with only one candidate.
    """

    contested: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    Per ownership question, the faces every one of its objects claims.

    Kept apart from the question itself because it is what the render is painted with
    rather than anything the run writes down: a picture that does not mark the contested
    faces shows them as belonging to whichever object was painted over them, which is the
    very thing in question.
    """


# %% the parts and the wholes they could belong to


@dataclass
class Memberships:
    """
    The parts of a scene that have a whole to choose, and those that have only one.
    """

    to_ask_about: List[MembershipQuestion] = field(default_factory=list)
    """
    The parts that meet more than one candidate.
    """

    forced: List[ForcedMembership] = field(default_factory=list)
    """
    The parts that meet exactly one, where there is nothing to choose between.
    """


# %% measuring the scene


@dataclass
class MeasureScene(PipelineStep):
    """
    The scene as it was measured, knowing no classes yet, and one render per label for
    the question asking what each label means.
    """

    question_template: str = "vocabulary_request.md"
    """
    The file the vocabulary request's question is written in, beside this step.
    """

    @property
    def question_prompt(self) -> str:
        """
        :return: What the vocabulary request asks.
        """
        prompts = PipelineTemplates(
            directory=Path(__file__).resolve().parent / "prompts"
        )
        return prompts.render_document(self.question_template)

    # %% the step

    @property
    def name(self) -> str:
        return "measure the scene"

    @property
    def rewrites_first_pass(self) -> bool:
        """
        :return: Whether this writes over what the first measurement wrote.
        """
        return False

    def vocabulary(self) -> Vocabulary:
        """
        :return: What each label was answered to mean, which nothing has been yet.
        """
        return Vocabulary(model="", scene="")

    @property
    def renders_exemplars(self) -> bool:
        """
        :return: Whether this renders one exemplar per label for the vocabulary question,
            which only a run shown the pictures needs.
        """
        return self.settings.show_the_pictures

    @property
    def renders_questions(self) -> bool:
        """
        :return: Whether this renders the open questions, which are not known yet.
        """
        return False

    def draw(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        request: VocabularyRequest,
        open_now: WhatIsOpen,
    ) -> None:
        """
        Render one exemplar per label, which the vocabulary question is put with.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param request: The vocabulary question, which the renders are filled into.
        :param open_now: What the measurement leaves open.
        """
        if not self.renders_exemplars:
            return
        self.render_exemplars(loader, measured, segments, request)

    @property
    def viewpoints(self) -> Optional[List[str]]:
        """
        :return: Which viewpoints to render, or None to render all of them so that one can
            be chosen between.
        """
        if self.settings.viewpoint_choice is ViewpointChoice.ALL:
            return list(self.settings.kept_viewpoints)
        return None

    @property
    def chooses_viewpoint(self) -> Optional[ViewpointChoice]:
        """
        :return: How to choose the one viewpoint to keep, or None to keep them all.
        """
        choice = self.settings.viewpoint_choice
        return None if choice is ViewpointChoice.ALL else choice

    def loader(self) -> WarsawWorldLoader:
        """
        :return: The scene, loaded and ready to be measured and rendered.
        """
        return WarsawWorldLoader(
            input_directory=self.settings.scene_directory,
            render_sizes=self.settings.render_sizes,
        )

    def vocabulary_classes(self) -> VocabularyClasses:
        """
        :return: What each label stands for, empty on the pass that has not asked yet.
        """
        return VocabularyClasses(
            vocabulary=self.vocabulary(), known=self.ontology_classes()
        )

    def carry_out(self) -> None:
        """
        Measure the scene, write what was measured, and say what is left open.
        """
        self.run.refuse_to_write_over(
            RunFile.RELATIONS,
            RunFile.VOCABULARY_REQUEST,
            RunFile.QUESTIONS,
            overwrite=self.rewrites_first_pass,
        )

        loader = self.loader()
        segments = {str(segment.name): segment for segment in loader.label_segments}
        measured = self.measured_scene(loader)
        self.logger.info(
            "%s pairs stand in some measurable relation", len(measured.pairs)
        )

        classes = self.vocabulary_classes().by_label()
        relations = self.relations_of(loader, measured, classes)

        request = self.vocabulary_request(loader, measured)
        open_now = self.open_questions(loader, measured, relations, classes)
        relations = replace(relations, settled=open_now.settled, forced=open_now.forced)
        self.logger.info(
            "%s class patterns and %s memberships are open; %s groups the ontology "
            "settles",
            len(open_now.questions.ownership),
            len(open_now.questions.membership),
            len(open_now.settled),
        )

        self.draw(loader, measured, segments, request, open_now)
        self.run.write_record(RunFile.VOCABULARY_REQUEST, request)
        self.run.write_record(RunFile.RELATIONS, relations)
        self.run.write_record(RunFile.QUESTIONS, open_now.questions)

        self.report(relations)

    # %% what the ontology makes of a pair

    # %% what the ontology admits between two segments

    def ontology_view(
        self, one_class: Optional[Type], other_class: Optional[Type]
    ) -> OntologyView:
        """
        Ask the ontology which of two classes may hold the other as a part.

        :param one_class: The class of one segment, or None when its label maps to none.
        :param other_class: The class of the other segment.
        :return: The admissible relations and what that leaves open.
        """
        if one_class is None or other_class is None:
            return OntologyView(status=RelationStatus.CLASS_UNKNOWN)

        view = OntologyView(
            status=RelationStatus.NO_LEGAL_RELATION,
            mounts=[
                AdmissibleMount(
                    whole=whole.__name__,
                    field_name=relation.field_name,
                    target=relation.target,
                    holds_many=relation.holds_many,
                    removes_geometry=relation.removes_geometry,
                )
                for whole, relation in admissible_mounts(one_class, other_class)
            ],
        )
        if len(view.mounts) == 1:
            return replace(view, status=RelationStatus.RELATION_KNOWN)
        if view.mounts:
            return replace(view, status=RelationStatus.RELATION_AMBIGUOUS)
        return view

    def measured_scene(self, loader: WarsawWorldLoader) -> SegmentRelations:
        """
        Measure how the scene's labelled objects meet, or take the measurement an
        earlier step of this run already made.

        The measurement is geometry and nothing else, so a scene measured twice in one
        run is measured to meet itself the same way twice. On a scanned room it is most
        of what the run costs, and the two steps that need it differ in what they know
        about the labels rather than in the geometry.

        :param loader: The loaded scene.
        :return: Every segment, and every pair standing in some measurable relation.
        """
        already = self.run.read_record_if_written(
            RunFile.RELATIONS, Relations(scene="")
        )
        if already.segments:
            self.logger.info(
                "%s segments; taking the measurement this run already made",
                len(already.segments),
            )
            return already.as_measured()
        self.logger.info(
            "%s segments; measuring how they meet ...", len(loader.label_segments)
        )
        return segment_evidence(loader, nearest=self.settings.nearest)

    def relations_of(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        classes: Dict[str, Optional[Type]],
    ) -> Relations:
        """
        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param classes: Per label, the class it was read as.
        :return: Every measured pair, with what the ontology makes of it.
        """
        pairs = []
        for pair in measured.pairs:
            one = measured.descriptors[pair.one]
            other = measured.descriptors[pair.other]
            pairs.append(
                PairRecord(
                    evidence=pair,
                    classes={
                        one.class_name: self.class_name_of(classes.get(one.class_name)),
                        other.class_name: self.class_name_of(
                            classes.get(other.class_name)
                        ),
                    },
                    view=self.ontology_view(
                        classes.get(one.class_name), classes.get(other.class_name)
                    ),
                    prompt_block=pair.as_prompt_block(measured.descriptors),
                )
            )
        return Relations(
            scene=str(loader.scene.mesh_path),
            segments=list(measured.descriptors.values()),
            pairs=pairs,
        )

    @staticmethod
    def class_name_of(annotation_class: Optional[Type]) -> Optional[str]:
        """
        :param annotation_class: A class, or None where a label maps to none.
        :return: Its name, or None.
        """
        return None if annotation_class is None else annotation_class.__name__

    # %% the question asking what the labels mean

    # %% choosing what to picture

    @staticmethod
    def exemplars(measured: SegmentRelations) -> Dict[str, str]:
        """
        Pick the instance of each label that shows the label best.

        The one with the most surface no other segment claims is the one a viewer can
        judge without judging an overlap at the same time. Ranking by the *share*
        instead picks slivers, which are wholly unclaimed precisely because they are
        fragments: on one scene it offered a strip of 545 faces as the example of a
        cabinet while a whole cabinet front of 960 stood beside it, equally unclaimed.

        :param measured: The measured scene.
        :return: Per label, the name of the segment standing for it.
        """
        best: Dict[str, str] = {}
        for descriptor in measured.descriptors.values():
            standing = best.get(descriptor.class_name)
            if (
                standing is None
                or descriptor.exclusive_area
                > measured.descriptors[standing].exclusive_area
            ):
                best[descriptor.class_name] = descriptor.name
        return best

    def vocabulary_request(
        self, loader: WarsawWorldLoader, measured: SegmentRelations
    ) -> VocabularyRequest:
        """
        :param loader: The loaded scene.
        :param measured: The measured scene.
        :return: The question asking which class each label means.
        """
        instances = Counter(
            descriptor.class_name for descriptor in measured.descriptors.values()
        )
        return VocabularyRequest(
            scene=str(loader.scene.mesh_path),
            question=self.question_prompt,
            labels=[
                LabelRequest(
                    label=label,
                    instances=instances[label],
                    exemplar=name,
                    exemplar_faces=measured.descriptors[name].faces,
                    exemplar_exclusive_share=round(
                        measured.descriptors[name].exclusive_share, 4
                    ),
                    exemplar_exclusive_area=round(
                        measured.descriptors[name].exclusive_area, 4
                    ),
                )
                for label, name in sorted(self.exemplars(measured).items())
            ],
        )

    # %% rendering

    # %% drawing it

    @staticmethod
    def neighbourhood(
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        names: Sequence[str],
    ) -> List[LabelSegment]:
        """
        Gather what a context view has to show for its subject to be recognisable.

        A picture of the whole room is useless for anything small -- a mug covers none
        of it. Framed on the mug together with the segments measured to stand nearest
        it, the same mug covers a tenth of the picture, and what surrounds it is what
        says which mug it is. The neighbours come from the same measurement the rest of
        the evidence does, so the neighbourhood is not a radius anyone chose.

        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param names: The segments the view is about.
        :return: Those segments together with the ones measured to stand near them.
        """
        wanted = set(names)
        for name in names:
            for pair in measured.pairs_of(name):
                wanted.update((pair.one, pair.other))
        return [segments[name] for name in sorted(wanted)]

    @staticmethod
    def group_highlights(
        segments: Sequence[LabelSegment], contested: np.ndarray
    ) -> Tuple[List[Tuple[Color, np.ndarray]], Dict[str, PaintColor]]:
        """
        Color a set of objects so that what they disagree about can be seen.

        Each gets a color of its own, and the faces all of them claim get one more,
        painted last. A picture without that last color shows the contested faces as
        belonging to whichever object was painted over them, which is the very thing in
        question.

        :param segments: The objects to color.
        :param contested: The faces they all claim.
        :return: What to paint, and what each color stands for.
        :raises TooManyToPaintError: If there are more objects than colors to tell
            them apart by.
        """
        colors = PaintColor.distinct(len(segments) + 1)
        highlights = [
            (color.color, segment.face_indices)
            for color, segment in zip(colors, segments)
        ]
        highlights.append((colors[-1].color, contested))
        legend = {str(segment.name): color for color, segment in zip(colors, segments)}
        legend["contested"] = colors[-1]
        return highlights, legend

    @staticmethod
    def write_images(
        pictures: RenderedPictures, directory: Path, prefix: str
    ) -> List[str]:
        """
        Write some renders, and record where the camera stood for each.

        The pose is kept because a picture nobody can say where it was taken from cannot
        be measured against anything afterwards, and it is worked out while rendering and
        otherwise lost.

        :param pictures: The renders to write, by viewpoint, with their camera poses.
        :param directory: Where to write them.
        :param prefix: What to name them after.
        :return: The names they were written as.
        """
        directory.mkdir(parents=True, exist_ok=True)
        named = {name: f"{prefix}__{name}.png" for name in pictures.images}
        for name, image in pictures.images.items():
            (directory / named[name]).write_bytes(image)
        by_filename = pictures.named(named)
        CameraPoses.record(
            directory=directory,
            frame=by_filename.frame,
            field_of_view=by_filename.field_of_view,
            poses=by_filename.camera_poses,
        )
        return sorted(named.values())

    def render_exemplars(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        request: VocabularyRequest,
    ) -> None:
        """
        Render one object per label, and record what each was painted.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param request: The question to fill the renders into.
        """
        self.logger.info("rendering %s exemplars ...", len(request.labels))
        for entry in request.labels:
            segment = segments[entry.exemplar]
            [color] = PaintColor.distinct(1)
            entry.images = self.write_images(
                loader.render_region(
                    [segment],
                    [(color.color, segment.face_indices)],
                    viewpoints=self.viewpoints,
                    headless=self.settings.headless,
                    choose_viewpoint=self.chooses_viewpoint,
                    context_segments=self.neighbourhood(
                        measured, segments, [entry.exemplar]
                    ),
                ),
                self.run.path(RunFile.EXEMPLARS),
                entry.label,
            )
            entry.color = color.word
            self.logger.info("  %s: %s", entry.label, entry.exemplar)

    def render_questions(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        questions: OpenQuestions,
        contested: Dict[str, np.ndarray],
    ) -> None:
        """
        Render each open question, and record what each color in it stands for.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param questions: The questions to fill the renders into.
        :param contested: Per question, the faces all its objects claim.
        """
        asked = questions.ownership + questions.membership
        self.logger.info("rendering %s questions ...", len(asked))
        for question in asked:
            shown = [segments[name] for name in question.shown]
            highlights, legend = self.group_highlights(
                shown, contested.get(question.name, np.array([], dtype=np.int64))
            )
            question.images = self.write_images(
                loader.render_region(
                    shown,
                    highlights,
                    viewpoints=self.viewpoints,
                    headless=self.settings.headless,
                    choose_viewpoint=self.chooses_viewpoint,
                    context_segments=self.neighbourhood(
                        measured, segments, question.shown
                    ),
                ),
                self.run.path(RunFile.QUESTION_RENDERS),
                question.name,
            )
            question.legend = {name: color.word for name, color in legend.items()}
            self.logger.info("  %s", question.name)

    # %% what is left to decide

    # %% what a model is told about the classes

    @staticmethod
    def ontology_slice(
        names: Sequence[str],
        labels: Dict[str, str],
        classes: Dict[str, Optional[Type]],
    ) -> OntologySlice:
        """
        Say what the ontology holds about a handful of objects.

        The questions about a set of objects were asked with the pictures and the
        measurements alone, and the ontology already knows things that bear on them:
        that a cabinet can hold a drawer says the drawer is the finer of the two, which
        is exactly what "whose surface is this" turns on.

        :param names: The segments in question.
        :param labels: Per segment, the label it carries.
        :param classes: Per label, the class it was read as.
        :return: Per segment its class, each class written out, and what the ontology
            admits between each pair of them.
        """
        read_as = {name: classes.get(labels[name]) for name in names}
        admits = []
        for one, other in combinations(names, 2):
            if read_as[one] is None or read_as[other] is None:
                continue
            for whole, relation in admissible_mounts(read_as[one], read_as[other]):
                admits.append(
                    f"{whole.__name__}.{relation.field_name} may hold a "
                    f"{relation.target} as a part"
                )
        return OntologySlice(
            read_as={
                name: None if held is None else held.__name__
                for name, held in read_as.items()
            },
            classes=[
                describe_class(held)
                for held in dict.fromkeys(
                    one for one in read_as.values() if one is not None
                )
            ],
            # The same mount turns up once per pair that could use it, and almost every
            # class has objects -> HasRootBody, so without this the list reads as though
            # containment were being urged six times over.
            admits=list(dict.fromkeys(admits)),
        )

    @staticmethod
    def measured_of(
        names: Sequence[str], measured: SegmentRelations
    ) -> List[MeasuredSegment]:
        """
        :param names: The segments in question.
        :param measured: The measured scene.
        :return: What was measured of each of them on its own.
        """
        return [
            MeasuredSegment(
                name=name,
                faces=int(measured.descriptors[name].faces),
                area=round(measured.descriptors[name].area, 3),
                height=round(measured.descriptors[name].height, 2),
                pieces=int(measured.descriptors[name].components),
            )
            for name in names
        ]

    # %% what is left to ask

    def open_questions(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        relations: Relations,
        classes: Dict[str, Optional[Type]],
    ) -> WhatIsOpen:
        """
        Work out what is actually left to decide, and how few questions it takes.

        Three things shrink the pile. A face belongs to one object, so the question is
        asked once per set of claimants rather than once per pair of them. A group whose
        every internal pair the ontology settled as a part-whole relation is not a question
        at all, since the part keeps the surface it is made of. And what is left repeats: a
        door and a window sharing a pane is one question however many glazed doors the room
        has, so the groups are gathered by the classes in them and asked once per pattern.

        Which whole a part belongs to is a separate question, and only where a part
        overlaps more than one candidate: a drawer that meets exactly one cabinet has
        nothing to choose between.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param relations: The pairs, as they were written out.
        :param classes: Per label, the class it was read as.
        :return: What is left to decide, and what each question is rendered around.
        """
        segments = loader.label_segments
        groups = claimant_groups(
            [segment.face_indices for segment in segments],
            [str(segment.name) for segment in segments],
            len(loader.scene_mesh.faces),
        )
        status = {
            tuple(sorted((pair.evidence.one, pair.evidence.other))): pair.status
            for pair in relations.pairs
        }
        labels = relations.labels

        settled, patterns = [], defaultdict(list)
        for group in groups:
            inside = [
                status.get(tuple(sorted(pair))) for pair in combinations(group.names, 2)
            ]
            if all(answer is RelationStatus.RELATION_KNOWN for answer in inside):
                settled.append(CountedClaimants.of(group))
            else:
                patterns[tuple(sorted(labels[name] for name in group.names))].append(
                    group
                )

        ownership, contested = [], {}
        for pattern, members in sorted(patterns.items(), key=lambda one: -len(one[1])):
            exemplar = max(members, key=lambda group: len(group.faces))
            question = OwnershipQuestion(
                name="__".join(pattern),
                pattern=list(pattern),
                shown=list(exemplar.names),
                covers=[CountedClaimants.of(group) for group in members],
                contested_faces=sum(len(group.faces) for group in members),
                exemplar_faces=int(len(exemplar.faces)),
                shares=[
                    ContestedShare(
                        name=name,
                        faces=int(measured.descriptors[name].faces),
                        contested_share=round(
                            len(exemplar.faces) / measured.descriptors[name].faces, 4
                        ),
                    )
                    for name in exemplar.names
                ],
                ontology=self.ontology_slice(exemplar.names, labels, classes),
                measured=self.measured_of(exemplar.names, measured),
            )
            ownership.append(question)
            contested[question.name] = exemplar.faces

        memberships = self.memberships(relations, labels, classes, measured)
        return WhatIsOpen(
            questions=OpenQuestions(
                scene=relations.scene,
                asked=ownership + memberships.to_ask_about,
            ),
            settled=settled,
            forced=memberships.forced,
            contested=contested,
        )

    def memberships(
        self,
        relations: Relations,
        labels: Dict[str, str],
        classes: Dict[str, Optional[Type]],
        measured: SegmentRelations,
    ) -> Memberships:
        """
        Work out which parts have a whole to choose and which have only one.

        A part that shares faces with a whole it could belong to, or touches one along an
        edge, is offered only those. A part that meets none is offered the wholes within
        :attr:`~experiments.warsaw.pipeline.settings.PipelineSettings.largest_gap_to_a_whole`
        of it, since a scan leaves gaps between what is mounted together.

        :param relations: The pairs, as they were written out.
        :param labels: Per segment, the label it carries.
        :param classes: Per label, the class it was read as.
        :param measured: The measured scene.
        :return: The parts with a whole to choose, and the parts with only one.
        """
        meeting: Dict[str, Dict[str, MembershipCandidate]] = defaultdict(dict)
        nearby: Dict[str, Dict[str, MembershipCandidate]] = defaultdict(dict)
        for pair in relations.pairs:
            if pair.status is not RelationStatus.RELATION_KNOWN:
                continue
            meets = pair.evidence.shared_faces or pair.evidence.touching_edges
            if (
                not meets
                and pair.evidence.distance > self.settings.largest_gap_to_a_whole
            ):
                continue
            admitted = pair.view.mounts[0]
            whole, part = pair.evidence.one, pair.evidence.other
            if pair.classes.get(labels[whole]) != admitted.whole:
                whole, part = part, whole
            (meeting if meets else nearby)[part][whole] = MembershipCandidate(
                name=whole,
                field_name=admitted.field_name,
                shared_faces=pair.evidence.shared_faces,
                touching_edges=pair.evidence.touching_edges,
                distance=round(pair.evidence.distance, 4),
            )
        candidates = {
            part: meeting.get(part) or nearby[part]
            for part in sorted(set(meeting) | set(nearby))
        }

        to_ask_about = [
            MembershipQuestion(
                name=part,
                part=part,
                shown=[part] + sorted(wholes),
                candidates=[wholes[name] for name in sorted(wholes)],
                ontology=self.ontology_slice([part] + sorted(wholes), labels, classes),
                measured=self.measured_of([part] + sorted(wholes), measured),
            )
            for part, wholes in sorted(candidates.items())
            if len(wholes) > 1
        ]
        forced = [
            ForcedMembership(
                part=part,
                whole=next(iter(wholes)),
                field_name=next(iter(wholes.values())).field_name,
                shared_faces=next(iter(wholes.values())).shared_faces,
                touching_edges=next(iter(wholes.values())).touching_edges,
                distance=next(iter(wholes.values())).distance,
            )
            for part, wholes in sorted(candidates.items())
            if len(wholes) == 1
        ]
        return Memberships(to_ask_about=to_ask_about, forced=forced)

    # %% saying what was measured

    def report(self, relations: Relations) -> None:
        """
        Say what the ontology made of the pairs.

        :param relations: The pairs, as they were written out.
        """
        taxonomy = self.run.read_json(RunFile.TAXONOMY)
        self.logger.info(
            "ontology: %s classes, %s mixins to compose new ones from",
            len(taxonomy["classes"]),
            len(taxonomy["part_whole_mixins"]),
        )
        counted = Counter(pair.status for pair in relations.pairs)
        self.logger.info("what the ontology makes of the pairs:")
        for status, count in counted.most_common():
            self.logger.info("  %-20s %s", status.value, count)
        self.logger.info("written to %s", self.run.directory)


# %% measuring again, knowing the classes


@dataclass
class FindOpenQuestions(MeasureScene):
    """
    The same measurement read again once the labels have been answered, saying what the
    ontology admits between them and what is left open, with one render per open
    question.
    """

    @property
    def name(self) -> str:
        return "measure again, knowing the classes"

    @property
    def rewrites_first_pass(self) -> bool:
        """
        :return: Whether this writes over what the first measurement wrote, which it
            does: it is what an answered vocabulary makes of the same measurement.
        """
        return True

    def vocabulary(self) -> Vocabulary:
        """
        :return: What each label was answered to mean.
        """
        return self.run.read_record(RunFile.VOCABULARY, Vocabulary)

    @property
    def renders_exemplars(self) -> bool:
        """
        :return: Whether this renders exemplars, which the first measurement already did.
        """
        return False

    @property
    def renders_questions(self) -> bool:
        """
        :return: Whether this renders the open questions, which only a run shown the
            contested faces needs.
        """
        return self.settings.show_the_contested_faces

    def draw(
        self,
        loader: WarsawWorldLoader,
        measured: SegmentRelations,
        segments: Dict[str, LabelSegment],
        request: VocabularyRequest,
        open_now: WhatIsOpen,
    ) -> None:
        """
        Render every open question, which the adjudication puts them with.

        :param loader: The loaded scene.
        :param measured: The measured scene.
        :param segments: The scene's segments by name.
        :param request: The vocabulary question, answered already.
        :param open_now: What the measurement leaves open, whose questions the renders
            are filled into.
        """
        if not self.renders_questions:
            return
        self.render_questions(
            loader, measured, segments, open_now.questions, open_now.contested
        )
