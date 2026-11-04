"""
What one step of the pipeline hands to the next.

Every step reads what the step before it wrote, so each field name is written in one
module and read in another. Mirroring each file in a dataclass writes those names once:
the reader and the writer are the same declaration, and a field that moves moves for both.

A record is a dataclass and its fields say what it holds, so krrood's serializer is what
reads and writes it; the mapping is not written out anywhere.

Where several records say the same thing -- both questions about an overlap, both answers
settling one, both answers naming a class -- they say it once, in a class they share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, ClassVar, Dict, List, Optional, Self, Set, Tuple


from experiments.warsaw.bases import JsonRecord
from experiments.warsaw.scene_split import Pairing
from experiments.warsaw.segment_relations import (
    SegmentRelations,
    ClaimedFaces,
    PairEvidence,
    SegmentDescriptor,
)

# %% named alternatives


class RelationStatus(StrEnum):
    """
    What the ontology makes of two labels measured to meet.
    """

    CLASS_UNKNOWN = "class-unknown"
    """
    Neither label has been mapped to a class yet, so the ontology has nothing to say.
    """

    NO_LEGAL_RELATION = "no-legal-relation"
    """
    The classes cannot hold one another as a part, so an overlap is something else.
    """

    RELATION_KNOWN = "relation-known"
    """
    Exactly one part-whole relation is admissible, so only the pair itself is in
    question.
    """

    RELATION_AMBIGUOUS = "relation-ambiguous"
    """
    Several relations are admissible, so which field a mount would use is in question
    too.
    """


# %% what every artefact says about itself


@dataclass
class RunArtefact(JsonRecord):
    """
    Something a step wrote about one scene.
    """

    scene: str
    """
    The mesh it is about.
    """


@dataclass
class AnsweredByModel(RunArtefact):
    """
    Something a step wrote down after asking a model.

    Which model answered is part of the artefact rather than of the run: a run may ask
    different steps of different models, and two runs are only comparable where the same
    model answered the same question.
    """

    model: str
    """
    The model that was asked.
    """


# %% what a model says, whatever it was asked


@dataclass(kw_only=True)
class ModelAnswer(JsonRecord):
    """
    What a model said, over and above the answer itself.

    Five steps ask a model something and each reads a different answer out of the reply,
    but every one of them asks for the same three things around it: how sure it is, why,
    and -- filled in here rather than by the model -- what makes what it said unusable.
    """

    confidence: Optional[float] = None
    """
    How sure the model said it was.
    """

    reason: Optional[str] = None
    """
    Why, in one sentence.
    """

    problems: List[str] = field(default_factory=list)
    """
    What makes the answer unusable, empty when nothing does.
    """


# %% an answer that names a class


@dataclass
class ClassAnswer(ModelAnswer):
    """
    An answer naming the class of something, whether a label or a single body.

    The same thing is asked twice in a run and about two different subjects: once about a
    label, which covers many objects, and once about each body in the room it stands in.
    Both come back as a class of the ontology or as one proposed by naming what it derives
    from, so both are read the same way.
    """

    label: Optional[str] = None
    """
    The scan's word for it, which is what was asked about or what the body was labelled.
    """

    class_name: Optional[str] = None
    """
    The class it is, or None where the ontology should hold nothing for it.
    """

    is_new_class: bool = False
    """
    Whether that class is proposed rather than found in the ontology.
    """

    superclass: Optional[str] = None
    """
    What a proposed class derives from.
    """

    spoken_class: ClassVar[str] = "class"
    """
    What the class is called in the words a model is asked to answer in.
    """

    @classmethod
    def spoken(cls, payload: Dict[str, Any]) -> Self:
        """
        Read an answer as the model wrote it, rather than as this record stores it.

        A model is asked for ``class``, because that is what the thing is called; the
        field is ``class_name``, because ``class`` is a Python keyword. The prompt cannot
        be written in the field's words, so the two are reconciled here -- once, for every
        answer that names a class, and at the only place a model's words come in.

        :param payload: What the model said, in the words it was asked for.
        :return: It, as an answer.
        """
        said = dict(payload)
        if cls.spoken_class in said:
            said["class_name"] = said.pop(cls.spoken_class)
        return cls.from_json(said)


# %% what the ontology admits between two classes


@dataclass
class AdmissibleMount(JsonRecord):
    """
    One way the ontology allows one of two classes to hold the other as a part.
    """

    whole: str
    """
    The class that would hold.
    """

    field_name: str
    """
    The field it would be held in.
    """

    target: str
    """
    What that field accepts.
    """

    holds_many: bool
    """
    Whether that field holds several of them rather than one.
    """

    removes_geometry: bool
    """
    Whether mounting takes what is held out of the geometry of what holds it.
    """


@dataclass
class OntologyView(JsonRecord):
    """
    What the ontology admits between the classes of two segments.

    ..note:: This is what is *admissible*, never what is the case: whether this cabinet
        holds this drawer is a question about the two objects, which no amount of reading
        the taxonomy answers.
    """

    status: RelationStatus
    """
    What that leaves open.
    """

    mounts: List[AdmissibleMount] = field(default_factory=list)
    """
    Every mount allowed between them.
    """


@dataclass
class OntologySlice(JsonRecord):
    """
    What the taxonomy holds about the handful of objects one question is about.

    The slice rather than the whole taxonomy: a question about three objects is not helped
    by a hundred and thirty-nine classes.
    """

    read_as: Dict[str, Optional[str]] = field(default_factory=dict)
    """
    Per segment, the class it was read as.
    """

    classes: List[str] = field(default_factory=list)
    """
    Each of those classes, written out.
    """

    admits: List[str] = field(default_factory=list)
    """
    What the ontology admits between them.
    """


# %% what the ontology settles about an overlap on its own


@dataclass
class CountedClaimants(JsonRecord):
    """
    A set of faces claimed by exactly the same segments, counted rather than listed.
    """

    claimants: Tuple[str, ...]
    """
    The segments claiming them.
    """

    faces: int
    """
    How many faces they all claim.
    """

    @classmethod
    def of(cls, group: ClaimedFaces) -> CountedClaimants:
        """
        Say what a measured group of claimants holds, without the faces themselves.

        A group carries *which* faces are contested, which is what the split works on and
        what no file can hold; a run records how many there are. They are the same set
        counted two ways, so the one that can be written is made from the one that cannot.

        :param group: The claimants as the measurement found them.
        :return: The same claimants, counted.
        """
        return cls(claimants=group.names, faces=int(len(group.faces)))


@dataclass(kw_only=True)
class MeasuredMeeting(JsonRecord):
    """
    How a part was measured to meet a whole it could belong to.

    The same measurements decide it whether or not there is anything to choose between:
    one candidate makes it a membership nothing needs to be asked about, several make it a
    question, and both carry the same numbers.
    """

    field_name: str
    """
    The field the whole would hold it in.
    """

    shared_faces: int = 0
    """
    How many faces the two share.
    """

    touching_edges: int = 0
    """
    How many edges they touch along.
    """

    distance: float = 0.0
    """
    How far apart their surfaces are, in metres.
    """


@dataclass(kw_only=True)
class ForcedMembership(MeasuredMeeting):
    """
    A part that meets exactly one candidate, so there is nothing to choose between.
    """

    part: str
    """
    The object that belongs to it.
    """

    whole: str
    """
    The one object it could belong to.
    """


# %% prepared_ontology.json


@dataclass
class PreparedOntology(JsonRecord):
    """
    What reading the ontology out and building its tables produced.
    """

    classes: int
    """
    How many classes the run may name.
    """

    mixins: int
    """
    How many of them a new class can be composed from.
    """

    tables: int
    """
    How many tables were built in the run's schema.
    """


# %% relations.json


@dataclass
class PairRecord(JsonRecord):
    """
    Two segments measured to meet, and what the ontology makes of them.
    """

    evidence: PairEvidence
    """
    How they were measured to meet.
    """

    classes: Dict[str, Optional[str]] = field(default_factory=dict)
    """
    Per label the two carry, the class it was read as.
    """

    view: Optional[OntologyView] = None
    """
    What the ontology admits between those classes.
    """

    prompt_block: str = ""
    """
    The measurements as the text a model reads them in.
    """

    @property
    def status(self) -> RelationStatus:
        """
        :return: What the ontology leaves open about the pair.
        """
        return self.view.status if self.view else RelationStatus.CLASS_UNKNOWN


@dataclass
class Relations(RunArtefact):
    """
    How a scene's labelled objects were measured to meet.
    """

    segments: List[SegmentDescriptor] = field(default_factory=list)
    """
    What each labelled object is, measured rather than judged.
    """

    pairs: List[PairRecord] = field(default_factory=list)
    """
    Every pair that shares faces, touches, or is among a segment's nearest.
    """

    settled: List[CountedClaimants] = field(default_factory=list)
    """
    The sets of contested faces the ontology decides on its own.
    """

    forced: List[ForcedMembership] = field(default_factory=list)
    """
    The memberships with only one candidate, which leave nothing to choose between.
    """

    def as_measured(self) -> SegmentRelations:
        """
        :return: The measurement this record was written from, as the step that measured
            it held it.
        """
        return SegmentRelations(
            descriptors={one.name: one for one in self.segments},
            pairs=[one.evidence for one in self.pairs],
        )

    @property
    def settled_claimants(self) -> Set[Tuple[str, ...]]:
        """
        :return: The claimant sets the ontology decides, as their names alone.
        """
        return {one.claimants for one in self.settled}

    @property
    def descriptors(self) -> Dict[str, SegmentDescriptor]:
        """
        :return: The segments by name.
        """
        return {descriptor.name: descriptor for descriptor in self.segments}

    @property
    def labels(self) -> Dict[str, str]:
        """
        :return: Per segment, the label it carries.
        """
        return {descriptor.name: descriptor.class_name for descriptor in self.segments}


# %% vocabulary_request.json


@dataclass
class LabelRequest(JsonRecord):
    """
    One label of a scene, and the object standing for it.
    """

    label: str
    """
    The word the scene's annotator chose.
    """

    instances: int
    """
    How many objects carry it.
    """

    exemplar: str
    """
    The one standing for it, chosen as the one whose surface is least claimed by others.
    """

    exemplar_faces: int = 0
    """
    How many faces that one is made of.
    """

    exemplar_exclusive_share: float = 0.0
    """
    The share of those no other segment claims.
    """

    exemplar_exclusive_area: float = 0.0
    """
    How much of its area no other segment claims.
    """

    images: List[str] = field(default_factory=list)
    """
    The renders of it, by filename.
    """

    color: Optional[str] = None
    """
    What it was painted in those renders.
    """


@dataclass
class VocabularyRequest(RunArtefact):
    """
    The question asking which class each of a scene's labels means.
    """

    question: str
    """
    What is being asked.
    """

    labels: List[LabelRequest] = field(default_factory=list)
    """
    One entry per label.
    """

    @property
    def label_names(self) -> List[str]:
        """
        :return: Every label of the scene, which is what says what each was left to mean:
            a room that labels handles separately does not mean them by ``drawer``.
        """
        return [entry.label for entry in self.labels]


# %% vocabulary.json


@dataclass
class SuperclassAnswer(ModelAnswer):
    """
    What one proposed class was settled to be a kind of.
    """

    class_name: str = ""
    """
    The class being placed.
    """

    superclass: Optional[str] = None
    """
    What it is a kind of, which is what the objects *are*.
    """

    mixins: List[str] = field(default_factory=list)
    """
    What it can hold, which is a different question from what it is.
    """


@dataclass
class LabelAnswer(ClassAnswer):
    """
    What was answered about one label.
    """

    mixins: List[str] = field(default_factory=list)
    """
    What a proposed class is composed with, which decides what it can hold.
    """

    exemplar: Optional[str] = None
    """
    The object that was pictured when it was asked.
    """

    @property
    def is_usable(self) -> bool:
        """
        :return: Whether the answer names a class and nothing is wrong with it.
        """
        return bool(self.class_name) and not self.problems

    @classmethod
    def of(cls, payload: Any) -> LabelAnswer:
        """
        Read an answer however it was written.

        A mapping written by hand to try something out names the class and nothing else,
        so a bare name and a null are read as answers too.

        :param payload: The answer, as a mapping or as the class name alone.
        :return: It, as an answer.
        """
        if payload is None or isinstance(payload, str):
            return cls(class_name=payload or None)
        return cls.spoken(payload)


@dataclass
class Vocabulary(AnsweredByModel):
    """
    What each of a scene's labels was answered to mean.
    """

    labels: List[LabelAnswer] = field(default_factory=list)
    """
    One answer per label.
    """

    @property
    def by_label(self) -> Dict[str, LabelAnswer]:
        """
        :return: The answers by the label each is about.
        """
        return {answer.label: answer for answer in self.labels if answer.label}

    def answer_for(self, label: str) -> LabelAnswer:
        """
        :param label: The label to look up.
        :return: What was answered about it, blank where nothing was.
        """
        return self.by_label.get(label, LabelAnswer())

    @property
    def proposals(self) -> List[LabelAnswer]:
        """
        :return: The answers that proposed a class rather than naming one.
        """
        return [
            answer
            for answer in self.labels
            if answer.is_new_class and answer.class_name
        ]


# %% what one of a scene's objects was measured to be


@dataclass
class NamedSegment(JsonRecord):
    """
    One of the scene's objects, named and sized.

    Everything a run writes addresses an object by the name the scan gave it, and how much
    of the scene an object holds is what decides almost every question asked about it, so
    the two travel together.
    """

    name: str
    """
    The object it is about.
    """

    faces: int
    """
    How many of the scene's faces it holds.
    """


@dataclass
class MeasuredSegment(NamedSegment):
    """
    What was measured of one object on its own.
    """

    area: float
    """
    Its surface area, in square metres.
    """

    height: float
    """
    How high its middle sits above the lowest point of the scene.
    """

    pieces: int
    """
    How many connected pieces it falls into.
    """


@dataclass
class ContestedShare(NamedSegment):
    """
    How much of one claimant the contested faces are.

    Without it the picture is all a reader has, and a picture cannot be read when one
    claimant is twenty times the size of the others: an island label covers the whole block
    including its drawers, so a drawer front reads as a patch of detail on the island
    rather than as the drawer.
    """

    contested_share: float
    """
    What share of them are contested.
    """


# %% questions.json


@dataclass(kw_only=True)
class OverlapQuestion(JsonRecord):
    """
    What is put to a model about a set of objects the scan drew over one another.

    Two things are asked about an overlap and they are asked the same way: whose the
    contested faces are, and which whole a part belongs to. Both show the same pictures of
    the same objects, say what the ontology admits between them, and say what each was
    measured to be, so all of that is written down once.
    """

    name: str
    """
    What the question is filed under, which is what its answer names back.
    """

    shown: List[str] = field(default_factory=list)
    """
    The segments in the pictures.
    """

    ontology: OntologySlice = field(default_factory=lambda: OntologySlice())
    """
    What the ontology admits between their classes.
    """

    measured: List[MeasuredSegment] = field(default_factory=list)
    """
    What was measured of each of them on its own.
    """

    images: List[str] = field(default_factory=list)
    """
    The renders of them, by filename.
    """

    legend: Dict[str, str] = field(default_factory=dict)
    """
    Per segment, what it was painted in those renders.
    """


@dataclass(kw_only=True)
class OwnershipQuestion(OverlapQuestion):
    """
    Whose surface a set of faces several labels claim is.

    Asked once per *pattern* of classes rather than once per occurrence: a door and a
    window sharing a pane is one question however many glazed doors the room has.
    """

    pattern: List[str] = field(default_factory=list)
    """
    The labels that meet like this.
    """

    covers: List[CountedClaimants] = field(default_factory=list)
    """
    Every set of faces this one question decides.
    """

    contested_faces: int = 0
    """
    How many faces those sets hold between them.
    """

    exemplar_faces: int = 0
    """
    How many the pictured set holds.
    """

    shares: List[ContestedShare] = field(default_factory=list)
    """
    Per claimant, how much of it the contested faces are.
    """


@dataclass(kw_only=True)
class MembershipCandidate(MeasuredMeeting):
    """
    One object a part could belong to, and how it was measured to meet it.
    """

    name: str
    """
    The object the part could belong to.
    """


@dataclass(kw_only=True)
class MembershipQuestion(OverlapQuestion):
    """
    Which whole a part belongs to, asked only where it meets more than one candidate.
    """

    part: str
    """
    The object that belongs to one of the candidates.
    """

    candidates: List[MembershipCandidate] = field(default_factory=list)
    """
    The wholes it could belong to.
    """

    @property
    def candidate_names(self) -> List[str]:
        """
        :return: The objects the part could belong to, which is what is being chosen
            between and so what an answer has to name one of.
        """
        return [candidate.name for candidate in self.candidates]


@dataclass
class OpenQuestions(RunArtefact):
    """
    What the measurements and the ontology leave open about a scene's overlaps.

    Both kinds are asked the same way and most readers want them together, so they are
    written as one list; each carries which kind it is, and the two that want one kind
    read it back out.
    """

    asked: List[OverlapQuestion] = field(default_factory=list)
    """
    Every question the overlaps raise, of either kind.
    """

    @property
    def ownership(self) -> List[OwnershipQuestion]:
        """
        :return: Whose the contested faces are, once per class pattern.
        """
        return [one for one in self.asked if isinstance(one, OwnershipQuestion)]

    @property
    def membership(self) -> List[MembershipQuestion]:
        """
        :return: Which whole each part belongs to, where more than one is possible.
        """
        return [one for one in self.asked if isinstance(one, MembershipQuestion)]


# %% adjudications.json


@dataclass(kw_only=True)
class Adjudication(ModelAnswer):
    """
    One open question, settled.

    An answer carries the question it answers rather than pointing at it, so a run's
    adjudications can be read on their own: what was decided sits beside what it was
    decided about.
    """

    name: str
    """
    The question it answers, by the name that question was filed under.
    """


@dataclass(kw_only=True)
class OwnershipAnswer(Adjudication):
    """
    Whose the contested faces of one class pattern are.

    The pattern is carried rather than pointed at, so an answer says both what it decided
    and what it decided about; which sets of faces that reaches is the question's to say,
    and is read from there.
    """

    pattern: List[str] = field(default_factory=list)
    """
    The labels that meet like this.
    """

    owner: Optional[str] = None
    """
    The label whose surface those faces are.
    """


@dataclass(kw_only=True)
class MembershipAnswer(Adjudication):
    """
    Which whole one part belongs to.
    """

    part: str
    """
    The object that belongs somewhere.
    """

    whole: Optional[str] = None
    """
    The object it belongs to.
    """


@dataclass
class Adjudications(AnsweredByModel):
    """
    What was answered about everything the measurements and the ontology left open.
    """

    answered: List[Adjudication] = field(default_factory=list)
    """
    Every question that was put, as it was settled, of either kind.
    """

    @property
    def ownership(self) -> List[OwnershipAnswer]:
        """
        :return: Whose the contested faces of each class pattern are.
        """
        return [one for one in self.answered if isinstance(one, OwnershipAnswer)]

    @property
    def membership(self) -> List[MembershipAnswer]:
        """
        :return: Which whole each part belongs to.
        """
        return [one for one in self.answered if isinstance(one, MembershipAnswer)]

    @property
    def owner_by_pattern(self) -> Dict[Tuple[str, ...], str]:
        """
        :return: Per class pattern, the label the faces belong to.
        """
        return {
            tuple(answer.pattern): answer.owner
            for answer in self.ownership
            if answer.owner
        }


# %% split.json


@dataclass
class TakenFaces(JsonRecord):
    """
    How many of one segment's faces a single owner took.
    """

    name: str
    """
    The body that took them.
    """

    faces: int
    """
    How many it took.
    """


@dataclass
class EmptiedSegment(JsonRecord):
    """
    A labelled object the split left with nothing, and where its faces went.

    An object that vanished is usually the interesting thing about a run: it says the
    scan labelled something that the ontology, or an answer, decided was really part of
    its neighbour.
    """

    name: str
    """
    The segment that was emptied.
    """

    taken_by: List[TakenFaces] = field(default_factory=list)
    """
    Per body that took some of its faces, how many it took.
    """


@dataclass
class SplitBody(NamedSegment):
    """
    One body the split built.
    """

    label: str
    """
    The label the scene gave it.
    """

    body_id: Optional[str] = None
    """
    The id the world addresses it by.

    The name is what everything else addresses a body by, and the world addresses it by
    an id of its own; a step reading the world back needs both to say the same thing.
    """


@dataclass
class RefusedMount(JsonRecord):
    """
    One mount the world would not carry out, and why.

    A pairing every earlier step agreed on can still be refused here, because the class a
    body was given may not admit the part another step said it holds. That is a part left
    where the split put it rather than where the run decided it belongs, so it is recorded
    beside the objects the split emptied and for the same reason.
    """

    pairing: Pairing
    """
    The mount that was refused.
    """

    reason: str
    """
    What the world said about it.
    """


@dataclass
class SplitRecord(RunArtefact):
    """
    What the split built, what it cost, and the mounts carried past it.
    """

    bodies: List[SplitBody] = field(default_factory=list)
    """
    Every body the split built.
    """

    emptied: List[EmptiedSegment] = field(default_factory=list)
    """
    Every segment left with no faces, and where they went.
    """

    still_contested: int = 0
    """
    How many faces are still claimed twice, which should be none.
    """

    pairings: List[Pairing] = field(default_factory=list)
    """
    The mounts that still have both ends.
    """

    refused: List[RefusedMount] = field(default_factory=list)
    """
    The mounts the world would not carry out, once the last step has tried them.
    """

    world_id: Optional[int] = None
    """
    The world the split was written to.
    """

    world_T_source: Optional[List[List[float]]] = None
    """
    The turn from the scene file's coordinates into the frame the bodies were built in.
    Nothing in runs written before scans were stood on their floor, which turned a scene
    by its :class:`~experiments.warsaw.world_loader.scene.SourceFrame` alone.
    """

    annotated_world_id: Optional[int] = None
    """
    The world the annotations were written to, once the last step has run.
    """


# %% classifications.json


@dataclass
class BodyAnswer(ClassAnswer):
    """
    What one body was answered to be.
    """

    name: Optional[str] = None
    """
    The body it is about, which is what a model is asked to name it by.
    """

    faces: int = 0
    """
    How many faces the body is made of.
    """


@dataclass
class Classifications(AnsweredByModel):
    """
    What each body of a split scene was answered to be.
    """

    bodies: List[BodyAnswer] = field(default_factory=list)
    """
    One answer per body.
    """


# %% taxonomy_amendments.json


@dataclass
class SourceEdit(JsonRecord):
    """
    One line of the ontology's own source as it was and as it became.
    """

    file: str
    """
    The file that holds it.
    """

    line: int
    """
    Which line.
    """

    before: str
    """
    What stood there.
    """

    after: str
    """
    What was written instead.
    """


@dataclass
class AmendmentRecord(ModelAnswer):
    """
    One mixin a class could be given, what raised it, and what became of the proposal.
    """

    whole: str
    """
    The class that would hold the part.
    """

    mixin: str
    """
    The mixin that would let it.
    """

    part: str
    """
    The class that would be held.
    """

    whole_labels: List[str] = field(default_factory=list)
    """
    The scene's labels that were read as the holding class.
    """

    part_labels: List[str] = field(default_factory=list)
    """
    The scene's labels that were read as the part.
    """

    measured_pairs: int = 0
    """
    How many measured pairs of overlapping objects raised it.
    """

    shared_faces: int = 0
    """
    How many faces those pairs share in total.
    """

    amend: bool = False
    """
    Whether the model said the class should have the mixin.
    """

    blocked: Optional[str] = None
    """
    Why the accepted amendment could not be written, when it could not.
    """

    edit: Optional[SourceEdit] = None
    """
    The line it would change.
    """

    applied: bool = False
    """
    Whether it is in force right now.
    """

    reverted: bool = False
    """
    Whether it was put back after having been in force.
    """


# %% how long a run spent


@dataclass
class StepDuration(JsonRecord):
    """
    How long one step of a run took.
    """

    step: str
    """
    The step it was spent in, under the name the run announced it by.
    """

    seconds: float
    """
    How long that took.
    """


@dataclass
class StepDurations(RunArtefact):
    """
    How long each step of a run took.

    Written again after every step rather than once at the end, so that a run which
    stopped halfway still says where its time went.
    """

    steps: List[StepDuration] = field(default_factory=list)
    """
    Each step that has finished, in the order the run carried them out.
    """

    @property
    def total_seconds(self) -> float:
        """
        :return: How long the run spent in the steps recorded so far.
        """
        return sum(one.seconds for one in self.steps)
