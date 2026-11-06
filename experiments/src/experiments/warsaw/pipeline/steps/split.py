"""
Cut a scene's single body into one body per labelled object.

Everything the cut needs was decided by the steps before it: which class each label
means, who owns the faces several labels claim, and which whole each part belongs to.
This applies those decisions and builds the world.

The world is built even where objects end up with nothing, and every such loss is
reported beside the answer that caused it: an ownership answer is given once per class
pattern, so one wrong answer empties every object of that kind at once, and a report
that merely said ten cabinets are missing would not say why.

The pairings are carried out of the split rather than measured again from the bodies,
because the overlap that says a handle is on *this* drawer is gone the moment the faces
stop being shared.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from semantic_digital_twin.world import World
from typing_extensions import Dict, List, Optional, Tuple, Type

from experiments.warsaw.pipeline.label_classes import VocabularyClasses
from experiments.warsaw.pipeline.records import (
    Adjudications,
    EmptiedSegment,
    Relations,
    SplitBody,
    SplitRecord,
    TakenFaces,
    Vocabulary,
)
from experiments.warsaw.pipeline.run import RunFile
from experiments.warsaw.pipeline.steps.step import PipelineStep
from experiments.warsaw.pipeline.database.world_store import WorldStore
from experiments.warsaw.scene_split import (
    FaceDivision,
    Ownership,
    Pairing,
    SplitFaces,
    exclusive_faces,
    owners_by_ontology,
    pairings,
    split_world,
)
from experiments.warsaw.segment_relations import ClaimedFaces, claimant_groups
from experiments.warsaw.world_loader.loader import WarsawWorldLoader


@dataclass
class SplitScene(PipelineStep):
    """
    One body per labelled object, built from the decisions the run has already made.
    """

    contested_shown: int = 5
    """
    How many still-contested faces to name before counting the rest.

    Applying one decision per set of claimants cannot leave any, so anything here means
    a set was reached by no decision rather than that two decisions disagreed.
    """

    unreached_shown: int = 5
    """
    How many sets of faces no answer reached to name before counting the rest.
    """

    # %% the step

    @property
    def name(self) -> str:
        return "cut the scene into bodies"

    def carry_out(self) -> None:
        """
        Apply the decisions, build the bodies, and record what it cost.
        """
        adjudications = self.run.read_record(RunFile.ADJUDICATIONS, Adjudications)
        relations = self.run.read_record(RunFile.RELATIONS, Relations)
        vocabulary = self.run.read_record(RunFile.VOCABULARY, Vocabulary)

        loader = WarsawWorldLoader(input_directory=self.settings.scene_directory)
        segments = loader.label_segments
        labels = {str(segment.name): segment.class_name for segment in segments}
        faces = {str(segment.name): segment.face_indices for segment in segments}
        self.logger.info(
            "%s segments over %s faces", len(segments), len(loader.scene_mesh.faces)
        )

        groups = claimant_groups(
            [segment.face_indices for segment in segments],
            [str(segment.name) for segment in segments],
            len(loader.scene_mesh.faces),
        )
        classes = VocabularyClasses(
            vocabulary=vocabulary, known=self.ontology_classes()
        ).by_label()
        ownerships, unreached = self.resolve_owners(
            groups, adjudications, relations, labels, classes
        )
        self.report_unreached(unreached)
        ownerships = self.divide_ties(
            ownerships, FaceDivision(mesh=loader.scene_mesh, segment_faces=faces)
        )

        split = exclusive_faces(faces, ownerships)
        self.report_split(split, ownerships, labels)

        carried = pairings(self.named_pairings(adjudications, relations), split)
        self.logger.info("%s pairings carried past the split", len(carried))

        world = self.build_world(loader, split)
        record = self.record_of(loader, split, carried, labels, world)
        self.run.write_record(RunFile.SPLIT, record)
        self.logger.info("written to %s", self.run.path(RunFile.SPLIT))

    # %% deciding whose the contested faces are

    @staticmethod
    def resolve_owners(
        groups: List[ClaimedFaces],
        adjudications: Adjudications,
        relations: Relations,
        labels: Dict[str, str],
        classes: Dict[str, Optional[Type]],
    ) -> Tuple[List[Ownership], List[ClaimedFaces]]:
        """
        Work out who each set of contested faces belongs to.

        A set the ontology settled is answered from the ontology; the rest are answered
        by the class pattern they belong to, since that is the grain the question was
        asked at. Where several claimants carry the label the answer names, the first is
        made the owner and the others are recorded as tied with it, for the faces to be
        divided between them.

        :param groups: The sets of faces and who claims them.
        :param adjudications: What the adjudication wrote.
        :param relations: What the measurement wrote, which is where the sets the
            ontology settled are recorded.
        :param labels: Per segment, the label it carries.
        :param classes: Per label, the class it was read as.
        :return: The ownerships, and the groups no answer reached.
        """
        settled = relations.settled_claimants
        answers = adjudications.owner_by_pattern

        ownerships, unreached = [], []
        for group in groups:
            by_class = {name: classes.get(labels[name]) for name in group.names}
            settled_here = tuple(group.names) in settled
            if settled_here:
                candidates = owners_by_ontology(group.names, by_class)
            else:
                wanted = answers.get(
                    tuple(sorted(labels[name] for name in group.names))
                )
                candidates = [name for name in group.names if labels[name] == wanted]

            if len({labels[name] for name in candidates}) != 1:
                unreached.append(group)
                continue
            ownerships.append(
                Ownership(
                    names=group.names,
                    owner=candidates[0],
                    faces=group.faces,
                    settled_by_ontology=settled_here,
                    tied_with=tuple(candidates[1:]),
                )
            )
        return ownerships, unreached

    def divide_ties(
        self, ownerships: List[Ownership], division: FaceDivision
    ) -> List[Ownership]:
        """
        Divide the faces of every tied decision between the claimants it tied.

        :param ownerships: The decisions, some naming claimants tied with their owner.
        :param division: What divides a set of faces between tied claimants.
        :return: The decisions, with every tied one replaced by one per claimant.
        """
        divided: List[Ownership] = []
        tied = 0
        for ownership in ownerships:
            shares = division.divide(ownership)
            divided += shares
            if not ownership.tied_with:
                continue
            tied += 1
            if len(shares) == 1:
                self.logger.warning(
                    "  %s claim the same faces and are kept as one: %s",
                    " & ".join((ownership.owner, *ownership.tied_with)),
                    shares[0].owner,
                )
        if tied:
            self.logger.info(
                "%s sets of faces divided between objects of the same label", tied
            )
        return divided

    @staticmethod
    def named_pairings(
        adjudications: Adjudications, relations: Relations
    ) -> List[Pairing]:
        """
        :param adjudications: What the adjudication wrote.
        :param relations: What the measurement wrote, which is where the memberships
            with only one candidate are recorded.
        :return: Every mount the run named, before the split drops any.
        """
        carried = [
            Pairing(whole=forced.whole, part=forced.part, field_name=forced.field_name)
            for forced in relations.forced
        ]
        carried += [
            Pairing(
                whole=answer.whole,
                part=answer.part,
                field_name=answer.field_name,
            )
            for answer in adjudications.membership
            if answer.whole
        ]
        return carried

    # %% building the bodies

    def build_world(self, loader: WarsawWorldLoader, split: SplitFaces) -> World:
        """
        Build one body per object, writing their geometry where a stored world can find
        it.

        :param loader: The loaded scene.
        :param split: What the split left.
        :return: The world, flat, one body per object under a single root.
        """
        self.logger.info("building the bodies ...")
        # A world in the database points at the files its meshes were written to, and the
        # place they go by default is emptied when this process ends.
        directory = self.run.path(RunFile.MESHES) if self.settings.persist else None
        if directory is not None:
            self.logger.info(
                "writing the meshes to %s, since the world is to be kept", directory
            )
        world = split_world(
            loader.scene.mesh,
            split.faces,
            loader.scene.world_T_source,
            directory=directory,
        )
        self.logger.info("the world holds %s bodies", len(world.bodies))
        return world

    def record_of(
        self,
        loader: WarsawWorldLoader,
        split: SplitFaces,
        carried: List[Pairing],
        labels: Dict[str, str],
        world: World,
    ) -> SplitRecord:
        """
        Write the world where a later step can read it, and say what the split left.

        The faces each body is made of are kept beside the record: the world built here
        dies with the process, so without them the split would have to be derived again
        from the answers to be used, and a later answer would silently give a different
        partition than the one that was reported.

        :param loader: The loaded scene.
        :param split: What the split left.
        :param carried: The mounts that still have both ends.
        :param labels: Per segment, the label it carries.
        :param world: The world that was built.
        :return: The record of it all.
        """
        identities = {str(body.name.name): str(body.id) for body in world.bodies}
        world_id = WorldStore().write(world) if self.settings.persist else None
        if world_id is not None:
            self.logger.info("the world was written to the database as %s", world_id)

        faces_path = self.run.path(RunFile.SPLIT_FACES)
        np.savez_compressed(faces_path, **split.faces)
        self.logger.info("the bodies' faces written to %s", faces_path)

        return SplitRecord(
            scene=str(loader.scene.mesh_path),
            bodies=[
                SplitBody(
                    name=name,
                    faces=int(len(kept)),
                    label=labels[name],
                    body_id=identities.get(name),
                )
                for name, kept in split.faces.items()
            ],
            emptied=[
                EmptiedSegment(
                    name=name,
                    # Largest taker first: an object that lost everything usually lost
                    # it to one neighbour, and that one is which answer to look at.
                    taken_by=[
                        TakenFaces(name=owner, faces=count)
                        for owner, count in sorted(
                            split.lost_to.get(name, {}).items(),
                            key=lambda took: -took[1],
                        )
                    ],
                )
                for name in split.emptied
            ],
            still_contested=len(split.contested),
            pairings=carried,
            world_id=world_id,
            world_T_source=loader.scene.world_T_source.to_np().tolist(),
        )

    # %% saying what it cost

    def report_unreached(self, unreached: List[ClaimedFaces]) -> None:
        """
        :param unreached: The sets of faces no answer reached.
        """
        if not unreached:
            return
        self.logger.warning("%s sets of faces no answer reached:", len(unreached))
        for group in unreached[: self.unreached_shown]:
            self.logger.warning(
                "  %s (%s faces)", " & ".join(group.names), len(group.faces)
            )

    def report_split(
        self,
        split: SplitFaces,
        ownerships: List[Ownership],
        labels: Dict[str, str],
    ) -> None:
        """
        Say what the split did, and what it cost.

        :param split: What the split left.
        :param ownerships: Who each set of faces was given to.
        :param labels: Per segment, the label it carries.
        """
        self.logger.info(
            "%s bodies, %s faces between them",
            len(split.faces),
            sum(len(faces) for faces in split.faces.values()),
        )
        settled = sum(1 for one in ownerships if one.settled_by_ontology)
        self.logger.info(
            "%s sets of contested faces given away: %s by the ontology, %s by an answer",
            len(ownerships),
            settled,
            len(ownerships) - settled,
        )

        if split.contested:
            self.logger.warning(
                "%s faces are still claimed twice:", len(split.contested)
            )
            for face, names in list(split.contested.items())[: self.contested_shown]:
                self.logger.warning("  face %s: %s", face, " & ".join(names))

        if not split.emptied:
            return

        # An ownership answer is given once per class pattern, so one wrong answer empties
        # every object of that kind. Grouping the losses by who took them says which answer
        # to look at rather than which objects went missing.
        took = defaultdict(set)
        for name in split.emptied:
            for owner in split.lost_to.get(name, {}):
                took[labels[owner]].add(name)
        self.logger.warning("%s segments lost every face they had:", len(split.emptied))
        for owner, lost in sorted(took.items(), key=lambda one: -len(one[1])):
            counted = Counter(labels[name] for name in lost)
            wording = ", ".join(
                f"{count} {label}" for label, count in counted.most_common()
            )
            self.logger.warning("  everything went to a %s: %s", owner, wording)
            self.logger.warning("      %s", ", ".join(sorted(lost)))
