"""
The directory one run of the pipeline writes everything into.

A run answers everything from the scene it was given and reads nothing an earlier run
concluded. Two runs of the same scene may disagree, and a later one silently inheriting
half of an earlier one's answers would be neither of them. So every run has a directory of
its own, named for when it started, and every file it reads is in it -- the reading of the
ontology included, which is written afresh at the start of a run once the ontology has
been put back to what is committed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Type, TypeVar

from experiments.warsaw.exceptions import RunOutputAlreadyWrittenError
from experiments.warsaw.bases import JsonRecord

RecordType = TypeVar("RecordType", bound=JsonRecord)
"""
One of the records a run writes into its directory and reads back.
"""

# %% everything a run writes


class RunFile(StrEnum):
    """
    Everything a run writes, named rather than spelled out where it is read.

    A step reads what the step before it wrote, so each of these names is read in one
    place and written in another; spelling it twice leaves nothing to rename and nothing
    to fail when the two drift apart.
    """

    TAXONOMY = "taxonomy.json"
    """
    The ontology as a model reads it: classes, what each can hold, which are abstract.
    """

    PREPARED_ONTOLOGY = "prepared_ontology.json"
    """
    How many classes the run's reading of the ontology holds, and how many tables were
    built for it.
    """

    RELATIONS = "relations.json"
    """
    How the scene's labelled objects were measured to meet, and what the ontology makes
    of each pair.
    """

    VOCABULARY_REQUEST = "vocabulary_request.json"
    """
    The question asking which class each of the scene's labels means.
    """

    VOCABULARY = "vocabulary.json"
    """
    What was answered about each label.
    """

    QUESTIONS = "questions.json"
    """
    What the measurements and the ontology leave open.
    """

    ADJUDICATIONS = "adjudications.json"
    """
    What was answered about each of those.
    """

    TAXONOMY_AMENDMENTS = "taxonomy_amendments.json"
    """
    The mixins the scene's measurements raised, and whether each was accepted.
    """

    SPLIT = "split.json"
    """
    What the split built, what it cost, and the mounts carried past it.
    """

    SPLIT_FACES = "split_faces.npz"
    """
    The faces each body is made of, keyed by the name it carries everywhere else.
    """

    CLASSIFICATIONS = "classifications.json"
    """
    What each body was answered to be.
    """

    INSPECTOR = "inspect_world.py"
    """
    The script the run leaves behind so its world can be opened without knowing anything.
    """

    PUBLISHER = "publish_world.py"
    """
    The script the run leaves behind so its bodies can be picked apart in RViz.
    """

    WORLD_GRAPH_PAGE = "world_graph.html"
    """
    The world's entities and relations as an interactive graph, written when the inspector
    is asked for it.
    """

    WORLD_MESH = "world_mesh"
    """
    The annotated world as a scene a modelling tool can open, one named object per body.

    Kept because the world itself lives only in the database, reachable only through an
    ORM built for the classes this run generated. Once one of those classes is taken into
    the ontology the two cannot be told apart and the run can no longer be read, so this
    is what is left of it.
    """

    REPORT = "report.md"
    """
    What the run made, gathered from what its steps wrote.
    """

    EXEMPLARS = "exemplars"
    """
    One render per label, for the question asking what the label means.
    """

    QUESTION_RENDERS = "questions"
    """
    One render per open question.
    """

    CLASSIFICATION_RENDERS = "classifications"
    """
    The room with a group of bodies painted, for the question naming them.
    """

    MESHES = "meshes"
    """
    The geometry each body was built from. A world in the database points at these, so
    they outlive the process that wrote them.
    """

    VOCABULARY_ANSWERS = "vocabulary_answers"
    """
    The replies to the vocabulary questions, as they came back.
    """

    QUESTION_ANSWERS = "question_answers"
    """
    The replies to the adjudication questions, as they came back.
    """

    CLASSIFICATION_ANSWERS = "classification_answers"
    """
    The replies to the classification questions, as they came back.
    """

    SUPERCLASS_ANSWERS = "superclass_answers"
    """
    What each class a run proposed was settled to be a kind of.
    """

    AMENDMENT_ANSWERS = "amendment_answers"
    """
    The replies to the amendment questions, as they came back.
    """

    MODEL_CALLS = "model_calls"
    """Every individual model request, response, timing, and validation result."""

    EVALUATION_GRAPH = "evaluation_graph.json"
    """The final semantic nodes and relation outcomes, independent of the database."""

    STEP_DURATIONS = "step_durations.json"
    """
    How long each step of the run took.
    """

    SCENE = "scene"
    """
    What the scene said about itself, besides the mesh.

    Copied in rather than pointed at. A converted scene carries the building and the room
    it was cut from, and which way up it is written; an evaluation reads a run back
    against those, and a run that only named the directory would stop meaning anything
    the moment that directory was converted over.
    """

    PROVENANCE = "provenance.json"
    """Settings, input hashes, interpreter identity, and source revision for the run."""

    PYTHON_ENVIRONMENT = "python_environment.txt"
    """Installed Python distribution versions at the start of the run."""

    SOURCE_PATCH = "source.patch"
    """Tracked local source changes relative to the recorded Git commit."""


# %% the directory a run is


@dataclass
class Run:
    """
    One run's directory, and the files in it.
    """

    directory: Path
    """
    Where the run writes.
    """

    @classmethod
    def create(cls, runs_directory: Path, name_format: str = "%Y-%m-%d_%H%M%S") -> Run:
        """
        Make a directory for a run to write everything into.

        :param runs_directory: Where to make it.
        :param name_format: How to name it. The default names a run for when it started,
            so runs sort by age and none can be mistaken for another.
        :return: The run, its directory freshly made and empty.
        """
        directory = Path(runs_directory) / datetime.now().strftime(name_format)
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory=directory)

    def path(self, run_file: RunFile) -> Path:
        """
        :param run_file: The file or directory to resolve.
        :return: Where it is in this run.
        """
        return self.directory / run_file.value

    def directory_for(self, run_file: RunFile) -> Path:
        """
        :param run_file: The directory to resolve.
        :return: It, made if it was not there.
        """
        made = self.path(run_file)
        made.mkdir(parents=True, exist_ok=True)
        return made

    @property
    def scene_directory(self) -> Path:
        """
        :return: The directory this run read its scene from, as its provenance records
            it.
        """
        return Path(self.read_json(RunFile.PROVENANCE)["settings"]["scene_directory"])

    def holds(self, run_file: RunFile) -> bool:
        """
        :param run_file: The file to look for.
        :return: Whether this run has written it.
        """
        return self.path(run_file).exists()

    def read_json(self, run_file: RunFile) -> Any:
        """
        :param run_file: The file to read.
        :return: What it holds.
        """
        return json.loads(self.path(run_file).read_text())

    def read_json_if_written(self, run_file: RunFile) -> Any:
        """
        Read a file a run may not have got as far as writing.

        :param run_file: The file to read.
        :return: What it holds, or an empty mapping when the run never wrote it.
        """
        return self.read_json(run_file) if self.holds(run_file) else {}

    def read_record(self, run_file: RunFile, reader: Type[RecordType]) -> RecordType:
        """
        :param run_file: The file to read.
        :param reader: The record it holds.
        :return: What it holds, as that record.
        """
        return reader.from_json(self.read_json(run_file))

    def read_record_if_written(
        self, run_file: RunFile, otherwise: RecordType
    ) -> RecordType:
        """
        Read a record from a file a run may not have got as far as writing.

        A report is asked for about runs that stopped halfway as much as about runs that
        finished, so a file that is not there stands for a record that holds nothing
        rather than for an error.

        :param run_file: The file to read.
        :param otherwise: What to stand in with when the run never wrote it.
        :return: What it holds, or what was given to stand in.
        """
        if not self.holds(run_file):
            return otherwise
        return type(otherwise).from_json(self.read_json(run_file))

    def write_record(self, run_file: RunFile, record: JsonRecord) -> Path:
        """
        :param run_file: The file to write.
        :param record: What to write.
        :return: The file written.
        """
        return self.write_json(run_file, record.to_json())

    def write_json(self, run_file: RunFile, content: Any) -> Path:
        """
        :param run_file: The file to write.
        :param content: What to write, as JSON-ready data.
        :return: The file written.
        """
        written = self.path(run_file)
        written.write_text(json.dumps(content, indent=2))
        return written

    def refuse_to_write_over(self, *written: RunFile, overwrite: bool = False) -> None:
        """
        Refuse to write over output a step has already written, unless told to.

        Asked about the files the step itself writes rather than about the directory: a
        run's directory holds the ontology it read out before its first step has written
        anything, and every step after that writes beside what the ones before it wrote.

        Overriding is another matter and is allowed when asked for: a step run twice in
        one run, the second time knowing something the first did not, writes what it wrote
        again rather than adding to it.

        :param written: The files the step is about to write.
        :param overwrite: Whether what is there is to be written over.
        :raises RunOutputAlreadyWrittenError: If any of them is there and overwriting was
            not asked for.
        """
        standing = [one.value for one in written if self.holds(one)]
        if overwrite or not standing:
            return
        raise RunOutputAlreadyWrittenError(directory=self.directory, written=standing)
