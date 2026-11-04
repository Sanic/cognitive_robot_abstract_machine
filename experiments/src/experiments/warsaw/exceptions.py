from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing_extensions import TYPE_CHECKING, List

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from experiments.warsaw.pipeline.steps.amend.amended_classes_check import (
        AmendedPair,
    )


@dataclass
class WarsawSceneNotFoundError(DataclassException, FileNotFoundError):
    """
    Raised when a directory holds no Warsaw scene mesh.
    """

    directory: Path
    """
    The directory that was searched.
    """

    scene_mesh_pattern: str
    """
    The pattern a scene mesh was looked for under.
    """

    def error_message(self) -> str:
        return (
            f"No Warsaw scene mesh matching '{self.scene_mesh_pattern}' in "
            f"'{self.directory}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Point the loader at a scene directory, which holds the one mesh the scene "
            "is written as. The dataset is not in the repository, so a fresh checkout "
            "has none: put the scan under 'experiments/src/experiments/warsaw/dataset/' "
            "as that package's README describes."
        )


@dataclass
class AmbiguousWarsawSceneError(DataclassException, FileNotFoundError):
    """
    Raised when a directory holds more than one mesh, leaving undecided which of them
    the scene is.
    """

    directory: Path
    """
    The directory that was searched.
    """

    scene_meshes: List[Path]
    """
    The meshes it holds.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(path.name for path in self.scene_meshes))
        return f"'{self.directory}' holds {len(self.scene_meshes)} meshes: {names}."

    def suggest_correction(self) -> str:
        return (
            "Give the scene a directory of its own, or name its mesh file directly with "
            "WarsawScene.from_file."
        )


@dataclass
class WarsawLabelsMissingError(DataclassException, ValueError):
    """
    Raised when a scene mesh carries no per-face class labels.
    """

    scene_mesh: Path
    """
    The mesh that was read.
    """

    def error_message(self) -> str:
        return f"'{self.scene_mesh}' carries no per-face class labels."

    def suggest_correction(self) -> str:
        return (
            "A scene writes its labels as one integer face property per class. Only the "
            "PLY reader keeps them, under the mesh's '_ply_raw' metadata, so a mesh that "
            "was re-exported or loaded with processing has lost them and has to be "
            "written again."
        )


@dataclass
class WarsawLabelsMisalignedError(DataclassException, ValueError):
    """
    Raised when the faces a world was built from are not the faces the labels were
    written for, which would leave every segment pointing at another object's geometry.
    """

    scene_mesh: Path
    """
    The mesh that was read.
    """

    labelled_faces: int
    """
    How many faces the labels were written for.
    """

    loaded_faces: int
    """
    How many faces the world's scene body ended up with.
    """

    def error_message(self) -> str:
        return (
            f"'{self.scene_mesh}' was labelled for {self.labelled_faces} faces, but the "
            f"world's scene body holds {self.loaded_faces} of them."
        )

    def suggest_correction(self) -> str:
        return (
            "The mesh has to reach the world unprocessed: welding vertices or dropping "
            "degenerate faces renumbers the faces and leaves every label pointing at "
            "another one."
        )


# %% preparing a run


@dataclass
class DatabaseNotConfiguredError(DataclassException, RuntimeError):
    """
    Raised when a step that reads or writes worlds has no database to reach.
    """

    variable: str
    """
    The environment variable the connection is read from.
    """

    def error_message(self) -> str:
        return f"'{self.variable}' is not set, so there is no database to reach."

    def suggest_correction(self) -> str:
        return (
            f"Export '{self.variable}' with a 'postgresql+psycopg://' URI before running "
            "the pipeline, or run it with persistence turned off, which stops it at the "
            "split's report."
        )


@dataclass
class OntologyLeftAmendedError(DataclassException, RuntimeError):
    """
    Raised when a run would start against an ontology an earlier run left edited.

    A run must start from what is committed. Classes generated for one scene and mixins
    one room argued for would otherwise be in force for the next, where nothing
    questions them and nobody remembers they were ever in doubt.
    """

    amended_paths: List[str]
    """
    The ontology's own files that differ from what is committed.
    """

    def error_message(self) -> str:
        return "The ontology's own files are left amended: " + ", ".join(
            sorted(self.amended_paths)
        )

    def suggest_correction(self) -> str:
        return (
            "Put them back by reverting the run that applied them, or start the pipeline "
            "with ignore_amendments set, which runs against them deliberately."
        )


@dataclass
class SubprocessStepFailedError(DataclassException, RuntimeError):
    """
    Raised when work a step handed to a new interpreter did not finish.

    Some work has to happen in an interpreter that started after the ontology or the ORM
    was rewritten, because the one asking is holding the version from before that.
    """

    what: str
    """
    What the interpreter was asked to do.
    """

    output: str
    """
    What it said before it stopped.
    """

    shown: int = 2000
    """
    How much of that output the message quotes, counting back from the end.
    """

    def error_message(self) -> str:
        return f"{self.what} failed:\n{self.output.strip()[-self.shown :]}"

    def suggest_correction(self) -> str:
        return (
            "Read the output above: it is the failure as the interpreter reported it, "
            "not a summary of it."
        )


@dataclass
class AmendmentNotInForceError(DataclassException, RuntimeError):
    """
    Raised when an amended class, read anew, does not derive from the mixin it was
    amended with.

    A field the ORM does not know about cannot be written to the database, so an
    amendment that is not in force reads as done and is not.
    """

    pairs: List[AmendedPair]
    """
    The amendments that did not take effect.
    """

    def error_message(self) -> str:
        missing = ", ".join(f"{pair.whole}({pair.mixin})" for pair in self.pairs)
        return f"These amendments were written but are not in force: {missing}."

    def suggest_correction(self) -> str:
        return (
            "Look at the amended lines in the ontology's source: the edit has to make the "
            "class derive from the mixin when the module is imported anew."
        )


@dataclass
class RunOutputAlreadyWrittenError(DataclassException, FileExistsError):
    """
    Raised when a step would write over output that is already there.

    Writing beside it is how a run comes to be part one run and part another: the files a
    step does not happen to rewrite stay as an earlier pass left them, and nothing says so.
    """

    directory: Path
    """
    The run the step meant to write into.
    """

    written: List[str]
    """
    The files that are already there.
    """

    def error_message(self) -> str:
        return f"'{self.directory}' already holds " + ", ".join(sorted(self.written))

    def suggest_correction(self) -> str:
        return (
            "Write into a run of its own, or let the step overwrite, which is what a step "
            "run twice in one run needs -- the second time knowing something the first "
            "did not."
        )


# %% the classes a run generates


@dataclass
class GeneratedClassesAlreadyImportedError(DataclassException, ImportError):
    """
    Raised when a run's generated classes cannot be the ones in use.

    By the time the module has been imported from somewhere else, everything holding one
    of those classes holds the wrong one, so pointing at the run's file now would leave
    two versions of the same class in one interpreter.
    """

    module_name: str
    """
    The name the classes are imported under.
    """

    imported_from: str
    """
    Where the module standing under that name was read from.
    """

    wanted: Path
    """
    The run's own file, which was to have been used instead.
    """

    def error_message(self) -> str:
        return (
            f"'{self.module_name}' was already imported from '{self.imported_from}', so "
            f"'{self.wanted}' cannot be the file in use."
        )

    def suggest_correction(self) -> str:
        return (
            "Point at the run's classes before anything imports the ORM or the "
            "annotations, which is why it is done at the top of a step."
        )


# %% reading a world back


@dataclass
class NoWorldRecordedError(DataclassException, LookupError):
    """
    Raised when a step needs the world an earlier step wrote and none was recorded.
    """

    step: str
    """
    The step whose record was read.
    """

    def error_message(self) -> str:
        return f"The {self.step} recorded no world in the database."

    def suggest_correction(self) -> str:
        return (
            "Run the pipeline with persistence turned on: everything after the split "
            "reads a world back, so without it there is nothing to read."
        )


@dataclass
class WorldNotInDatabaseError(DataclassException, LookupError):
    """
    Raised when the world a step was told to read is not in the database it is looking
    in.
    """

    world_id: int
    """
    The id that was looked for.
    """

    def error_message(self) -> str:
        return f"No world in the database with id {self.world_id}."

    def suggest_correction(self) -> str:
        return (
            "Check that the connection points at the schema the run wrote into: a run "
            "writes into a schema of its own, and its worlds are invisible from another."
        )


# %% finding what to render


@dataclass
class SceneBodyNotFoundError(DataclassException, LookupError):
    """
    Raised when no body in the world carries the name the scene was loaded under.
    """

    scene_body_name: str
    """
    The name a body was looked for under.
    """

    body_names: List[str]
    """
    The names the world's bodies with collision carry instead.
    """

    def error_message(self) -> str:
        carried = ", ".join(sorted(self.body_names)) or "none"
        return (
            f"No body named '{self.scene_body_name}' carries collision geometry; "
            f"the world carries: {carried}."
        )

    def suggest_correction(self) -> str:
        return (
            "Name the body the scene was loaded as in scene_body_name, or load the "
            "scene through the loader so it is named the way the loader names it."
        )


@dataclass
class NoSegmentsGivenError(DataclassException, ValueError):
    """
    Raised when a render is asked for over no segments at all.
    """

    def error_message(self) -> str:
        return "No segments were given to render."

    def suggest_correction(self) -> str:
        return (
            "Pass the segments to be rendered. An empty selection is a question about "
            "nothing, which is decided by the caller rather than here."
        )


@dataclass
class CameraHasNoDirectionError(DataclassException, ValueError):
    """
    Raised when a camera's placement leaves the direction it faces undefined.

    A camera standing where it looks has no forward direction, and one looking straight
    along the world's up axis has no sideways one. Either leaves the transform full of
    NaN rather than pointing anywhere.
    """

    eye: List[float]
    """
    Where the camera stands.
    """

    target: List[float]
    """
    What it was to face.
    """

    def error_message(self) -> str:
        return (
            f"A camera at {self.eye} facing {self.target} has no direction to look in."
        )

    def suggest_correction(self) -> str:
        return (
            "Place the camera away from what it looks at, and off the axis running "
            "straight up through it."
        )


@dataclass
class BlankRenderError(DataclassException, RuntimeError):
    """
    Raised when the renderer hands back a picture of one flat color.
    """

    viewpoint: str
    """
    The viewpoint the picture was taken from.
    """

    def error_message(self) -> str:
        return f"The render from '{self.viewpoint}' is a single flat color."

    def suggest_correction(self) -> str:
        return (
            "Many graphics drivers refuse to draw into a hidden window and hand back "
            "blank. If PipelineSettings.headless is True, set it to False on a machine "
            "with a display; on a machine without one, run the pipeline under "
            "'xvfb-run -a'."
        )


# %% keeping what a model was asked


@dataclass
class UnsupportedMessagePartError(DataclassException, TypeError):
    """
    Raised when a message part is neither text nor an image, so a trace has no way to
    record it.
    """

    part_type: type
    """
    The type of the part that was sent.
    """

    def error_message(self) -> str:
        return f"A model trace cannot record a message part of type {self.part_type.__name__}."

    def suggest_correction(self) -> str:
        return (
            "Send text as a TextPart and pictures as an ImagePart, or teach "
            "ModelCallPart.of to record the new kind of part."
        )


# %% rebuilding a run's ORM


@dataclass
class RunClassTakenOverByTheOntologyError(DataclassException, ValueError):
    """
    Raised when the ontology has gained a class an earlier run had to generate.

    Two classes of one name are two tables of one name, and an ORM holding both cannot
    be imported at all -- so rebuilding for that run would leave every other run unable
    to read anything. The run's world stays where it is; it is only no longer reachable
    through an ORM that now means something else by the name.
    """

    directory: str
    """
    The run whose classes were read.
    """

    class_names: List[str]
    """
    The ones the ontology has since gained.
    """

    def error_message(self) -> str:
        return (
            f"The ontology now has {', '.join(self.class_names)}, which the run in "
            f"'{self.directory}' generated for itself."
        )

    def suggest_correction(self) -> str:
        return (
            "Read that run through the files it wrote rather than through the database: "
            "evaluation_graph.json holds its bodies and relations and needs no ORM. A "
            "run made since the ontology gained the class does not generate it and opens "
            "normally."
        )


@dataclass
class HabitatAnnotationsUnreadableError(DataclassException, ValueError):
    """
    Raised when a file is not one of the tables an HM3D scene annotates itself with.
    """

    table: Path
    """
    The file that was read.
    """

    complaint: str
    """
    What is wrong with it.
    """

    row: str
    """
    The line the complaint is about.
    """

    def error_message(self) -> str:
        return f"'{self.table}' is not an HM3D annotation table: {self.complaint}."

    def suggest_correction(self) -> str:
        return (
            "The table is the '.semantic.txt' beside a scene's semantic mesh. It opens "
            "with a line naming itself and then writes one object per line as "
            "'object_id,hex_color,\"label\",room_id'. The line read was: "
            f"'{self.row}'."
        )


@dataclass
class HabitatSceneNotFoundError(DataclassException, FileNotFoundError):
    """
    Raised when an HM3D release does not hold all three files a scene is written as.
    """

    root: Path
    """
    The release that was searched.
    """

    scene_name: str
    """
    The scene that was asked for.
    """

    missing: List[Path]
    """
    The files it does not hold.
    """

    holds: List[str]
    """
    The scenes it does annotate.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(str(path) for path in self.missing))
        return f"'{self.scene_name}' is not whole in '{self.root}': {names} missing."

    def suggest_correction(self) -> str:
        return (
            "A scene is three files under two trees of a release: the annotation table "
            "and the mesh painted one colour per object under the semantic annotations, "
            "and the mesh painted in the building's own colours under the GLB tree. "
            f"This release annotates {', '.join(self.holds) or 'no scene at all'}."
        )


@dataclass
class HabitatMeshesDisagreeError(DataclassException, ValueError):
    """
    Raised when a scene's two meshes are not written over the same triangles.
    """

    semantic_mesh: Path
    """
    The mesh painted one flat colour per object.
    """

    textured_mesh: Path
    """
    The mesh painted in the colours the building was photographed in.
    """

    complaint: str
    """
    How they disagree.
    """

    def error_message(self) -> str:
        return (
            f"'{self.semantic_mesh.name}' and '{self.textured_mesh.name}' are not the "
            f"same triangles: {self.complaint}."
        )

    def suggest_correction(self) -> str:
        return (
            "The colours the room was seen in are read off the textured mesh by face, "
            "which only names the same face while the two files are the same triangles. "
            "Check that both were taken from the same release rather than mixed between "
            "versions."
        )
