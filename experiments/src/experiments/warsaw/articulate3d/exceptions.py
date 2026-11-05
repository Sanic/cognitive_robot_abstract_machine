from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% an incomplete or inconsistent protocol


@dataclass
class PartNameNotRuledError(DataclassException, LookupError):
    """
    Raised when a scene holds a part whose name the protocol says nothing about.
    """

    part_name: str
    """
    The name the annotator gave the part.
    """

    def error_message(self) -> str:
        return f"The protocol has no ruling for the part name '{self.part_name}'."

    def suggest_correction(self) -> str:
        return (
            "List the dataset's part names with "
            "'python -m experiments.warsaw.articulate3d.labels' and add a ruling for "
            "this one, before any run is judged."
        )


@dataclass
class PartNameUndecidedError(DataclassException, ValueError):
    """
    Raised when the protocol lists a part name but has not decided what it means yet.
    """

    part_name: str
    """
    The name the annotator gave the part.
    """

    def error_message(self) -> str:
        return f"The protocol leaves the part name '{self.part_name}' undecided."

    def suggest_correction(self) -> str:
        return (
            "Decide the name's ruling in the protocol: a class of the ontology, "
            "ambiguous, or ignored."
        )


@dataclass
class ClassNotInOntologyError(DataclassException, LookupError):
    """
    Raised when the protocol names a class the ontology does not hold.
    """

    class_name: str
    """
    The class the protocol names.
    """

    def error_message(self) -> str:
        return (
            f"The protocol names the class '{self.class_name}', which the ontology "
            f"does not hold."
        )

    def suggest_correction(self) -> str:
        return (
            "Correct the protocol, or extend the ontology before the protocol is "
            "frozen."
        )


@dataclass
class AmbiguousPartFieldError(DataclassException, LookupError):
    """
    Raised when a whole could hold a part through several fields and the protocol does
    not say which.
    """

    whole_class: str
    """
    The class of the whole.
    """

    part_class: str
    """
    The class of the part.
    """

    fields: list[str]
    """
    The fields that would accept the part.
    """

    def error_message(self) -> str:
        return (
            f"A {self.whole_class} could hold a {self.part_class} through any of "
            f"{self.fields}, and the protocol does not choose."
        )

    def suggest_correction(self) -> str:
        return "Add a field choice for this pair of classes to the protocol."


@dataclass
class AmbiguousWholeError(DataclassException, ValueError):
    """
    Raised when a part lies below a place in the part hierarchy that several parts
    share, so which of them it belongs to is not annotated.
    """

    scene_id: str
    """
    The scene the part is annotated in.
    """

    part_label: str
    """
    The part's label.
    """

    whole_labels: list[str]
    """
    The labels of the parts sharing the place above it.
    """

    def error_message(self) -> str:
        return (
            f"'{self.part_label}' in scene {self.scene_id} could belong to any of "
            f"{self.whole_labels}."
        )

    def suggest_correction(self) -> str:
        return (
            "Leave the scene out with the protocol's AmbiguousWhole rule, or ignore all "
            "but one of the parts sharing the place."
        )


# %% a run that cannot be scored


@dataclass
class SceneNotConvertedError(DataclassException, FileNotFoundError):
    """
    Raised when a run's scene directory holds no record of the Articulate3D scene it was
    converted from.
    """

    scene_directory: Path
    """
    The directory the run read its scene from.
    """

    def error_message(self) -> str:
        return (
            f"'{self.scene_directory}' was not converted from Articulate3D, so there is "
            f"no ground truth to score it against."
        )

    def suggest_correction(self) -> str:
        return (
            "Score only runs of scenes written by "
            "'python -m experiments.warsaw.articulate3d.convert'."
        )


@dataclass
class ConvertedLabelsCollideError(DataclassException, ValueError):
    """
    Raised when two different part names would be written as the same face property.
    """

    part_names: list[str]
    """
    The part names that would collide.
    """

    property_name: str
    """
    The face property they would share.
    """

    def error_message(self) -> str:
        return (
            f"The part names {self.part_names} would all be written as the face "
            f"property '{self.property_name}'."
        )

    def suggest_correction(self) -> str:
        return "Convert the scene with neutral labels, which never collide."


# %% a scene the rules leave out


@dataclass
class SceneExcludedError(DataclassException, ValueError):
    """
    Raised when a scene the protocol's rules leave out is to be evaluated.
    """

    scene_id: str
    """
    The scene.
    """

    findings: list[str]
    """
    What makes each rule leave it out.
    """

    def error_message(self) -> str:
        return f"The protocol leaves scene {self.scene_id} out: {self.findings}"

    def suggest_correction(self) -> str:
        return "Evaluate only scenes the protocol keeps."


# %% a validation scene before the freeze


@dataclass
class ValidationSceneLockedError(DataclassException, PermissionError):
    """
    Raised when a validation scene is to be converted, run or scored from a checkout
    that is not frozen.
    """

    scene_id: str
    """
    The validation scene.
    """

    commit: str | None
    """
    The commit checked out, or nothing where the checkout is no Git work tree.
    """

    local_changes: list[str]
    """
    The changes the checkout holds beyond its commit, as Git reports them.
    """

    tags: list[str]
    """
    The tags on the commit checked out.
    """

    tag_prefix: str
    """
    How the name of a tag freezing the code starts.
    """

    def error_message(self) -> str:
        return (
            f"Scene {self.scene_id} is a validation scene, and the checkout at commit "
            f"{self.commit} is not frozen: tags {self.tags}, "
            f"{len(self.local_changes)} local changes."
        )

    def suggest_correction(self) -> str:
        return (
            "Tune on train scenes only. Validation scenes open once the code, the "
            "ontology and the protocol are frozen: commit everything, tag the commit "
            f"'{self.tag_prefix}<name>', and run from that commit with no local changes."
        )
