"""
Keep the classes a run generates inside that run.

A scene needs classes the taxonomy does not have -- a ``Ceiling``, a ``KitchenIsland``.
Written into the ontology's own package, as they were, they are a tracked file that
every run leaves modified, one careless ``git add`` away from becoming part of the
shared ontology, and importable by the next run's taxonomy export.

They belong to the run that proposed them, so they are written into its directory. The
run's directory is then put at the *front* of the annotations package's search path,
which makes its file the one imported under the name the ORM refers to them by, ahead of
the package's own empty one. The ORM generator finds them by walking that path, so it
maps them without being told anything.

This has to be done before anything imports the ORM or the classes, which is why it is a
call at the top of a step rather than something a step can opt into later.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    declared_annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from experiments.warsaw.exceptions import (
    GeneratedClassesAlreadyImportedError,
    RunClassTakenOverByTheOntologyError,
)


@dataclass
class GeneratedClasses:
    """
    The classes one run generated, and the name they are imported under.
    """

    directory: Path
    """
    The run's directory, which is where they are written.
    """

    module_name: str = "semantic_digital_twin.semantic_annotations.generated_classes"
    """
    The name the ORM refers to a generated class by, whatever directory it is written
    in.
    """

    file_name: str = "generated_classes.py"
    """
    What the file is called.
    """

    holding_directory: str = "annotations"
    """
    The directory inside the run that holds the generated classes and nothing else.

    Its own directory rather than the run's, because what is put on the annotations
    package's search path becomes part of that package: with the run's directory there,
    the inspector script a finished run leaves behind is walked by the ORM generator and
    mapped as though it were an annotation class, and a run cannot be annotated twice.
    """

    @property
    def path(self) -> Path:
        """
        :return: Where the run's generated classes are written.
        """
        return self.searched_directory / self.file_name

    @property
    def searched_directory(self) -> Path:
        """
        :return: The directory put on the annotations package's search path.
        """
        return Path(self.directory).resolve() / self.holding_directory

    @property
    def were_generated(self) -> bool:
        """
        :return: Whether the run generated any.
        """
        return self.path.exists()

    @property
    def class_names(self) -> list[str]:
        """
        :return: The names of the classes this run generated, read out of the file rather
            than by importing it, so asking costs nothing and changes nothing.
        """
        return list(self.declared_bases)

    @property
    def declared_bases(self) -> dict[str, list[str]]:
        """
        :return: The names of the classes each class this run generated derives from, in
            declaration order, by class name. Read out of the file rather than by
            importing it, so asking costs nothing and changes nothing.
        """
        if not self.were_generated:
            return {}
        return {
            statement.name: [
                base.id for base in statement.bases if isinstance(base, ast.Name)
            ]
            for statement in ast.parse(self.path.read_text()).body
            if isinstance(statement, ast.ClassDef)
        }

    def taken_over_by_the_ontology(self) -> list[str]:
        """
        Say which of this run's own classes the ontology has since gained.

        Read from the classes the ontology *declares*, not from every class below the
        annotation root: composing a proposed class registers it there for the rest of the
        process, and the steps of a run share one, so the live root would report a class
        this very run invented a moment ago and refuse the run its own classes.

        :return: Those names, empty where its classes are still its own.
        """
        declared = declared_annotation_classes(SemanticAnnotation)
        return sorted(name for name in self.class_names if name in declared)

    def refuse_classes_taken_over(self) -> None:
        """
        Refuse to map this run's classes once the ontology has gained one of them.

        :raises RunClassTakenOverByTheOntologyError: If the ontology has since gained a
            class this run generated, which would leave two classes of one name and an
            ORM nothing can import.
        """
        taken_over = self.taken_over_by_the_ontology()
        if taken_over:
            raise RunClassTakenOverByTheOntologyError(
                directory=str(self.directory), class_names=taken_over
            )

    def use(self) -> ModuleType | None:
        """
        Make this run's generated classes the ones this interpreter means.

        :return: The module, once something imports it, or None if the run generated
            none.
        :raises GeneratedClassesAlreadyImportedError: If the classes were already
            imported from somewhere else, since by then everything holding one of them
            holds the wrong one.
        """
        if not self.were_generated:
            return None

        already = sys.modules.get(self.module_name)
        if already is not None:
            imported_from = Path(already.__file__ or "").resolve()
            if imported_from == self.path:
                return already
            raise GeneratedClassesAlreadyImportedError(
                module_name=self.module_name,
                imported_from=str(imported_from),
                wanted=self.path,
            )

        import semantic_digital_twin.semantic_annotations as annotations

        # In front of the package's own file, so a walk of the package finds this one
        # first.
        directory = str(self.searched_directory)
        if directory not in annotations.__path__:
            annotations.__path__.insert(0, directory)
        return sys.modules.get(self.module_name)
