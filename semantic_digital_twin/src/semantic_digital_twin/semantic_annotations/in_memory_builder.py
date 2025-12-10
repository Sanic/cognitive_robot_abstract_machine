import os.path
from dataclasses import field, MISSING, make_dataclass, dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Iterable, Tuple, Dict, Callable, Union
from jinja2 import Environment, FileSystemLoader
from typing_extensions import Any

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


class SpecialFieldTypes(Enum):
    """
    Special field types for semantic annotation class building.
    """

    NO_DEFAULT = object
    """
    Indicates that a field has no default value. Needed because `None` can be a valid default.
    """


SemanticAnnotationFieldDefaultTypes = Union[SpecialFieldTypes, Any]


class SemanticAnnotationFilePaths(Enum):
    """
    File paths related to semantic annotations for easy access.
    """

    MAIN_SEMANTIC_ANNOTATION_FILE = os.path.join(
        Path(__file__).resolve().parent, "semantic_annotations.py"
    )


@dataclass
class SemanticAnnotationClassBuilder:
    """
    A builder class for creating semantic annotation classes either in memory or as source code files.
    """

    name: str
    """
    The name of the semantic annotation class to create.
    """

    fields: Iterable[Tuple[str, Any, Any]] = field(default_factory=list)
    """
    The fields of the semantic annotation class to create.
    """

    bases: Tuple[type, ...] = field(default_factory=tuple)
    """
    The base classes of the semantic annotation class to create.
    """

    namespace: Dict[str, Any] = field(default_factory=dict)
    """
    The namespace (methods, class variables) of the semantic annotation class to create.
    """

    template_directory: Path = field(
        default=Path(__file__).resolve().parent / "templates"
    )
    template_name: str = field(kw_only=True)
    _env: Environment = field(init=False)

    def __post_init__(self):
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_directory)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def add_field(
        self,
        name: str,
        type_: type,
        default: SemanticAnnotationFieldDefaultTypes = SpecialFieldTypes.NO_DEFAULT,
    ):
        self.fields = list(self.fields) + [(name, type_, default)]
        return self

    def add_method(self, name: str, func: Callable):
        self.namespace[name] = func
        return self

    def add_base(self, base: type):
        self.bases = tuple(self.bases) + (base,)
        return self

    # %% In memory dataclass creation

    def _expand_fields_for_make_dataclass(self):
        """
        Expands the fields to be compatible with make_dataclass, converting fields without defaults
        """
        expanded = []
        for name, type_, default in self.fields:
            if default is SpecialFieldTypes.NO_DEFAULT:
                # required kw-only field (no default)
                expanded.append((name, type_, field(default=MISSING, kw_only=True)))
            else:
                # explicit default (including None)
                expanded.append((name, type_, default))
        return expanded

    def build(self):
        """
        Builds the semantic annotation class in memory as a dataclass.
        :return: The semantic annotation class.
        :raises TypeError: If at least one base class is not a subclass of SemanticAnnotation.
        """
        if not any(issubclass(base, SemanticAnnotation) for base in self.bases):
            raise TypeError(
                f"At least one base class must be a subclass of SemanticAnnotation."
            )
        expanded_fields = self._expand_fields_for_make_dataclass()
        return make_dataclass(
            self.name,
            expanded_fields,
            bases=self.bases,
            namespace=dict(self.namespace),
            eq=False,
        )

    # %% Jinja rendering
    def _fields_for_template(self):
        """
        Prepares the fields for rendering in the Jinja template.
        """
        out = []
        for name, type_, default in self.fields:
            out.append(
                {
                    "name": name,
                    "type_hint": getattr(type_, "__name__", repr(type_)),
                    "default": (
                        None
                        if default is SpecialFieldTypes.NO_DEFAULT
                        else repr(default)
                    ),
                }
            )
        return out

    def render_source(self, include_imports: bool = False) -> str:
        """
        Renders the semantic annotation class as source code using the specified Jinja template.
        """
        template = self._env.get_template(self.template_name)
        rendered = template.render(
            name=self.name,
            bases=[b.__name__ for b in self.bases],
            fields=self._fields_for_template(),
            include_imports=include_imports,
        )
        return rendered.rstrip() + "\n"

    def append_to_file(self, filepath: Path, include_imports: bool = False):
        """
        Appends the rendered semantic annotation class source code to the specified file.
        """
        src = self.render_source(include_imports=include_imports)
        with open(str(filepath), "a", encoding="utf-8") as f:
            f.write("\n\n" + src)
