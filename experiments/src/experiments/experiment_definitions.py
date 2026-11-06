from __future__ import annotations

import enum
import json
import pathlib
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum

from typing_extensions import Any

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.exceptions import DataclassException
from krrood.bounds import Bounds
from krrood.ormatic.utils import classproperty


from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    AttributeIntrospector,
    DiscoveredAttribute,
)


class Unit(StrEnum):
    """
    The physical unit a :class:`MeanAndStandardDeviation` is expressed in.
    """

    NONE = ""
    SECONDS = "s"
    MILLISECONDS = "ms"


_SECONDS_PER_UNIT: dict[Unit, float] = {
    Unit.SECONDS: 1.0,
    Unit.MILLISECONDS: 0.001,
}


@dataclass
class IncompatibleUnitConversionError(DataclassException):
    """
    Raised when a :class:`MeanAndStandardDeviation` is converted to or from a unit that
    has no known conversion factor.
    """

    source_unit: Unit
    """
    The unit the value was expressed in.
    """

    target_unit: Unit
    """
    The unit conversion was requested to.
    """

    def error_message(self) -> str:
        return (
            f"Cannot convert a value from {self.source_unit} to {self.target_unit}: "
            f"no conversion factor is known for one of these units."
        )

    def suggest_correction(self) -> str:
        return "Only convert between units that are both listed in _SECONDS_PER_UNIT."


@dataclass
class MeanAndStandardDeviation:
    """
    Class that represents a mean and standard deviation for tables that will be directly
    rendered as mean +- standard deviation.

    Use this in experiment results when you want to render a mean +- standard deviation.
    """

    mean: float
    """
    The mean of the measurements.
    """

    standard_deviation: float
    """
    The standard deviation of the measurements.
    """

    unit: Unit = Unit.NONE
    """
    The physical unit :attr:`mean` and :attr:`standard_deviation` are expressed in.
    """

    def __str__(self) -> str:
        suffix = f" {self.unit.value}" if self.unit is not Unit.NONE else ""
        return f"{round(self.mean, 2)} ± {round(self.standard_deviation, 2)}{suffix}"

    @classmethod
    def from_measurements(
        cls, measurements: list[float], unit: Unit = Unit.NONE
    ) -> MeanAndStandardDeviation:
        std = 0.0 if len(measurements) < 2 else statistics.stdev(measurements)
        return cls(
            mean=round(statistics.mean(measurements), 2),
            standard_deviation=round(std, 4),
            unit=unit,
        )

    def to(self, unit: Unit) -> MeanAndStandardDeviation:
        """
        Convert this value to another unit.

        :param unit: The unit to convert to.
        :raises IncompatibleUnitConversionError: If either the current or the requested
            unit has no known conversion factor.
        """
        if self.unit not in _SECONDS_PER_UNIT or unit not in _SECONDS_PER_UNIT:
            raise IncompatibleUnitConversionError(
                source_unit=self.unit, target_unit=unit
            )
        factor = _SECONDS_PER_UNIT[self.unit] / _SECONDS_PER_UNIT[unit]
        return MeanAndStandardDeviation(
            mean=round(self.mean * factor, 2),
            standard_deviation=round(self.standard_deviation * factor, 4),
            unit=unit,
        )


@dataclass
class VolumeBound(Bounds[float]):
    """
    A :class:`Bounds` interval known to contain some volume, for tables that report a
    bounded estimate instead of a single point value.

    Use this in experiment results whenever the true volume cannot be measured exactly
    (e.g. it was estimated by sampling) but valid lower and/or upper bounds can be.
    """

    def __str__(self) -> str:
        return f"[{round(self.lower, 4)}, {round(self.upper, 4)}]"


@dataclass
class PercentageBound(Bounds[float]):
    """
    A :class:`Bounds` interval, in percent, bounding the ratio of two
    :class:`VolumeBound` quantities.

    Use this to express one bounded volume as a share of another (e.g. how much of the
    true free volume a covering is known to reach) without collapsing either bound into
    a single, falsely precise point estimate.
    """

    def __str__(self) -> str:
        return f"[{round(self.lower, 2)}%, {round(self.upper, 2)}%]"

    @classmethod
    def ratio_of(
        cls, numerator: VolumeBound, denominator: VolumeBound
    ) -> PercentageBound:
        """
        :param numerator: Bound on the covered quantity.
        :param denominator: Bound on the reference quantity.
        :return: A percentage bound on ``numerator / denominator``, clipped to
            ``[0, 100]``. The worst case for each end is used: the lower end pairs the
            smallest numerator with the largest denominator, and vice versa for the
            upper end.
        """
        raw_lower = 100.0 * numerator.lower / denominator.upper
        raw_upper = 100.0 * numerator.upper / denominator.lower
        lower = max(0.0, min(raw_lower, 100.0))
        upper = max(lower, min(raw_upper, 100.0))
        return cls(lower=lower, upper=upper)


@dataclass
class ExperimentResult:
    """
    Class for results from experiments.

    Use this when you want to create a table of results (measurements) from experiments
    for scientific articles.

    This class is like a single row in a table of experiments.

    Assumptions made here are that there are only built in like fields or one-to-one
    relationships with other ExperimentResult classes.
    """

    @classmethod
    def introspector(cls) -> AttributeIntrospector:
        return DataclassOnlyIntrospector()

    @classmethod
    def recursive_fields(cls) -> list[DiscoveredAttribute]:
        """
        :return: One attribute per column of the row, with the attributes of a result held
            by this one taking the place of the field holding it.

        ..note:: A field annotated with anything other than a class, such as a measurement
            that may be absent, reports a value of its own and is therefore a column.
        """
        result = []
        type_hints = get_type_hints_of_object(cls)
        for field_ in cls.introspector().discover(cls):
            resolved_type = type_hints[field_.field.name]
            if isinstance(resolved_type, type) and issubclass(
                resolved_type, ExperimentResult
            ):
                result.extend(resolved_type.recursive_fields())
            else:
                result.append(field_)
        return result

    @classmethod
    def get_column_names(cls) -> list[str]:
        return [field_.field.name for field_ in cls.recursive_fields()]

    def get_column_values(self) -> list[Any]:
        """
        :return: One value per column of the row, with a result held by this one
            contributing its own values in place of the field holding it.

        ..note:: The values are recursed the same way :meth:`recursive_fields` recurses the
            columns, since a nested result's values live on the nested result and not on the
            row holding it.
        """
        values: list[Any] = []
        type_hints = get_type_hints_of_object(type(self))
        for field_ in self.introspector().discover(type(self)):
            value = getattr(self, field_.field.name)
            resolved_type = type_hints[field_.field.name]
            if isinstance(resolved_type, type) and issubclass(
                resolved_type, ExperimentResult
            ):
                values.extend(value.get_column_values())
            else:
                values.append(value)
        return values


class RowsOfDifferingTypes(Exception):
    """
    Raised when a table is built over rows of more than one type, since a table's
    columns come from its row type and rows of different types have no common set of
    columns.
    """


class RowIsNotAnExperimentResult(Exception):
    """
    Raised when a table is built over rows that are not experiment results, which have
    no columns to present.
    """


@dataclass
class ExperimentsTable:
    """
    A collection of experiments ready to be presented as a table in a scientific
    article.

    This class assumes that all rows in the table have the same type and are a subclass
    of ExperimentResult.
    """

    experiments: list[ExperimentResult]
    """
    The list of experiments to be presented in the table.
    """

    def __post_init__(self):
        if not self.experiments:
            return
        row_types = {type(row) for row in self.experiments}
        if len(row_types) > 1:
            raise RowsOfDifferingTypes(
                "a table presents one kind of row, but was given "
                f"{', '.join(sorted(each.__name__ for each in row_types))}"
            )
        [row_type] = row_types
        if not issubclass(row_type, ExperimentResult):
            raise RowIsNotAnExperimentResult(
                f"{row_type.__name__} is not an experiment result, so it has no columns"
            )

    @property
    def row_class(self) -> type[ExperimentResult] | None:
        if not self.experiments:
            return None
        return type(self.experiments[0])

    def write_manifest(
        self, output_directory: pathlib.Path, file_name: str
    ) -> pathlib.Path:
        """
        Records the rows as JSON alongside the table presenting them.

        :param output_directory: Directory the manifest is written to.
        :param file_name: Name of the manifest file.
        :return: Path the manifest was written to.
        """
        manifest_path = output_directory / file_name
        manifest_path.write_text(json.dumps(self.as_json_rows(), indent=2))
        return manifest_path

    def as_json_rows(self) -> list[dict[str, Any]]:
        """
        :return: One entry per row, holding the same columns the table presents, so that a
            result reported within another is recorded flat rather than nested.
        """
        return [
            {
                field_.field.name: self.as_json_value(value)
                for field_, value in zip(
                    row.recursive_fields(), row.get_column_values()
                )
            }
            for row in self.experiments
        ]

    @classmethod
    def as_json_value(cls, value: Any) -> Any:
        """
        :param value: A value that may contain enum members or nested results, neither of
            which :mod:`json` can serialise on its own.
        :return: The value with every enum member replaced by its name and every nested
            result replaced by its fields.
        """
        if isinstance(value, enum.Enum):
            return value.name
        if isinstance(value, dict):
            return {key: cls.as_json_value(entry) for key, entry in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.as_json_value(entry) for entry in value]
        if is_dataclass(value) and not isinstance(value, type):
            return cls.as_json_value(asdict(value))
        return value


# %% presenting a table in a document


@dataclass
class TableRenderer(ABC):
    """
    Presents an :class:`ExperimentsTable` in the markup language of a document.
    """

    experiments_table: ExperimentsTable
    """
    The experiments to render.
    """

    reported_decimals: int = 2
    """
    Number of decimals a measured quantity is reported to.
    """

    absent_measurement: str = "--"
    """
    What is written where an experiment established no value.
    """

    left_out_columns: frozenset[str] = frozenset()
    """
    The names of the columns not to present.
    """

    @abstractmethod
    def escaped(self, text: str) -> str:
        """
        :param text: Text taken from a result.
        :return: The markup a cell or a column title holds for that text.
        """

    def render_cell(self, value: Any) -> str:
        """
        Renders a single value as a reader of the table should see it.

        A value the experiment holds as a name is written as that name, a measured
        quantity is written at the precision the table reports, and a measurement the
        experiment did not establish is marked absent rather than written as a number.

        :param value: One value of a row.
        :return: The cell content for it.
        """
        if value is None:
            return self.absent_measurement
        if isinstance(value, enum.Enum):
            return self.escaped(value.name.replace("_", " ").title())
        if isinstance(value, float):
            return f"{value:.{self.reported_decimals}f}"
        return self.escaped(str(value))

    def rendered_cells(self, row: ExperimentResult) -> list[str]:
        """
        :param row: One row of the table.
        :return: The content of each of its cells, in the order of the columns.
        """
        return [
            self.render_cell(value)
            for name, value in zip(row.get_column_names(), row.get_column_values())
            if name not in self.left_out_columns
        ]

    def column_titles(self, row_class: type[ExperimentResult]) -> list[str]:
        """
        :param row_class: The type of the table's rows.
        :return: The title of each column, as a reader of the table should see it.
        """
        return [
            self.escaped(name.replace("_", " ").title())
            for name in row_class.get_column_names()
            if name not in self.left_out_columns
        ]

    def columns_showing_one_value(self) -> dict[str, str]:
        """
        :return: Per column every row shows the same value in, as a reader sees it, that
            value. Two values written alike count as one, however they differ underneath.
        """
        rows = self.experiments_table.experiments
        if not rows:
            return {}
        values = zip(*(row.get_column_values() for row in rows))
        cells = [{self.render_cell(value) for value in column} for column in values]
        return {
            name: next(iter(shown))
            for name, shown in zip(rows[0].get_column_names(), cells)
            if len(shown) == 1
        }

    @abstractmethod
    def render_row(self, row: ExperimentResult) -> str:
        """
        :param row: One row of the table.
        :return: The markup of that row.
        """

    @abstractmethod
    def render_table(self) -> str:
        """
        :return: The markup of the entire table, which stays valid for a table without rows.
        """

    @abstractmethod
    def render_figure(self, caption: str) -> str:
        """
        Renders the table with a caption.

        Use this instead of :meth:`render_table` whenever the table is presented to a
        reader, so every table in a document explains what it shows.

        :param caption: Caption text describing what the table shows, in the document's
            markup language.
        :return: The markup of the captioned table.
        """


@dataclass
class TypstRenderer(TableRenderer):
    """
    Renders an :class:`ExperimentsTable` into Typst markup.
    """

    def escaped(self, text: str) -> str:
        return text

    def render_row(self, row: ExperimentResult) -> str:
        return ", ".join(f"[{cell}]" for cell in self.rendered_cells(row))

    def render_table(self) -> str:
        row_class = self.experiments_table.row_class
        if not row_class:
            return "#table()"

        titles = self.column_titles(row_class)
        lines = [", ".join(f"[*{title}*]" for title in titles)]
        lines += [self.render_row(row) for row in self.experiments_table.experiments]
        all_cells = ",\n  ".join(lines)
        return (
            f"#table(\n"
            f"  columns: {len(titles)},\n"
            f"  align: center + horizon,\n"
            f"  {all_cells}\n"
            f")"
        )

    def render_figure(self, caption: str) -> str:
        return f"#figure(\n{self.render_table()},\n  caption: [{caption}]\n)"


@dataclass
class LatexRenderer(TableRenderer):
    """
    Renders an :class:`ExperimentsTable` into LaTeX markup.

    .. note:: The horizontal rules come from the ``booktabs`` package, so a document
        holding these tables has to load it.
    """

    fit_to_line_width: bool = False
    """
    Whether a table wider than a line is shrunk to fit, which needs the ``adjustbox``
    package in the document.

    A narrower table keeps its size.
    """

    @classproperty
    def markup_of_characters(cls) -> dict[int, str]:
        """
        :return: Per character LaTeX would not print as it is, the markup that prints it,
            in the form :meth:`str.translate` reads. Brackets are protected because LaTeX
            reads a ``[`` after the line break ending a row as a part of that break.
        """
        return str.maketrans(
            {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
                "[": "{[}",
                "]": "{]}",
                "±": r"$\pm$",
            }
        )

    def escaped(self, text: str) -> str:
        return text.translate(self.markup_of_characters)

    def render_cell(self, value: Any) -> str:
        """
        :meth:`TableRenderer.render_cell`, with a mean and its spread written at the
        table's precision, so that the numbers of a column line up.

        :param value: One value of a row.
        :return: The cell content for it.
        """
        if not isinstance(value, MeanAndStandardDeviation):
            return super().render_cell(value)
        unit = f" {value.unit.value}" if value.unit is not Unit.NONE else ""
        decimals = self.reported_decimals
        return (
            f"{value.mean:.{decimals}f} $\\pm$ "
            f"{value.standard_deviation:.{decimals}f}{unit}"
        )

    def render_row(self, row: ExperimentResult) -> str:
        return " & ".join(self.rendered_cells(row)) + r" \\"

    def render_table(self) -> str:
        row_class = self.experiments_table.row_class
        if not row_class:
            return "\\begin{tabular}{c}\n\\end{tabular}"

        titles = self.column_titles(row_class)
        title_row = " & ".join(rf"\textbf{{{title}}}" for title in titles) + r" \\"
        table = "\n".join(
            [
                rf"\begin{{tabular}}{{{'c' * len(titles)}}}",
                r"\toprule",
                title_row,
                r"\midrule",
                *[self.render_row(row) for row in self.experiments_table.experiments],
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        if not self.fit_to_line_width:
            return table
        return f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n{table}\n\\end{{adjustbox}}"

    def render_figure(self, caption: str, label: str | None = None) -> str:
        """
        :meth:`TableRenderer.render_figure`, in a floating ``table`` environment.

        :param caption: Caption text describing what the table shows, in LaTeX.
        :param label: What the document refers to the table by with ``\\ref``.
        :return: The markup of the captioned table.
        """
        return "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                rf"\caption{{{caption}}}",
                *([rf"\label{{{label}}}"] if label else []),
                self.render_table(),
                r"\end{table}",
            ]
        )
