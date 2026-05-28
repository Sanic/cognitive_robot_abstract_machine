"""
Semantic roles for fragment colour markup and the EQL-type-to-role mapping.

:class:`SemanticRole` tags a
:class:`~krrood.entity_query_language.verbalization.fragments.base.RoleFragment`
with a semantic category; :data:`ROLE_COLORS` maps each role to a hex colour.
The colours match the ``QueryGraph.ColorLegend`` palette so that verbalization
output is visually consistent with query graph visualizations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from typing_extensions import Type

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import LogicalOperator
from krrood.entity_query_language.query.query import Entity, SetOf


class SemanticRole(Enum):
    """
    Semantic role attached to a
    :class:`~krrood.entity_query_language.verbalization.fragments.base.RoleFragment`.

    The role determines the colour applied by the active
    :class:`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter`
    (see :data:`ROLE_COLORS`).

    :cvar KEYWORD: EQL structure words — *If*, *Then*, *Find*, *Where*, *Such that*.
    :cvar VARIABLE: Type and instance names — *Robot*, *Employee 1*.
    :cvar AGGREGATION: Aggregation phrases — *sum of*, *number of*, *average of*.
    :cvar OPERATOR: Comparator phrases — *is greater than*, *equals*.
    :cvar LOGICAL: Logical connectives — *and*, *or*, *not*, *for all*, *there exists*.
    :cvar LITERAL: Literal values — ``42``, ``"hello"``, ``True``.
    :cvar ATTRIBUTE: Attribute and field names — *battery*, *tasks*, *name*.
    :cvar PLAIN: Neutral connecting text with no special colour.
    """

    KEYWORD = "keyword"
    VARIABLE = "variable"
    AGGREGATION = "aggregation"
    OPERATOR = "operator"
    LOGICAL = "logical"
    LITERAL = "literal"
    ATTRIBUTE = "attribute"
    PLAIN = "plain"


#: Hex colour strings (or ``None`` for no colour) for each :class:`SemanticRole`.
#:
#: Colours are taken from ``QueryGraph.ColorLegend`` to keep verbalization output
#: visually consistent with query graph visualizations.
#:
#: :type: dict[SemanticRole, str | None]
ROLE_COLORS: dict[SemanticRole, Optional[str]] = {
    SemanticRole.KEYWORD:     "#eded18",        # ConclusionSelector yellow
    SemanticRole.VARIABLE:    "cornflowerblue",
    SemanticRole.AGGREGATION: "#F54927",         # Aggregator red-orange
    SemanticRole.OPERATOR:    "#ff7f0e",         # Comparator orange
    SemanticRole.LOGICAL:     "#2ca02c",         # LogicalOperator green
    SemanticRole.LITERAL:     "#949292",         # Literal gray
    SemanticRole.ATTRIBUTE:   "#8FC7B8",         # MappedVariable teal
    SemanticRole.PLAIN:       None,
}


def _build_role_map() -> Dict[type, SemanticRole]:
    """Return the mapping of EQL expression types to their :class:`SemanticRole`."""
    return {
        LogicalOperator: SemanticRole.LOGICAL,
        Aggregator:      SemanticRole.AGGREGATION,
        Comparator:      SemanticRole.OPERATOR,
        MappedVariable:  SemanticRole.ATTRIBUTE,
        Literal:         SemanticRole.LITERAL,   # before Variable in MRO traversal
        Variable:        SemanticRole.VARIABLE,
        Entity:          SemanticRole.VARIABLE,
        SetOf:           SemanticRole.VARIABLE,
    }


_role_map: Optional[Dict[type, SemanticRole]] = None


def role_for(expression) -> SemanticRole:
    """
    Return the :class:`SemanticRole` for an EQL expression instance using MRO lookup.

    Traverses the MRO of ``type(expression)`` and returns the role of the first ancestor
    found in the role map.  Falls back to :attr:`~SemanticRole.PLAIN` when no
    match is found (e.g. for custom expression types).

    :param expression: Any EQL expression instance.
    :returns: The most-specific matching :class:`SemanticRole`.
    :rtype: SemanticRole
    """
    global _role_map
    if _role_map is None:
        _role_map = _build_role_map()
    for cls in type(expression).__mro__:
        if cls in _role_map:
            return _role_map[cls]
    return SemanticRole.PLAIN
