from __future__ import annotations

from enum import StrEnum

from typing_extensions import Optional


class SemanticRole(StrEnum):
    """Semantic category of a fragment, determining its colour markup."""

    KEYWORD = "keyword"
    """EQL structure words — *If*, *Then*, *Find*, *Where*, *Such that*."""
    VARIABLE = "variable"
    """Type and instance names — *Robot*, *Employee 1*."""
    AGGREGATION = "aggregation"
    """Aggregation phrases — *sum of*, *number of*, *average of*."""
    OPERATOR = "operator"
    """Comparator phrases — *is greater than*, *equals*."""
    LOGICAL = "logical"
    """Logical connectives — *and*, *or*, *not*, *for all*, *there exists*."""
    LITERAL = "literal"
    """Literal values — ``42``, ``"hello"``, ``True``."""
    ATTRIBUTE = "attribute"
    """Attribute and field names — *battery*, *tasks*, *name*."""
    PLAIN = "plain"
    """Neutral connecting text with no special colour."""


#: Hex colour string (or ``None`` for no colour) for each semantic role, matching the
#: query-graph palette.
ROLE_COLORS: dict[SemanticRole, Optional[str]] = {
    SemanticRole.KEYWORD: "#eded18",  # ConclusionSelector yellow
    SemanticRole.VARIABLE: "cornflowerblue",
    SemanticRole.AGGREGATION: "#F54927",  # Aggregator red-orange
    SemanticRole.OPERATOR: "#ff7f0e",  # Comparator orange
    SemanticRole.LOGICAL: "#2ca02c",  # LogicalOperator green
    SemanticRole.LITERAL: "#949292",  # Literal gray
    SemanticRole.ATTRIBUTE: "#8FC7B8",  # MappedVariable teal
    SemanticRole.PLAIN: None,
}
