"""
The ``RULES`` registry auto-discovers every concrete grammar rule by walking the grammar package,
rather than hand-maintaining a list of imports — so adding a construct's ``rules`` module needs no
edit to the registry. ``concrete_subclasses`` is the single subclass-discovery primitive shared by
the registry and the :class:`SpecificityRule` families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
)
from krrood.entity_query_language.verbalization.grammar.framework.registry import RULES
from krrood.entity_query_language.verbalization.grammar.framework.specificity import (
    concrete_subclasses,
)


def test_concrete_subclasses_excludes_abstract_bases():
    class Base(ABC):
        @abstractmethod
        def do(self): ...

    class Middle(Base, ABC):  # still abstract — does not implement ``do``
        pass

    class Leaf(Base):
        def do(self):
            return 1

    found = concrete_subclasses(Base)
    assert Leaf in found
    assert Middle not in found and Base not in found


def test_rules_registry_is_exactly_one_instance_per_concrete_rule():
    """The registry holds exactly one instance of every concrete ``PhraseRule`` subclass — proving
    auto-discovery is complete and free of duplicates."""
    discovered = {type(rule) for rule in RULES}
    assert discovered == set(concrete_subclasses(PhraseRule))
    assert len(RULES) == len(discovered)  # no duplicate instances


def test_every_construct_folder_contributes_rules():
    """A representative rule from each construct package is auto-discovered (the walk reaches every
    ``rules`` module, not a hand-listed subset)."""
    modules = {type(rule).__module__.rsplit(".", 2)[-2] for rule in RULES}
    for construct_folder in (
        "terms",
        "chain",
        "conditions",
        "query",
        "inference",
        "aggregation",
        "clauses",
        "instantiated",
    ):
        assert (
            construct_folder in modules
        ), f"no rules discovered for grammar/{construct_folder}"
