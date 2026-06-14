"""
Doctest harness for the verbalization grammar rules and forms.

Every dispatch rule / form docstring carries a concrete ``>>> verbalize_expression(...)`` example
next to its template form, so the documented output is *executed* and cannot silently drift from
what the grammar actually produces.

The examples are kept to a single readable line by injecting a shared namespace — the EQL
factories, ``Not``, ``verbalize_expression`` and the example-domain classes — so a docstring need
only write ``verbalize_expression(variable(Task, []).completed)`` rather than re-import everything.
The example domain is the same module Sphinx AutoAPI documents, so the rendered examples also
hyperlink to real API pages.
"""

from __future__ import annotations

import doctest

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.verbalization import example_domain
from krrood.entity_query_language.verbalization.grammar import english, restriction
from krrood.entity_query_language.verbalization.grammar.assembly import (
    aggregation_value,
    chains,
    clauses as assembly_clauses,
    query as assembly_query,
)
from krrood.entity_query_language.verbalization.grammar.planning import (
    chains as planning_chains,
    clauses as planning_clauses,
    query as planning_query,
)
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

# The grammar modules whose rule/form/assembler/planner docstrings carry executable examples.
# Assemblers show the rendered string (verbalize_expression); planners show the plan decision
# they compute (a data record, not a rendered string).
_MODULES = [
    english,
    restriction,
    chains,
    assembly_query,
    assembly_clauses,
    aggregation_value,
    planning_chains,
    planning_clauses,
    planning_query,
]

# A shared namespace for all examples. It includes the EQL factories, so an example can write
# *"verbalize_expression(variable(Task, []).completed)"* rather than re-import everything.
# It also includes the example-domain classes, so the rendered examples also hyperlink to real API pages.
factories = [
    eql.variable, # variables
    eql.a, eql.an, eql.the, # quantifiers
    eql.entity, eql.set_of, # query construction
    eql.and_, eql.or_, # boolean logic
    eql.max, eql.min, eql.sum, eql.count, # aggregations
    eql.contains, eql.in_, # membership
    eql.for_all, eql.exists # quantified conditionals
]
_GLOBS = {factory.__name__: factory for factory in factories}
_GLOBS.update(
    verbalize_expression=verbalize_expression,
    Not=Not,
    Exists=Exists,
    ForAll=ForAll,
)
# The example-domain classes (defined in that module, not imported into it).
_GLOBS.update(
    {
        name: obj
        for name, obj in vars(example_domain).items()
        if isinstance(obj, type) and obj.__module__ == example_domain.__name__
    }
)


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__)
def test_rule_docstring_examples_execute(module):
    """Each rule/form docstring's ``>>>`` example produces exactly the documented output."""
    finder = doctest.DocTestFinder()
    failures: list[str] = []
    for test in finder.find(module, module.__name__, extraglobs=_GLOBS):
        # A fresh runner per docstring — DocTestRunner.run reports cumulative counts.
        runner = doctest.DocTestRunner(optionflags=doctest.FAIL_FAST)
        result = runner.run(test, clear_globs=False)
        if result.failed:
            failures.append(test.name)
    assert not failures, f"doctest failures in {module.__name__}: {failures}"
