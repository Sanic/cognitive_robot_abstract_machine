"""
Tests for de-cluttered grouped ``set_of`` verbalization (ROUND 9):

- the literal *"sets of"* head is dropped — a set-of reads *"Find (v1, v2, …) …"*;
- a query whose chains all share one root variable pronominalises to *"its …"* / *"their …"*
  instead of restating the full root (a ``set_of`` previously had no discourse focus);
- a ``limit`` on a set-of renders as a ranking (*"Find the top three (…)"*) and suppresses the
  standalone *"ordered by …"* clause.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import a, variable, set_of
import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

from ...dataset.department_and_employee import Employee


@dataclass
class Bound:
    month: int


@dataclass
class Period:
    begin: Bound
    end: Bound


@dataclass
class Money:
    amount: float


@dataclass
class Revenue:
    money: Money


@dataclass
class ProfitAndLossStatement:
    period: Period
    revenue: Revenue


# ── "sets of" is dropped ──────────────────────────────────────────────────────


def test_set_of_drops_the_sets_of_head():
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(e.department, e.salary)))
    assert text.startswith("Find (")
    assert "sets of" not in text


# ── single-root pronominalisation ─────────────────────────────────────────────


def test_single_root_set_of_pronominalises():
    """Every chain roots at one Employee → the repeated root collapses to "its"/"their"."""
    e = variable(Employee, [])
    text = verbalize_expression(
        a(set_of(e.department, eql.sum(e.salary)).grouped_by(e.department))
    )
    # the grouped-by key no longer restates "of the Employee"
    assert "grouped by their department" in text
    assert "grouped by the department of the Employee" not in text


def test_multi_root_set_of_does_not_pronominalise():
    """Two distinct root variables → no single focus → no pronoun, full chains kept."""
    e1 = variable(Employee, [])
    e2 = variable(Employee, [])
    text = verbalize_expression(a(set_of(e1.department, e2.salary)))
    assert "its " not in text
    assert "their " not in text


def test_unlimited_ordered_set_of_keeps_clause_pronominalised():
    e = variable(Employee, [])
    text = verbalize_expression(a(set_of(e).ordered_by(e.salary, descending=True)))
    assert text == "Find (an Employee) ordered by its salary (descending)"


# ── limit on a set_of → ranking, ordered-by suppressed ───────────────────────


def _pnl_query(n: int):
    pnl = variable(ProfitAndLossStatement, domain=None)
    revenue = eql.sum(pnl.revenue.money.amount)
    return a(
        set_of(pnl.period, revenue)
        .grouped_by(pnl.period)
        .where(pnl.period.begin.month == pnl.period.end.month)
        .ordered_by(revenue, descending=True)
        .limit(n)
    )


def test_set_of_limit_renders_ranking_and_suppresses_ordered_by():
    text = verbalize_expression(_pnl_query(3))
    assert text == (
        "Find the top three (the period of a ProfitAndLossStatement, "
        "the sum of the amount of the money of its revenue) "
        "such that the month of the begin of its period is the month of the end of its period, "
        "grouped by its period"
    )
    # the ranking conveys the ordering — no standalone clause, and the root appears once
    assert "ordered by" not in text
    assert "descending" not in text
    assert text.count("ProfitAndLossStatement") == 1


def test_set_of_limit_ascending_is_bottom():
    pnl = variable(ProfitAndLossStatement, domain=None)
    revenue = eql.sum(pnl.revenue.money.amount)
    q = a(
        set_of(pnl.period, revenue).grouped_by(pnl.period).ordered_by(revenue).limit(2)
    )
    text = verbalize_expression(q)
    assert text.startswith("Find the bottom two (")
    assert "ordered by" not in text
