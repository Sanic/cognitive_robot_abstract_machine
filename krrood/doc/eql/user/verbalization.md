---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Verbalization

EQL queries are Python objects — they can be inspected, composed, stored, and now: **read aloud**.

Verbalization turns any EQL expression into a plain-English sentence. This is useful for:

- **Debugging** — instantly understand what a complex query actually asks.
- **Explainability** — surface query intent in logs, UIs, or reports.
- **Testing** — assert on what a query means, not just what it returns.

## The Quick API

The simplest way to verbalize any EQL expression is `verbalize_expression`.

```{code-cell} ipython3
from dataclasses import dataclass
from krrood.entity_query_language.factories import variable, entity, an
from krrood.entity_query_language.verbalization.verbalizer import verbalize_expression

@dataclass
class Robot:
    name: str
    battery: int

robots = [Robot("R2D2", 95), Robot("C3PO", 20), Robot("BB8", 80)]
r = variable(Robot, domain=robots)

query = an(entity(r).where(r.battery > 50))
print(verbalize_expression(query))
```

The output reads like a natural sentence describing exactly what the query selects.

## More Conditions, Still Readable

Adding more `.where()` conditions does not break the sentence — EQL connects them naturally.

```{code-cell} ipython3
query = an(entity(r).where(r.battery > 50, r.name != "BB8"))
print(verbalize_expression(query))
```

## Cross-Variable Conditions

Verbalization handles cross-variable comparisons too — the sentence describes the *relationship* between variables.

```{code-cell} ipython3
@dataclass
class Mission:
    assigned_to: Robot
    priority: int

missions = [Mission(robots[0], 1), Mission(robots[1], 3)]
m = variable(Mission, domain=missions)

query = an(entity(r).where(m.assigned_to == r, m.priority > 2))
print(verbalize_expression(query))
```

## Logical Operators

Conditions combined with `and_`, `or_`, and `not_` are verbalized with natural connectives.

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, and_, or_, not_

x = variable(int, [1, 5, 12])
print(verbalize_expression(and_(x > 1, x < 10, x != 5)))
print(verbalize_expression(or_(x > 10, x < 0)))
print(verbalize_expression(not_(x > 5)))
```

Notice the `or_` form opens with *"either … or …"* for readability, and chained `and_` conditions
separate each clause with a comma and the final *"and"*.

## Boolean and Indexed Attributes

An attribute whose type is `bool` uses a predicative form — *"<nav-path> is <attribute>"* —
rather than the possessive form used for non-boolean attributes.

```{code-cell} ipython3
from typing import List
from krrood.entity_query_language.factories import variable, not_

@dataclass
class Task:
    name: str
    completed: bool

@dataclass
class Worker:
    name: str
    tasks: List[Task]

w = variable(Worker, domain=None)
print(verbalize_expression(w.tasks[0].completed))
print(verbalize_expression(not_(w.tasks[0].completed)))
```

A numeric index like `[0]` becomes an ordinal (*"the first …"*), and the terminal boolean
field maps to *"is completed"* / *"is not completed"*.

## Aggregations

Aggregation functions (`count`, `sum`, `average`, `max`, `min`) are wrapped with
the definite article and a descriptive phrase when verbalized.  Here we need a domain with
a numeric field:

```{code-cell} ipython3
import datetime
from krrood.entity_query_language.factories import variable
import krrood.entity_query_language.factories as eql

@dataclass
class AmountDetails:
    amount: float

@dataclass
class BankTransaction:
    amount_details: AmountDetails
    booking_date: datetime.datetime

t = variable(BankTransaction, domain=None)

print(verbalize_expression(eql.count(t)))
print(verbalize_expression(eql.sum(t.amount_details.amount)))
print(verbalize_expression(eql.average(t.amount_details.amount)))
print(verbalize_expression(eql.max(t.amount_details.amount)))
print(verbalize_expression(eql.min(t.amount_details.amount)))
```

All aggregations use the definite article (*"the number of"*, *"the sum of"*, ...).
The attribute chain following the aggregation uses a possessive *"of the ..."* path.

## Date Range Folding

When a lower-bound and an upper-bound comparison on the same datetime attribute appear
together, the verbalizer folds them into a single *"between ... and ..."* phrase.

```{code-cell} ipython3
bt = variable(BankTransaction, domain=None)
lo = datetime.datetime(2026, 5, 15)
hi = datetime.datetime(2026, 5, 30)

query = an(entity(bt).where(bt.booking_date >= lo, bt.booking_date <= hi))
print(verbalize_expression(query))
```

The output uses *"is between ... and ..."* with the datetime values formatted in a
human-readable form.  The same folding happens when the comparisons appear on an
aggregation sub-query's WHERE clause (see the next section).

## Nested Sub-Queries and Aggregation Scoping

Aggregation sub-queries nest naturally.  A scoped aggregation — *"the sum of amounts
among BankTransactions whose booking_date is between ..."* — is produced when an aggregate
appears inside an `entity()` wrapper with its own WHERE conditions.

```{code-cell} ipython3
bt  = variable(BankTransaction, domain=None)
bt_sum = variable(BankTransaction, domain=None)
start = datetime.datetime(2026, 5, 15)
end   = datetime.datetime(2026, 5, 30)

sum_val = an(entity(eql.sum(bt_sum.amount_details.amount)).where(
    bt_sum.booking_date >= start, bt_sum.booking_date <= end,
))
query = an(entity(bt).where(bt.amount_details.amount == sum_val))
print(verbalize_expression(query))
```

The possessive pronoun *"its"* replaces a repeated *"of the BankTransaction"* on the
outer condition.  The scoped aggregation automatically uses the preposition *"among"*
and the WHERE conditions inside the sub-query continue to work (date-range folding,
pronouns, etc.).

A maximum-value variant produces a similarly compact form:

```{code-cell} ipython3
bt_max = variable(BankTransaction, domain=None)
max_val = an(entity(eql.max(bt_max.amount_details.amount)))
query = an(entity(bt).where(bt.amount_details.amount == max_val))
print(verbalize_expression(query))
```

And a scoped aggregation can stand alone as the main query — no outer entity needed:

```{code-cell} ipython3
cutoff = datetime.datetime(2024, 5, 17)
scoped_sum = an(entity(eql.sum(bt.amount_details.amount)).where(bt.booking_date < cutoff))
print(verbalize_expression(scoped_sum))
```

## Same-Type Variable Disambiguation

When two variables of the same type appear in a query, the verbalizer distinguishes
them by appending a numeric index — *"Employee 1"*, *"Employee 2"*.

```{code-cell} ipython3
@dataclass
class Employee:
    name: str
    department: str
    salary: int
    starting_salary: int

emp1 = variable(Employee, domain=None)
emp2 = variable(Employee, domain=None)
query = an(entity(emp1).where(emp1.salary > emp2.salary))
print(verbalize_expression(query))
```

The same mechanism also handles disambiguation when an aggregate and an entity share
a type:

```{code-cell} ipython3
emp = variable(Employee, domain=None)
query_agg = an(entity(eql.average(emp.salary)).where(emp.starting_salary > 20000))
print(verbalize_expression(query_agg))
```

## Custom Predicates

A custom predicate can control its verbalization by implementing
`_verbalization_template_`.  The template is a string with ``{field_name}`` placeholders
corresponding to the predicate's dataclass fields.

```{code-cell} ipython3
from krrood.entity_query_language.predicate import Predicate

@dataclass(eq=False)
class Location:
    name: str

@dataclass(eq=False)
class IsReachable(Predicate):
    body: object
    def __call__(self):
        return True
    @classmethod
    def _verbalization_template_(cls):
        return "{body} is reachable"

loc = variable(Location, domain=None)
print(verbalize_expression(IsReachable(loc)))
```

Predicates with multiple fields receive their arguments in positional order:

```{code-cell} ipython3
@dataclass(eq=False)
class WorksIn(Predicate):
    employee: object
    department: object
    def __call__(self):
        return True
    @classmethod
    def _verbalization_template_(cls):
        return "{employee} works in {department}"

@dataclass(eq=False)
class Department:
    name: str

@dataclass(eq=False)
class StaffMember:
    name: str
    department: Department

dept = variable(Department, domain=None)
emp = variable(StaffMember, domain=None)
print(verbalize_expression(WorksIn(emp, dept)))
```

When a predicate does not define `_verbalization_template_`, the verbalizer falls
back to a generic description.

## Grouped Queries (`set_of` + `grouped_by` + `having`)

Queries built with `set_of`, `grouped_by`, and `having` are verbalized as a structured
sentence with the selection, GROUP BY, and HAVING clauses clearly separated.

```{code-cell} ipython3
from krrood.entity_query_language.factories import a, set_of

emp = variable(Employee, domain=None)
avg_salary = eql.average(emp.salary)
query = a(
    set_of(emp.department, avg_salary)
    .grouped_by(emp.department)
    .having(avg_salary > 30000)
)
print(verbalize_expression(query))
```

The HAVING clause uses the *compact* (copula-less) operator form — *"greater than 30000"*
instead of *"is greater than 30000"* — matching SQL-style conciseness.  The GROUP BY
clause states only the grouping key without restating the full selection tuple.

## Colored Terminal Output

For richer output in a terminal, use `VerbalizationPipeline.ansi()`. Each part of the sentence is
color-coded by its semantic role.

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

pipeline = VerbalizationPipeline.ansi()
print(pipeline.verbalize(query))
```

Color legend:

| Color | Role | Example |
|---|---|---|
| Cornflower blue | **Variable type** | `Robot`, `Mission` |
| Teal | **Attribute** | `battery`, `assigned_to` |
| Orange | **Operator** | `is greater than`, `is not` |
| Green | **Logical connective** | `and`, `or`, `such that` |
| Gray | **Literal value** | `50`, `"BB8"` |
| Yellow | **Keyword / rule structure** | `If`, `then`, `whose` |

## HTML Output for Notebooks

`VerbalizationPipeline.html()` produces `<span>` tags for direct use in Jupyter or any HTML context.

```{code-cell} ipython3
from IPython.display import HTML
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

pipeline = VerbalizationPipeline.html()
HTML(pipeline.verbalize(query))
```

### Hierarchical HTML

Pass `hierarchical=True` to get an indented bullet structure — great for rule trees.

```{code-cell} ipython3
HTML(VerbalizationPipeline.html(hierarchical=True).verbalize(query))
```

## Verbalizing Rule Trees

Verbalization really shines on rule trees. The if/then structure is rendered clearly.

```{code-cell} ipython3
from krrood.entity_query_language.factories import (
    variable, entity, an, deduced_variable, add, inference, refinement, Symbol, not_
)
@dataclass
class Bird:
    name: str

@dataclass
class LoveBirds:
    bird_1: Bird
    bird_2: Bird
    strong_love: bool

@dataclass
class BirdView(Symbol):
    bird: Bird

@dataclass
class StrongLoveBird(BirdView): pass

@dataclass
class WeakLoveBird(BirdView): pass

birds = [Bird('tweety'), Bird('snappy'), Bird('sleepy')]
bird = birds[0]
love_birds = [
    LoveBirds(birds[0], birds[1], True),
    LoveBirds(birds[1], birds[2], False),
]

love_birds = variable(LoveBirds, domain=love_birds)
bird_view = deduced_variable(BirdView)
rule_query = an(entity(inference(StrongLoveBird)(bird=love_birds.bird_1)).where(love_birds.strong_love))

HTML(VerbalizationPipeline.html(hierarchical=True).verbalize(rule_query))
```

The hierarchical renderer shows the **If/then** structure with each condition and conclusion
on its own line, indented under the relevant clause.

### Deep Nesting in Hierarchical Mode

The hierarchical view really shines on rules with *deeply nested* attribute chains — the
bullet structure makes the relationship between conditions visually clear.  Here is a
drawer-detection rule with a multi-hop path:

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, entity, an, inference

@dataclass
class Handle:
    name: str

@dataclass
class Container:
    name: str

@dataclass
class FixedConnection:
    parent: Container
    child: Handle

@dataclass
class PrismaticConnection:
    parent: Container
    child: Container

@dataclass
class Drawer:
    container: Container
    handle: Handle

fc = variable(FixedConnection, domain=None)
pc = variable(PrismaticConnection, domain=None)
h  = variable(Handle, domain=None)
drawer_rule = an(entity(inference(Drawer)(
    container=fc.parent,
    handle=fc.child,
)).where(
    fc.parent == pc.child,
    fc.child == h,
))

HTML(VerbalizationPipeline.html(hierarchical=True).verbalize(drawer_rule))
```

And a cabinet rule that aggregates over multiple drawers — notice the *aggregated* antecedent
uses *"there are"* and the THEN clause bindings use plural *"are"*:

```{code-cell} ipython3
@dataclass
class Cabinet:
    container: Container
    drawers: list

pc = variable(PrismaticConnection, domain=None)
dr  = variable(Drawer, domain=None)
cabinet_rule = an(entity(inference(Cabinet)(
    container=pc.parent,
    drawers=dr,
)).where(pc.child == dr.container))

HTML(VerbalizationPipeline.html(hierarchical=True).verbalize(cabinet_rule))
```

## Verbalization as an Explanation Tool

Because EQL tracks *how* inferences were made, you can verbalize the query that produced any
inferred result — not just hand-written queries.

```{code-cell} ipython3
from krrood.entity_query_language.explanation.explanation import explain_inference

inferred_views = list(rule_query.evaluate())
inferred_object = inferred_views[0]

explanation = explain_inference(inferred_object)
print(verbalize_expression(explanation.query_root))
```

This outputs the exact query that matched and produced `inferred_object`, described in English.
It is directly useful for displaying *why* a robot perceives something as a Drawer, a Door, etc.

## Hyperlinks to Source Code

Pass `link_resolver=AutoAPIResolver(...)` to any pipeline factory and class and attribute
names become clickable links — opening the corresponding Sphinx AutoAPI documentation page.

`AutoAPIResolver` works in two modes:

| Mode | How to construct | When it works |
|---|---|---|
| **Local** | `AutoAPIResolver.for_package("krrood")` | After `sphinx-build doc doc/_build/html` |
| **GitHub Pages** | `AutoAPIResolver(base_url="https://cram2.github.io/…/krrood")` | Always, no local build needed |

The demo below uses `verbalization_domain.Robot` and `verbalization_domain.Mission` — classes
defined in `doc/eql/user/verbalization_domain.py` whose mock API page is committed alongside
this notebook and deployed with the docs.

```{code-cell} ipython3
:tags: [remove-cell]

import sys
from pathlib import Path
# verbalization_domain.py lives in doc/eql/user/.
# When the notebook kernel runs from test_tmp/, Path("..") resolves there.
for _candidate in [Path("doc/eql/user"), Path("eql/user"), Path(".."), Path(".")]:
    if (_candidate / "verbalization_domain.py").exists():
        sys.path.insert(0, str(_candidate.resolve()))
        break
```

```{code-cell} ipython3
from verbalization_domain import Robot, Mission
from krrood.entity_query_language.factories import variable, entity, an
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import AutoAPIResolver

vd_robots = [Robot("R2D2", 95), Robot("C3PO", 20)]
vd_missions = [Mission(vd_robots[0], 3)]
r = variable(Robot, domain=vd_robots)
m = variable(Mission, domain=vd_missions)
linked_query = an(entity(r).where(m.assigned_to == r, m.priority > 2))

# Local — requires docs to be built first: sphinx-build doc doc/_build/html
resolver = AutoAPIResolver.for_package("krrood")
VerbalizationPipeline.html(link_resolver=resolver).display(linked_query)
```

```{code-cell} ipython3
# GitHub Pages — always available, no local build needed.
resolver = AutoAPIResolver(base_url="https://cram2.github.io/cognitive_robot_abstract_machine/krrood")
VerbalizationPipeline.html(link_resolver=resolver).display(linked_query)
```

## API Reference

- {py:func}`~krrood.entity_query_language.verbalization.verbalizer.verbalize_expression` — plain text, one-liner
- {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline` — full control over format and color
  - `.plain()` — plain text, paragraph prose
  - `.ansi()` — ANSI true-color terminal output
  - `.ansi(hierarchical=True)` — indented bullet structure, ANSI
  - `.html()` — HTML `<span>` colors, paragraph prose
  - `.html(hierarchical=True)` — HTML, indented bullet structure
  - `.html(link_resolver=...)` — adds clickable hyperlinks to class and attribute names
  - `.display(expression)` — renders inline in Jupyter or opens a browser tab elsewhere
