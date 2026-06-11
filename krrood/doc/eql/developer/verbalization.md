---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Verbalization Internals

This guide explains the architecture of the EQL verbalization subsystem for developers who want to understand, extend, or debug it.  End-user documentation lives in {doc}`../user/verbalization`.

## Overview

The verbalization subsystem translates any EQL symbolic expression into a human-readable English string (or a structured fragment tree that can be rendered in plain text, ANSI colour, or HTML).

The single entry point is {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`, with {py:func}`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression` as the plain-text shortcut:

```python
# Simplest — plain text, no colour (== VerbalizationPipeline.plain().verbalize(...))
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
text = verbalize_expression(query)

# Colour / layout / links — choose a pipeline factory
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

VerbalizationPipeline.plain().verbalize(query)                    # plain prose
VerbalizationPipeline.ansi().verbalize(query)                     # ANSI true-colour prose
VerbalizationPipeline.ansi(hierarchical=True).verbalize(query)    # ANSI, indented bullets
VerbalizationPipeline.html(link_resolver=resolver).verbalize(query)  # HTML <span> + links

# Full control — construct a pipeline from a renderer + formatter directly
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import HierarchicalRenderer
pipeline = VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter(), resolver))
html = pipeline.verbalize(query)
```

Pass a shared {py:class}`~krrood.entity_query_language.verbalization.context.VerbalizationContext` to `verbalize` across calls to get cross-mention coreference (*"a Robot"* … *"the Robot"*).  A construct with no grammar rule raises {py:class}`~krrood.entity_query_language.verbalization.engine.UnverbalizableExpressionError` — coverage gaps fail loudly rather than degrading to a bare class name.

---

## Architecture: build → realise → render

```{mermaid}
graph LR
    A[EQL Expression] --> B[fold + realisation passes]
    B -- realised VerbFragment tree --> C[FragmentRenderer]
    C -- formatted string --> D[Output]
    E[grammar RULES / select] -. dispatch .-> B
    F[VerbalizationContext] -. services .-> B
    G[Formatter] -. markup .-> C
    H[SourceLinkResolver] -. URLs .-> C
```

### Layer 1 — Fragment building + realisation (`EQLVerbalizer.build`)

{py:class}`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer` is the internal fragment builder behind the pipeline (use it directly only when you want the fragment tree itself, e.g. in tests).  `build(expression, context)`:

1. **Folds** the EQL tree into a {py:class}`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` tree via the grammar (see [Rule dispatch](#rule-dispatch-the-fold)).
2. Runs the ordered **realisation passes** ({py:func}`~krrood.entity_query_language.verbalization.rendering.realization.realize_tree`): coreference → determiner → morphology (see [Realisation passes](#realisation-passes)).

It never produces strings — formatting is Layer 2/3.

### Layer 2 — Fragment rendering (`FragmentRenderer`)

{py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer` traverses the realised tree and produces a single string.

| Renderer | Output style |
|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` | Flat prose; `BlockFragment`s joined inline |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer` | Indented bullet lists; each `BlockFragment` nesting level adds one indent |

### Layer 3 — Format markup (`Formatter`)

{py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter` injects format-specific characters into the renderer output.

| Formatter | Colour encoding | Space | Newline | Links |
|---|---|---|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` | none | `" "` | `"\n"` | no |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.ANSIFormatter` | `\033[38;2;R;G;Bm` | `" "` | `"\n"` | OSC 8 |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.HTMLFormatter` | `<span style="color:…">` | `&nbsp;` | `<br>` | `<a href>` |

---

## Fragment Type Hierarchy

All verbalization output is expressed as a tree of `VerbFragment` subclasses.  There are **leaf** nodes (text), **structural** containers (hold children), and two **coreference markers** the realisation passes consume and strip.

```{mermaid}
classDiagram
    class VerbFragment {
        <<abstract>>
    }
    class WordFragment {
        role-less text: articles, punctuation, connectives
    }
    class RoleFragment {
        text + SemanticRole + optional SourceRef (for hyperlinking)
    }
    class PhraseFragment {
        inline sequence of children joined by a separator
    }
    class NounPhrase {
        head + Definiteness + Number + modifiers + referent_id (a DP spec, lowered later)
    }
    class BlockFragment {
        header + item fragments (flattens or indents on render)
    }
    class SubjectScope {
        marks the pronoun-eligible subject region (stripped after coreference)
    }
    class PossessiveChain {
        a chain whose its/of form coreference decides (stripped after coreference)
    }
    VerbFragment <|-- WordFragment
    VerbFragment <|-- RoleFragment
    VerbFragment <|-- PhraseFragment
    VerbFragment <|-- NounPhrase
    VerbFragment <|-- BlockFragment
    VerbFragment <|-- SubjectScope
    VerbFragment <|-- PossessiveChain
```

`NounPhrase` is a *spec*, not a lowered phrase: rules emit it with grammatical features (`Definiteness`, `Number`) but **no** determiner; the [determiner pass](#realisation-passes) lowers it to a `PhraseFragment`.  The two recursion helpers over this tree are {py:func}`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment` (a catamorphism to any value — used by the renderer/flatten) and {py:func}`~krrood.entity_query_language.verbalization.fragments.base.map_structural_children` (a structure-preserving rebuild — used by the realisation passes).

### SemanticRole and Colours

{py:class}`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole` is a *presentation* classification (decoupled from the EQL type taxonomy) that determines the colour applied by formatters, and is read by the morphology pass for copula agreement (`OPERATOR`).  Colours match the `QueryGraph.ColorLegend` palette.

| Role          | Example | Colour |
|---------------|---|---|
| `KEYWORD`     | *Find*, *If*, *such that* | yellow `#eded18` |
| `VARIABLE`    | *Robot*, *Employee 1* | cornflower blue |
| `AGGREGATION` | *sum of*, *number of* | red-orange `#F54927` |
| `OPERATOR`    | *is greater than*, *is* | orange `#ff7f0e` |
| `LOGICAL`     | *and*, *or*, *not*, *for all* | green `#2ca02c` |
| `LITERAL`     | `42`, `"hello"` | gray `#949292` |
| `ATTRIBUTE`   | *battery*, *tasks* | teal `#8FC7B8` |
| `PLAIN`       | *of*, *the*, *,* (`WordFragment`, role-less) | none |

### Building Fragments

Construct fragments **directly** (there are no factory helpers), and prefer the vocabulary constants for any fixed word so no English string is ever inlined:

```python
from krrood.entity_query_language.verbalization.fragments.base import (
    WordFragment, RoleFragment, PhraseFragment, NounPhrase, BlockFragment, oxford_and,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles, Conjunctions, Punctuation,
)

# Fixed words / punctuation → vocabulary constants (avoids typos, carries the role)
the   = Articles.THE.as_fragment()
comma = Punctuation.COMMA.as_fragment()

# Coloured word with a source reference (for hyperlinking variables / attributes)
role_frag = RoleFragment.for_variable("Robot", robot_var)   # VARIABLE role + source link
attr_frag = RoleFragment.for_attribute("battery", Robot)    # ATTRIBUTE role + link
op_frag   = RoleFragment.for_operator("is greater than")    # OPERATOR role, no link

# Inline sequence (default " " separator)
phrase = PhraseFragment(parts=[the, role_frag, op_frag])

# Oxford-comma join
list_frag = oxford_and([frag_a, frag_b, frag_c], Conjunctions.AND.as_fragment())

# Noun-phrase spec (the determiner is added by the determiner pass, not here)
np = NounPhrase(head=RoleFragment.for_variable("Robot", robot_var), referent_id=robot_var._id_)

# Block structure (renders as bullets in HierarchicalRenderer)
block = BlockFragment(header=keyword_frag, items=[item1, item2])
```

---

## Rule dispatch (the fold)

{py:func}`~krrood.entity_query_language.verbalization.engine.fold` is the **single** place the EQL tree is recursed — an F-algebra catamorphism.  For each node it:

1. Checks `context.binding.binding_overrides` for the node's `_id_` — returns the override immediately if present (used for `InstantiatedVariable` field references).
2. {py:func}`~krrood.entity_query_language.verbalization.grammar.phrase_rule.select`s the most-specific {py:class}`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule` and calls its `build`, handing it a fresh {py:class}`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx` whose `child` re-enters `fold`.
3. If no rule matches, raises {py:class}`~krrood.entity_query_language.verbalization.engine.UnverbalizableExpressionError`.

A rule never recurses by hand — it calls `ctx.child(sub_expression)`.

### Specificity

`select` ranks the rules whose `construct` matches (`isinstance`) and whose `when` guard passes, by the key **(construct MRO depth, guarded-over-unguarded, explicit `tiebreak`)**, highest wins.  Specificity comes from the *construct* class, not from a rule-class hierarchy, so rules stay flat.  For example `Literal <: Variable`, so `LiteralRule` (deeper construct) shadows `VariableRule`; a guarded `RangeConjunctionRule` (an `AND` containing a lo/hi pair) outranks the plain `AndRule`.

### How to Extend — A Worked Example

This walkthrough adds verbalization for a hypothetical `Between` operator (`between(x, lo, hi)`) rendering as *"x is between lo and hi"*.

#### Step 1 — Where does the code go?

* **A simple construct** (one phrase, no content decisions) → add one `PhraseRule` subclass to {py:mod}`~krrood.entity_query_language.verbalization.grammar.english`.  That's the whole change.
* **A complex construct** (needs *what to say* analysis separate from *how to say it*) → add a `Planner` ({py:mod}`grammar.planning <krrood.entity_query_language.verbalization.grammar.planning>`) + `Assembler` ({py:mod}`grammar.assembly <krrood.entity_query_language.verbalization.grammar.assembly>`) pair and have the rule's `build` delegate to the assembler.

`Between` is simple, so it's a single rule in `grammar/english.py`.

#### Step 2 — Write the rule

Set `construct`, optionally `name`/`tiebreak`/`when`, and implement `build(node, ctx)`:

```python
class BetweenRule(PhraseRule):
    """Verbalizes a Between operator as *"x is between lo and hi"*."""

    construct = Between
    name = "between"

    def build(self, node, ctx: Ctx) -> VerbFragment:
        return PhraseFragment(
            parts=[
                ctx.child(node.left),                       # recurse via the fold
                RangePhrases.IS_BETWEEN.as_fragment(),      # fixed phrase from the lexicon
                oxford_and(
                    [ctx.child(node.lo), ctx.child(node.hi)],
                    Conjunctions.AND.as_fragment(),
                ),
            ]
        )
```

#### Step 3 — Registration is automatic

`RULES` is **auto-discovered**: it instantiates every concrete `PhraseRule` subclass *defined in* `grammar/english.py` (via `classes_of_module`).  Defining `BetweenRule` there is enough — no list to maintain.  (`select` is specificity-based, so definition order is irrelevant.)

#### Step 4 — Recurse with `ctx.child`, decide with `ctx` services

`ctx.child(sub_expression)` re-enters the fold, so sub-expressions get coreference, binding overrides, and pronoun resolution for free.  Reach cross-cutting state only through the `Ctx` services — `ctx.refer` (referring expressions), `ctx.scope` (binding), `ctx.config` (render flags) — never `ctx.context.<service-method>` directly.  Never call `verbalize_expression(child)` from a rule: that starts a fresh context and breaks coreference.

#### Step 5 — Use constructors + vocabulary constants

Build fragments with the constructors (`WordFragment`/`RoleFragment`/`PhraseFragment`/`NounPhrase`), and source every fixed word from the lexicon (`Keywords`, `Copulas`, `Operators`, `Punctuation`, …) via `.as_fragment()`.  No English string — including punctuation — belongs in a rule.

| Enum | Example | `.as_fragment()` role |
|---|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords` | `Keywords.FIND` | KEYWORD |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Copulas` | `Copulas.IS` / `Copulas.for_number(n)` | OPERATOR |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Operators` | `Operators.from_callable(op.le).select(...)` | OPERATOR |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Logicals` | `Logicals.FOR_ALL` | LOGICAL |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Aggregations` | `Aggregations.COUNT` | AGGREGATION |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Articles` | `Articles.THE` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Conjunctions` | `Conjunctions.AND` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Prepositions` | `Prepositions.OF` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Punctuation` | `Punctuation.COMMA` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Pronouns` | `Pronouns.ITS` | PLAIN |

For coloured / linked fragments referencing a Python class or attribute, use the `RoleFragment` class methods:

```python
RoleFragment.for_variable("Robot", var)       # VARIABLE role + SourceRef to type
RoleFragment.for_attribute("battery", Robot)  # ATTRIBUTE role + SourceRef to attr
RoleFragment.for_operator("is greater than")  # OPERATOR role, no link
```

#### Step 6 — Write a test

Add a golden-string case to `test/krrood_test/test_eql/test_verbalization/test_verbalization.py`:

```python
def test_verbalize_between():
    assert verbalize_expression(between(x, lo, hi)) == "x is between lo and hi"
```

```bash
pytest test/krrood_test/test_eql/test_verbalization -x
```

---

## The planner / assembler split (complex constructs)

Queries, inference rules, and instantiated variables are too involved for a one-method rule, so they split *what to say* from *how to say it*:

* a {py:class}`~krrood.entity_query_language.verbalization.grammar.planning.base.Planner` performs pure structural analysis into a frozen plan (no fragments, no context mutation, no recursion);
* an {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.base.Assembler` realises that plan into fragments (it owns recursion and the render-scope mutations).

The rule's `build` just calls `XAssembler(ctx).assemble(node)`.  The orchestrating {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.query.QueryAssembler` delegates cohesive sub-forms to their own components: the trailing clauses to {py:mod}`~krrood.entity_query_language.verbalization.grammar.assembly.clauses`, the WHERE partition to {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.restrictions.RestrictionAssembler`, and the aggregation value-subquery to {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.aggregation_value.AggregationValueAssembler`.

Navigation chains are realisation-only (no plan): {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.chains.ChainAssembler` analyses the chain once into a {py:class}`~krrood.entity_query_language.verbalization.chain_utils.ChainAnalysis` and dispatches to the plural / bool-predicative / possessive / pronominal-deferred form, the possessive/pronominal surface built by {py:mod}`~krrood.entity_query_language.verbalization.microplanning.possessive`.

---

## Realisation passes

After the fold, {py:func}`~krrood.entity_query_language.verbalization.rendering.realization.realize_tree` runs three ordered passes over the fragment tree (each a `map_fragment` rebuild):

1. **{py:class}`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`** — resolves referring expressions in document order and strips `SubjectScope` / `PossessiveChain` markers (below).
2. **{py:class}`~krrood.entity_query_language.verbalization.rendering.determiner_processor.DeterminerProcessor`** — lowers every `NounPhrase` to a determiner-bearing `PhraseFragment` via the concord table (INDEFINITE×SINGULAR → *a/an*; INDEFINITE×PLURAL → bare; DEFINITE → *the*; UNIQUE → *the unique*), and tags the head with its `Number`.
3. **{py:class}`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`** — inflects every leaf tagged `Number.PLURAL` (pluralise nouns; copula suppletion *is*→*are*).

### Coreference

Rules emit the *first-mention* form — a `NounPhrase` tagged with a `referent_id` (and `Definiteness` INDEFINITE / UNIQUE / BARE), optionally wrapped in a `SubjectScope(subject_id, …)`, with variable-rooted chains emitted as a `PossessiveChain`.  The pass then, in output order:

* keeps the first mention of a referent (e.g. *"a Robot"*) and marks it introduced;
* downgrades a repeat **singular** mention to definite, dropping the first-mention modifiers (*"the Robot"*); `UNIQUE` (*"the unique Robot"*) downgrades to DEFINITE (*"the Robot"*); numbered labels (*"Robot 2"*, `BARE`) never downgrade;
* renders a `PossessiveChain` as *"its …"* when its root is the current `SubjectScope` subject, else as the possessive *"the … of …"*.

The build is therefore free of in-fold coreference mutation.  {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` holds only the pre-computed disambiguation map and a `seen` **set** of introduced referent ids — the latter solely to *seed* the pass across builds sharing one context.  `numbered_label(var)` is the single source of the disambiguation lookup (returns `(label, is_numbered)` and records the mention); `noun_for_parts(var)` builds on it for the first-mention `Definiteness`.

### Disambiguation map

Built by `VerbalizationContext.from_expression(expression)`, which pre-scans the tree.  Types with a single variable keep the plain type name; collisions get numbered labels:

```
Robot    (single)  →  "Robot"
Apple 1  (first)   →  "Apple 1"
Apple 2  (second)  →  "Apple 2"
```

---

## Microplanning services

A single `VerbalizationContext` threads three single-responsibility services through one build; rules reach them via the `Ctx` properties.

| Service (`Ctx` accessor) | Responsibility |
|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` (`ctx.refer`) | disambiguation map + introduced-referent `seen` set |
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope` (`ctx.scope`) | deferred-constraint frames + field-reference overrides |
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfig` (`ctx.config`) | render-mode flags (query depth, compact predicates) |

`VerbalizationContext` itself exposes only those three service objects plus the one cross-service helper, `type_name_of_value` (Python value → text).

### Constraint frames (BindingScope)

Used by the `InstantiatedVariable` path: when an `Entity` is a chain root inside an `InstantiatedVariable`, its WHERE condition is *deferred* into a frame rather than verbalized inline.  After all binding overrides are registered, the deferred expressions are emitted as a *"such that …"* clause.

```python
ctx.scope.push_constraint_frame()
ctx.scope.defer_constraint(expression)
deferred = ctx.scope.pop_constraint_frame()
```

### Binding overrides (BindingScope)

`ctx.scope.binding_overrides` maps `expression._id_` → `VerbFragment`.  `fold` checks it before dispatch, so when a bound variable appears again as a WHERE value it reuses the *"the field of the Type"* fragment instead of re-verbalizing the raw variable.

---

## Coordination (aggregation / conjunction reduction)

{py:mod}`~krrood.entity_query_language.verbalization.microplanning.coordination` owns the EQL-level *aggregation* microplanning task: {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.fold_range_pairs` folds complementary lower/upper bound comparisons on the same chain into a `RangeFold`, rendered as *"… is between lo and hi"* by `build_between`.  (Fragment-level Oxford-comma joining is {py:func}`~krrood.entity_query_language.verbalization.fragments.base.oxford_and`.)

---

## Source References and Link Resolvers

{py:class}`~krrood.entity_query_language.verbalization.fragments.source_ref.SourceRef` identifies the Python entity a `RoleFragment` represents:

```python
SourceRef.for_type(Robot)                 # class reference
SourceRef.for_attribute(Robot, "battery") # attribute reference
```

A {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver` maps these to URLs; the built-in {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver` builds Sphinx AutoAPI URLs:

```python
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import AutoAPIResolver
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

resolver = AutoAPIResolver.for_package("krrood")
pipeline = VerbalizationPipeline.html(link_resolver=resolver)
```

The resolver is passed to the renderer at construction (via the `VerbalizationPipeline` factories).  When a renderer meets a `RoleFragment` with a non-`None` `source_ref` and a resolver, it wraps the coloured text in a hyperlink.

---

## Module map

| Area | Modules |
|---|---|
| Engine | `engine.py` (`fold`, `UnverbalizableExpressionError`), `verbalizer.py` (`EQLVerbalizer`), `pipeline.py` (`VerbalizationPipeline`, `verbalize_expression`), `context.py` |
| Fragment IR | `fragments/base.py` (the `VerbFragment` hierarchy + `fold_fragment` / `map_structural_children` / `oxford_and`), `fragments/features.py`, `fragments/roles.py`, `fragments/source_ref.py` |
| Lexicon | `vocabulary/english.py`, `vocabulary/words.py` — **all** English words/phrases/punctuation |
| Grammar | `grammar/english.py` (one `PhraseRule` per construct + auto-discovered `RULES`), `grammar/phrase_rule.py` (`PhraseRule`, `Ctx`, `select`), `grammar/selection.py`, `grammar/restriction.py`, `grammar/aggregation_kinds.py` |
| Assembly | `grammar/assembly/` — `base.Assembler`, `query`, `chains`, `clauses`, `restrictions`, `aggregation_value`, `instantiated`, `inference` |
| Planning | `grammar/planning/` — `base.Planner`, `query`, `clauses`, `instantiated`, `inference` |
| Conditions | `grammar/conditions/` — `verbalizer.ConditionVerbalizer`, `recognition`, `operator_phrase` |
| Microplanning | `microplanning/` — `referring`, `binding_scope`, `config`, `coordination` |
| Rendering | `rendering/` — `realization` (the passes), `coreference_processor`, `determiner_processor`, `morphology_processor`, `possessive`, `renderer`, `formatter`, `source_link_resolver` |
| Utilities | `chain_utils.py` (`walk_chain`, `build_path_parts`, `ChainAnalysis`), `morphology.py` (inflect facade), `subquery.py` |

---

## How to Add a New Output Format

Subclass {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter` (and optionally a renderer):

```python
from dataclasses import dataclass
from krrood.entity_query_language.verbalization.rendering.formatter import Formatter
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole

@dataclass
class MarkdownFormatter(Formatter):
    """Renders colour as Markdown bold (no true colour in plain Markdown)."""

    def colorize(self, text: str, role: SemanticRole) -> str:
        if role in (SemanticRole.KEYWORD, SemanticRole.VARIABLE):
            return f"**{text}**"
        return text

    @property
    def space(self) -> str:
        return " "

    @property
    def newline(self) -> str:
        return "\n"

    def wrap_link(self, text: str, url: str) -> str:
        return f"[{text}]({url})"
```

Then build a pipeline from it:

```python
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.renderer import ParagraphRenderer

text = VerbalizationPipeline(ParagraphRenderer(MarkdownFormatter())).verbalize(query)
```

---

## API Reference

### Entry points & engine

- {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline` — the single public entry point (`.plain()` / `.ansi()` / `.html()` + `.verbalize`)
- {py:func}`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression` — plain-text shortcut
- {py:class}`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer` — internal fragment builder (`build`)
- {py:func}`~krrood.entity_query_language.verbalization.engine.fold` / {py:class}`~krrood.entity_query_language.verbalization.engine.UnverbalizableExpressionError`
- {py:class}`~krrood.entity_query_language.verbalization.context.VerbalizationContext`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.features.Definiteness` / {py:class}`~krrood.entity_query_language.verbalization.fragments.features.Number`

### Grammar / dispatch

- {py:class}`~krrood.entity_query_language.verbalization.grammar.phrase_rule.PhraseRule` / {py:class}`~krrood.entity_query_language.verbalization.grammar.phrase_rule.Ctx` / {py:func}`~krrood.entity_query_language.verbalization.grammar.phrase_rule.select`
- {py:data}`~krrood.entity_query_language.verbalization.grammar.english.RULES`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.base.Assembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.planning.base.Planner`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.query.QueryAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.restrictions.RestrictionAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.aggregation_value.AggregationValueAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.assembly.chains.ChainAssembler`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.planning.inference.RuleStructure` / {py:class}`~krrood.entity_query_language.verbalization.grammar.planning.inference.ConditionPlan`
- {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase.comparator_operator`

### Fragment hierarchy

- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` / `WordFragment` / `RoleFragment` / `PhraseFragment` / `NounPhrase` / `BlockFragment` / `SubjectScope` / `PossessiveChain`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment` / {py:func}`~krrood.entity_query_language.verbalization.fragments.base.map_structural_children` / {py:func}`~krrood.entity_query_language.verbalization.fragments.base.oxford_and`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole` / {py:func}`~krrood.entity_query_language.verbalization.fragments.roles.role_for` / {py:class}`~krrood.entity_query_language.verbalization.fragments.source_ref.SourceRef`

### Rendering passes & output

- {py:func}`~krrood.entity_query_language.verbalization.rendering.realization.realize_tree`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor` / {py:class}`~krrood.entity_query_language.verbalization.rendering.determiner_processor.DeterminerProcessor` / {py:class}`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
- {py:mod}`~krrood.entity_query_language.verbalization.microplanning.possessive`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` / `HierarchicalRenderer`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` / `ANSIFormatter` / `HTMLFormatter`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver` / `AutoAPIResolver`

### Microplanning & utilities

- {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` / {py:class}`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope` / {py:class}`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfig`
- {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.fold_range_pairs` / {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.build_between`
- {py:func}`~krrood.entity_query_language.verbalization.chain_utils.walk_chain` / {py:func}`~krrood.entity_query_language.verbalization.chain_utils.build_path_parts` / {py:class}`~krrood.entity_query_language.verbalization.chain_utils.ChainAnalysis`

### Vocabulary (lexicon)

- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords` / `Logicals` / `Aggregations` / `Copulas` / `Operators` / `Articles` / `Prepositions` / `Conjunctions` / `Punctuation` / `Pronouns` / `ExistentialPhrase` / `FallbackNouns`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.words.PlainWord` / `RoleWord` / `OperatorPhrase` / `VocabEnum`
