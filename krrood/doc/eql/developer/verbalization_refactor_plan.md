# Verbalization — Architectural Review & Refactor Plan

> Status: **proposed** (design document; not yet implemented).
> Audience: maintainers of the EQL verbalization subsystem.
> Companion docs: {doc}`verbalization` (current internals), {doc}`../user/verbalization` (user guide).

This document is a from-scratch architectural review of the verbalization subsystem
(`krrood/src/krrood/entity_query_language/verbalization`, ~10.7 kLOC source + ~7.2 kLOC tests)
and a multi-phased, test-driven plan to refactor it for cohesion, SOLID compliance, and
extensibility — without changing the English it produces.

It is written from two perspectives: SOLID software design, and Natural Language Generation (NLG)
/ computational linguistics.

---

## 1. What the system is (and what is good)

The subsystem is a real, literature-grounded NLG pipeline that already maps onto the
Reiter & Dale three-stage model:

- **Content planning** — `Planner` classes produce frozen *plans* ("what to say").
- **Microplanning** — referring expressions, aggregation/coordination, lexicalization.
- **Surface realization** — `Assembler` classes build a `Fragment` IR; ordered *passes*
  (coreference → determiner → morphology → orthography) lower it; renderers + formatters emit
  plain / ANSI / HTML.

Dispatch is a Montague rule-to-rule grammar: `engine.fold` is an F-algebra catamorphism over the
EQL tree, and `select` picks the most-specific `PhraseRule`. The `Fragment` IR is a clean
catamorphism + map (`fold_fragment` / `map_fragment`). Referring-expression generation and
centering-theory pronominalization are genuinely well done.

**The problems are organizational, not conceptual.** The right ideas are implemented two or three
times, across files named for incidental groupings rather than concerns, with ~95 free functions
threaded between them.

---

## 2. Findings

### 2.1 Two parallel specificity-dispatch systems (Critical — DRY/SRP)

`PhraseRule.select` (over EQL nodes; key = construct-MRO depth, guarded-over-unguarded, rule-MRO
depth) and `SpecificityRule.most_applicable` (over an argument tuple; key = MRO depth) are the same
idea — *"most-specific guarded alternative wins; ties are forbidden"* — implemented twice. The base
`SpecificityRule` is defined once, but is subclassed by **four** families:

| Family | Location | Concern |
|---|---|---|
| `ConditionForm` | `grammar/conditions/forms.py` | how a condition attaches to a subject (slot) |
| `PredicateTransform` | `grammar/conditions/transforms.py` | how a comparator is said as a predicate |
| `RestrictionSubjectRule` | `grammar/conditions/restriction.py` | which variable the WHERE restricts |
| `RankingForm` | `grammar/query/ranking.py` | the ranking surface (`top three` / `with the highest`) |

### 2.2 The `conditions/` package has lost cohesion (Critical)

One construct (the comparator/condition) is realized across **seven** files and **three**
`SpecificityRule` registries plus a `PhraseRule` set:

| File | Holds | Pattern |
|---|---|---|
| `rules.py` | `ComparatorRule`, `And/Or/Not…`, `Range/CoindexedFoldRule` | `PhraseRule` (entry) |
| `assembler.py` | `ConditionAssembler` (predicate/modifier surfaces) | `Assembler` |
| `transforms.py` | `PredicateTransform` family + `render_absence` | `SpecificityRule` |
| `forms.py` | `ConditionForm` family + `Slot`/`Placement` + `as_subject_restrictions` | `SpecificityRule` |
| `restriction.py` | `RestrictionSubjectRule` family | `SpecificityRule` |
| `recognition.py` | 10 structural recognizers | free functions |
| `operator_phrase.py` | `comparator_operator`, `coindexed_operator` | free functions |

`transforms.py` is misnamed (it is *predication*, not "transforms"). `restriction.py` and `rules.py`
both "contain rules" but of different kinds — nothing signals that to a reader.

### 2.3 Folding / simplification / grouping is scattered, not a pass (Critical)

"Aggregation" (the NLG term for combining clauses) is one microplanning task, fragmented across
four locations:

- range fold + co-indexed fold — `microplanning/coordination.py`, triggered ad-hoc from **two**
  condition call sites (`ConditionAssembler.as_statements`, `forms.as_subject_restrictions`);
- co-owned genitive **selection** folding — buried in `QueryAssembler._folded_selections`;
- match *"…respectively"* grouping — buried in `MatchAssembler`;
- aggregate-subquery collapsing — in `subquery.py`.

`RangeFold` / `CoindexedFold` are already first-class verbalizable artifacts; the others are not.
There is no single simplification stage and adding a fold strategy edits an assembler (not
open/closed).

### 2.4 Rendering passes have no shared abstraction (Critical)

Three different tree-walks: `CoreferenceProcessor` hand-rolls `_walk`/`_dispatch`;
determiner/morphology use `map_fragment`; orthography uses `map_structural_children`. Pass order is
hardcoded inline in `realize_tree` (not a first-class pipeline). There is no `RealizationPass`
base, so adding/reordering a pass is not open/closed and pass ordering/state-handoff is untested.

### 2.5 `QueryAssembler` is a god-object (Critical)

689 lines, ~37 methods spanning entity selectors, set-of, aggregation/grouped/ordered reports,
rankings, nested + inline noun forms, and selection folding. Single Responsibility is gone.

### 2.6 EQL-structure helpers live in verbalization (Major — layering + import cycle)

`core/expression_structure.py` already exists, and its own docstring states structural queries
belong in core, "usable by any consumer (evaluation, optimization, verbalization …)". Yet:

- `verbalization/subquery.py` holds **7** pure-structure functions (no English):
  `unwrap_result_quantifiers`, `selected_aggregator`, `is_aggregation_subquery`,
  `is_calculation_value`, `is_collapsible_aggregation_subquery`, `aggregation_leaf_attribute`,
  `aggregation_source_root`.
- `conditions/recognition.py` holds more structural predicates: `single_hop_attribute`,
  `is_none_literal`, `references`, `attribute_names`, `is_boolean_attribute_chain`,
  `superlative_aggregation`, `is_concrete_object_literal`.

This also creates a **`conditions ↔ chain` import cycle**: `conditions/{rules,transforms}.py`
import `chain`, while `chain/planner.py` imports `conditions.recognition.relational_verb` (a
morphological recognizer that belongs in a shared layer).

### 2.7 AGENTS.md violations (Major)

- **Global mutable state**: `morphology.py` module-level dicts `_plural_overrides`,
  `_article_overrides` (forces `clear_overrides()` test coupling). Forbidden by "avoid global
  variables".
- **`getattr`**: 35 calls across 15 files. Forbidden by "always access attributes via `.`, never
  via `getattr`".
- **Attribute-access `try/except`**: e.g. `recognition.references` catches `AttributeError`;
  `operator_phrase` catches `KeyError`. Forbidden by "do not wrap attribute access in try-except;
  programs in illegal states should raise appropriate exceptions".

### 2.8 Doctest harness cannot see most modules (Major — blocks the doctest goal)

`test_rule_doctests._MODULES` is a hand-curated list of 17 modules. Doctests added to `forms.py`,
`transforms.py`, `match/*`, `inference/*`, all of `microplanning/*`, `rendering/*`, `fragments/*`,
`vocabulary/*` would **silently not run**.

### 2.9 Minor

- Dead code: `coordination.fragment_for_folded_conjunct` (no callers), likely `coordination.has_pair`.
- ~Several single-call-site free helpers that should be private methods (`_subject_id`, `_quality`,
  `_join_residual`, `_relation_target`, `_boolean_constraint`, `place`).
- Inconsistent assembler conventions: `chain` exposes a public surface-form toolkit; `match` /
  `inference` hide everything behind `realize` — no documented rule for which.
- `Operators` enum boilerplate (9 members × 8 fields); tolerable, table-driven would be cleaner.

### 2.10 Docs

The two `verbalization.md` files are accurate **today** but tightly coupled to the present
structure (they name `subquery.py`, `transforms.py`, the four registries, a module map). They are
not stale yet, but **will** be the moment the code changes — so doc updates are in-scope for every
phase, not an afterthought.

---

## 3. Linguistic / NLG assessment

- The pipeline shape (content → microplanning → realization) is correct and worth preserving.
- **Aggregation is mis-modularized.** In NLG, aggregation (shared-owner genitives, range merging,
  co-indexed "have the same", "…respectively") is one microplanning subtask. Making it one pass is
  both SOLID hygiene and truer to the theory.
- **Naming should follow linguistic concepts**: `transforms.py` → *predication*; `Slot`/`Placement`
  is topological-field / surface-slot assignment, and `RankingSurface` is the same concept (a
  microplanned constituent with a target slot + agreement features) — unify them.
- **Lexicalization boundary is blurred** between recognition (`relational_verb`, `is_past_participle`)
  and morphology/lexicon. A crisp morphology/lexicon layer (no EQL imports) vs. structure-recognition
  layer (in core, no English) sharpens both.

---

## 4. Target architecture

Six unifying moves; each independently valuable and testable.

- **A. One dispatch abstraction.** A single `GuardedAlternative` (`applies(*args) -> bool` + a
  specificity key) and one `select`. `PhraseRule` becomes the specialization where `applies =
  isinstance(node, construct) and when(...)`; the four `SpecificityRule` families become registries
  of the same base. One mechanism; the ties-forbidden contract stated once.
- **B. Per-construct cohesion + stepdown order.** Keep per-construct folders, but make each read
  top-down (entry `PhraseRule` → `Assembler.realize` → public surface methods → private helpers; no
  recognizers — those move out). Collapse `conditions/` into clearly-named concern units:
  `predication.py` (was `transforms` + `operator_phrase` + `ConditionAssembler`), `placement.py`
  (was `forms`), `subject.py` (was `restriction`).
- **C. One simplification/coordination pass.** A `Simplification` registry (range fold, coindexed
  fold, co-owned genitive selection fold, match-respectively grouping, aggregate-subquery collapse),
  each a `GuardedAlternative` that recognizes → reduces → yields a first-class verbalizable artifact.
  A single entry applies them wherever sibling conjuncts/selections form.
- **D. Realization passes as a pipeline.** A `RealizationPass` ABC (`process(fragment) -> Fragment`),
  a `PASSES` list iterated by `realize_tree`, and a shared `RewritePass` built on `map_fragment`
  (stateless passes stop re-walking by hand; coreference stays a stateful traversal variant).
- **E. Structure-recognition in core; morphology/lexicon as a clean layer.** Move pure structural
  predicates (`subquery.py` + structural half of `recognition.py`) into `core/expression_structure.py`
  (or sibling). Move `relational_verb`/participle recognition to a morphology/recognition layer that
  both `chain` and `conditions` depend on — breaking the cycle.
- **F. Auto-discovery + zero global state.** Auto-discover `RULES` (and all doctest-bearing modules)
  by walking the package. Replace `morphology`'s module dicts with an injected `Morphology` object.
  Eliminate `getattr` / attribute-`try/except` via typed accessors on the EQL base.

---

## 5. Multi-phased TDD plan

Principles: every phase ends green; tests are added *before* the change; never weaken a test to make
it pass (only update import paths when a symbol moves). Blast radius is the ~120 white-box tests
(planners, processors, fragments, registries, `test_restriction_placement`, `test_microplanning`);
the ~300+ black-box string tests are the safety net.

### Phase 0 — Safety net & tooling (no behavior change)
- Establish a runnable env (install workspace + test deps).
- Auto-discover doctests: replace `_MODULES` with a package walk; assert every verbalization module
  is scanned.
- Characterization tests for under-covered areas: deep mixed nesting (Entity+Aggregation+Inference+
  Match), folding combinations (3+ co-owned, overlapping/again ranges, nested AND-in-OR, NOT-of-NOT),
  error paths (`Unverbalizable…`), and full pass-pipeline order/state handoff.
- Guard tests (initially baseline-counted / xfail): "no `getattr` in verbalization", "no
  attribute-access `try/except`", "no module-level mutable dict".

### Phase 1 — Relocate EQL-structure to core (low risk; breaks the cycle)
Move `subquery.py` + structural recognizers into `core/expression_structure.py`; move
`relational_verb`/participle recognition to a shared morphology/recognition home; re-point imports;
add an import-graph test proving the `conditions ↔ chain` cycle is gone. Add core unit tests +
doctests for moved functions.

### Phase 2 — Unify the specificity dispatch
Introduce `GuardedAlternative` + `select`; re-express `PhraseRule.select` and the four families on
top of it. Dispatcher unit tests (subsumption wins; disjoint guards; ties forbidden). Behavior
identical.

### Phase 3 — One simplification/coordination pass
Define the `Simplification` registry; migrate range + coindexed folds (parity tests from Phase 0);
then migrate selection genitive-folding (out of `QueryAssembler`) and match-respectively grouping
(out of `MatchAssembler`) into it as first-class artifacts. Add generality tests. Delete dead
`fragment_for_folded_conjunct` / `has_pair`.

### Phase 4 — Make `conditions/` cohesive (headline ask)
Reorganize into `predication.py` / `placement.py` / `subject.py` with strict stepdown order; merge
`operator_phrase` + `ConditionAssembler` + `PredicateTransform`. Output identical; update only the
*import paths* in white-box tests, never their assertions. Update the developer doc module map in the
same commit.

### Phase 5 — Realization passes as a pipeline
Introduce `RealizationPass` + `PASSES` + `RewritePass`; refactor the four processors onto it; the
Phase-0 pipeline-order test guards it. Behavior identical.

### Phase 6 — Decompose `QueryAssembler`
Extract `ReportAssembler` / `RankedSetOfAssembler` / `SelectionAssembler` (selection folding already
moved in Phase 3), leaving `QueryAssembler` a thin shape-dispatcher. Add doctests per extracted unit.

### Phase 7 — Eliminate AGENTS.md violations & global state
Flip the Phase-0 guard tests to enforcing. Replace `getattr` / `try-except` with typed accessors;
convert `morphology` to an injected object (tests build a fresh instance instead of `clear_overrides`).

### Phase 8 — Doctest saturation + doc sync
Add doctests across assemblers/planners/recognizers/folds/passes/fragment helpers/vocabulary (now
auto-run). Rewrite both `verbalization.md` files to the new structure; remove references to
`subquery.py` / `transforms.py`. Add a doctest-coverage test (every public assembler method has an
executable example).

**Sequencing rationale:** 0 builds the net; 1–2 are mechanical and de-risk everything before the
visible reorganizations; 3–4 deliver the cohesion; 5–6 tackle the remaining god-objects; 7–8 enforce
the rules and lock in comprehension. Each phase is a small, green, reviewable unit.

---

## 6. Progress log

### Phase 0 — done

- **Runnable environment.** The workspace needs Python **3.12** (a sibling package, `giskardpy`,
  uses 3.12-only nested f-strings that fail to import under 3.11) and the system `graphviz`/
  `libgraphviz-dev` headers (for `pygraphviz`). With those, `uv sync --extra dev --python 3.12`
  produces a `.venv` that runs the suite: `.venv/bin/python -m pytest
  test/krrood_test/test_eql/test_verbalization`.
- **Doctest auto-discovery** (`test_rule_doctests.py`). Replaced the hand-maintained 17-module list
  with a package walk, so a doctest added *anywhere* in the verbalization package is executed. A new
  assertion locks in that the previously-curated modules stay covered and that
  `grammar.conditions.forms` (silently dropped before) is now included.
  - *This immediately paid off*: it surfaced a **stale, never-executed doctest** in
    `conditions/forms.py::AbsenceForm`, whose example used a *relational* attribute (`assigned_to`)
    and so produced *"has not been assigned to any Robot"*, not the *"has no <attribute>"* form its
    own prose describes. Fixed the example to a plain-noun attribute (`priority`) so it is faithful.
- **Ratchet guardrails** (`test_code_quality_guardrails.py`). AST-based, fail only on *new*
  violations, with baselines recorded as the debt inventory (driven to 0 in Phase 7):
  `getattr` calls = **37**, attribute-access `except` handlers = **3**, module-level mutable
  containers = **2**.
- **Characterization tests** (`test_characterization_coverage.py`). Black-box golden strings for the
  under-covered areas (3-way genitive folding, range fold beside an unrelated conjunct, AND-in-OR,
  double negation — recorded as *not* simplified today, superlative-in-WHERE collapse, and
  plural-subject agreement across the whole pipeline).

### Phase 1 — done

- The 7 `verbalization/subquery.py` functions moved to a new **`query/aggregation_structure.py`**
  (query layer, clean global imports; `core/` would have forced lazy imports to dodge a
  `query→core→query` cycle). `subquery.py` deleted; all importers re-pointed.
- The relational-attribute recognizer (`RelationVerb` / `relational_verb` / `relational_verb_phrase`)
  moved out of `conditions/recognition.py` into **`verbalization/relational_attributes.py`** (above
  the lexicon, below the grammar). This removed the only `chain → conditions` import, **breaking the
  `conditions ↔ chain` package cycle** (verified by a test + grep). `navigation_path` keeps a
  TYPE_CHECKING-only reference to `RelationVerb`, so no runtime cycle.
- New `test_aggregation_structure.py` + doctests on the relocated recognizer. 568 passed, 7 skipped.

### Phase 2 — done

- Extracted one **`concrete_subclasses(base)`** primitive (`framework/specificity.py`), now used by
  *both* `RULES` and `SpecificityRule.alternatives` — the subclass-discovery logic was duplicated.
- **`registry.py` now auto-discovers** the construct `rules` modules by walking the `grammar`
  package, replacing the hand-maintained 8-import list (answering "is the registry needed?" — yes,
  but it should not hand-list imports). Adding a `grammar/<construct>/rules.py` now needs no registry
  edit. New `test_grammar_registry.py` asserts `RULES` is exactly one instance per concrete rule and
  that every construct folder is reached.
- The `PhraseRule` selection (`select`) and the `SpecificityRule` families both route through the
  same `most_specific` / `mro_depth` primitives in `specificity.py`; the contract now lives in one
  place. (A deeper merge of the two into a single class hierarchy was considered and rejected — they
  differ legitimately: instance-vs-class, `isinstance`-construct-gate vs pure `applies` guard — so
  forcing one base would be a leaky abstraction. They share the selection *primitive*, not the shape.)

### Phase 3 — done (EQL-level folding)

- The EQL-level conjunct reduction is now a cohesive, class-based pass: **`ConjunctReducer`** applies
  an ordered registry of **`ConjunctFold`** strategies (`RangeBoundFold`, `CoindexedComparisonFold`)
  — adding a fold is a new strategy, nothing else changes (open/closed). `reduce_conjuncts` is the
  thin function entry the two callers use; the fold algorithms and recognizers are preserved as the
  (test-facing) implementation, and the file now reads top-down: artifacts → pass → algorithms →
  recognizers → fragment builders.
- Removed the dead `fragment_for_folded_conjunct` (no callers). `has_pair` is kept (it has tests).
- New `test_conjunct_reduction.py` adds generality cases (reverse-order bounds, distinct chains, a
  third unpaired bound, empty/lone/ non-comparator inputs). 572 passed, 7 skipped.
### Phase 3b — done (owner-based folding: set_of selections + match grouping)

Folding/grouping was also happening on `set_of` selectables (a co-owned genitive run) and in
match/`underspecified` assembly (the *"…respectively"* grouping by object) — both *owner-based*
aggregation. These now share one mechanism in the coordination home:

- new **`OwnerGroup`** + **`group_by_owner`** (all same-owner items together, first-seen order) +
  **`group_consecutive_by_owner`** (maximal adjacent same-owner runs → an `OwnerGroup`, others pass
  through — the order-preserving analogue of `reduce_conjuncts`) in `coordination.py`.
- `MatchPlanner.plan` now groups its construction-pattern equalities via `group_by_owner` (its
  bespoke dict loop is gone). The grouping stays *content determination* in the construct planner —
  the right NLG home — but uses the shared primitive.
- `QueryAssembler._folded_selections` now folds via `group_consecutive_by_owner`; the bespoke
  `_co_owned_run` walker is deleted (`_foldable_attribute` remains as the classifier callable).
- New `test_coordination_grouping.py`. Suite green.

**The complete folding/aggregation map** (all of it now reachable from `coordination.py`):

| What folds | Mechanism (in `coordination.py`) | Recognizer / classifier | Used by |
|---|---|---|---|
| complementary `>=`/`<=` bounds on one chain | `RangeBoundFold` → `RangeFold` | `coordination` (internal) | conditions (`reduce_conjuncts`) |
| co-indexed `a.X⊙b.X` across two prefixes | `CoindexedComparisonFold` → `CoindexedFold` | `coordination` (internal) | conditions (`reduce_conjuncts`) |
| a position's x/y/z (match construction) | `group_by_owner` → `OwnerGroup` | `MatchPlanner._as_assignment` | match / underspecified |
| co-owned attributes in a `set_of` tuple | `group_consecutive_by_owner` → `OwnerGroup` | `QueryAssembler._foldable_attribute` | query (set_of selection) |

Each construct supplies *which* items and *how to classify* them (content determination, per
construct); the *grouping/folding mechanism* lives once, in `coordination.py`.

The fragment **builders** that render the folded results (`build_between`, `coordinated_genitive`,
`oxford_comma`, the match *"respectively"* phrase) are a separate, later stage and stay with the
fragment layer / their construct assembler — they assemble already-rendered pieces and need the
recursion, so they are deliberately not in the (pure) coordination mechanism.
