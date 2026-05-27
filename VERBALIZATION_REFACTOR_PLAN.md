# EQL Verbalization Refactor — One-System, Single-Source-of-Truth

## Context

The EQL verbalization subsystem (`krrood/src/krrood/entity_query_language/verbalization/`)
turns query expression trees into English. It currently runs on **two parallel
dispatch systems** with no boundary between them:

1. **Rules** (`rules/*.py`): declarative `applies()` + `transform()`, MRO-sorted by
   `RuleEngine`. Good Open/Closed design.
2. **Sub-verbalizers** (`EntityVerbalizer` 645 lines, `ChainVerbalizer`,
   `RuleVerbalizer`): imperative god-classes full of private `_methods_`.

Rules relate to verbalizers *inconsistently*: some rules hold all their logic inline
(`AndRule`, `ComparatorRule`, `GroupedByRule`), while others are thin shells that just
call a verbalizer method (`EntityRule`→`_entity.verbalize_query`,
`MappedVariableRule`→`_chain.verbalize_mapped`, `NotBoolAttrRule`, `PronominalChainRule`,
`InstantiatedVariableRule`). There is **no principle** for whether new behavior is a rule
or a verbalizer method, so contributors don't know where to add things — the stated pain
point. Rules also reach into verbalizer internals (`delegate._chain.verbalize_possessive`),
coupling every rule to the verbalizer's concrete shape.

**Goal:** collapse to a single system — **one rule is the single source of truth for one
expression type.** Rules only need a `build(expr, ctx)` callback for recursion. The
sub-verbalizers are absorbed into rules and deleted. Structural facts that EQL already
knows (or should own) move to EQL core. Output stays **byte-identical** — the ~215
verbalization tests are the regression lock.

Decisions confirmed with the user:
- **Consolidation:** one rule per expression type; delete `EntityVerbalizer` /
  `ChainVerbalizer` / `RuleVerbalizer`; shrink the delegate to `build`/`verbalize` only.
- **EQL core:** allowed to move structural facts down into core and reuse existing core
  capabilities.
- **Behavior:** byte-identical output; do not change any verbalized string.

The fragment / renderer / formatter layers (`fragments/`, `rendering/`, `vocabulary/`)
are clean and **out of scope** — leave them as-is.

---

## Target architecture (the single mental model)

| Component | Role after refactor |
|---|---|
| `VerbalizationRule` (`rule_engine.py`) | The **only** extension unit. `applies()` precondition + `transform()` rendering. |
| `RuleEngine` (`rule_engine.py`) | First-match dispatch, MRO-sorted. Unchanged. |
| `ALL_RULES` (`rules/registry.py`) | The **one** place to register a new rule. |
| `Verbalizer` protocol + `EQLVerbalizer` (`verbalizer.py`) | The recursion callback. Exposes only `build(expr, ctx) -> VerbFragment` and `verbalize(expr, ctx) -> str`. **No** `_chain` / `_entity` / `_rule` fields. |
| `VerbalizationContext` (`context.py`) | Per-pass state only (coreference, constraints, depth, subjects). Gains a `seen_reference` helper; loses the structural `flatten_same_type` algorithm. |
| `fragments/`, `rendering/`, `vocabulary/` | Untouched. |
| EQL core (`core/`, `query/`, `operators/`) | Owns structural facts: chain access path, selection aggregator split, operand flattening, sub-query classification, temporality. |

**Extension story after refactor (what we want a contributor to read):** "To verbalize a
new expression type or change a phrasing, write one `VerbalizationRule` in `rules/`, use
`delegate.build(child, ctx)` for sub-expressions, and add it to `ALL_RULES`. That's it."

---

## Changes by concern

### A. Narrow the delegate (the linchpin)

- Define a `Verbalizer` `Protocol` (in `verbalizer.py` or `rule_engine.py`) with just
  `build(expr, ctx) -> VerbFragment` and `verbalize(expr, ctx) -> str`.
- `transform(cls, expr, ctx, delegate)` keeps its signature, but `delegate` is now typed as
  `Verbalizer`. Rules must stop accessing `delegate._chain` / `delegate._entity` /
  `delegate._rule`. Every such access is removed as its target logic moves into a rule.
- `EQLVerbalizer.__post_init__` loses the three sub-verbalizer fields; it keeps only
  `_engine = RuleEngine(ALL_RULES)`.

### B. Absorb `ChainVerbalizer` into the chain rules (`rules/chains.py`)

- Move `_verbalize_mapped_chain_`, `_render_path_`, `_render_possessive_path_`,
  `_verbalize_chain_root_`, `_verbalize_navigation_chain_`, `_verbalize_bool_attribute_chain_`
  into module-level helpers in `rules/chains.py` (they take `build`/`ctx`, not a delegate).
- `MappedVariableRule` / `PronominalChainRule` / `FlatVariableRule` call those helpers
  directly. `NotBoolAttrRule` (in `rules/logical.py`) calls the negated bool-attr helper
  directly instead of `delegate._chain.verbalize_mapped_negated`.
- The `as_inline_noun` callback that `ChainVerbalizer` received via `entity_inline_fn`
  becomes a normal call into the query-rule helper (see C) — both are rule helpers now, so a
  shared import replaces the injected callable.
- Delete `chain_verbalizer.py`.

### C. Absorb `EntityVerbalizer` into the query rules (`rules/query.py`, new `rules/inference_rule.py`)

- Move the query-body assembly (`_verbalize_query_body_`, `_where_clause`,
  `_grouped_by_clause`, `_having_clause`, `_ordered_by_clause`), the noun forms
  (`as_noun`, `as_inline_noun`), the aggregation-value collapse
  (`_verbalize_aggregation_value_`, `_aggregation_scope_`), and the subject-restriction
  orchestration into `rules/query.py` as module-level helpers used by `EntityRule` /
  `SetOfRule`.
- `EntityRule.transform` keeps its top-level-vs-nested branch on `ctx.query_depth` (that is
  a legitimate precondition on runtime state, not removable glue) but each branch is a clear
  one-line call to a named helper.
- **Convert the buried inference-rule glue into a declarative rule.** Today
  `EntityVerbalizer.verbalize_query` contains `if self._d._rule.can_handle(expr): ...`. Replace
  with a dedicated `InferenceRuleRule(EntityRule)` whose `applies()` is exactly the
  `RuleAnalyzer.can_handle` precondition (Entity whose selected variable is an
  `InstantiatedVariable`). MRO-depth ordering makes it shadow `EntityRule` automatically —
  no `if` in the generic path. This directly matches the project's declarative-rules
  preference.
- Move `RuleVerbalizer` + `RuleAnalyzer` into `rules/inference_rule.py` largely as-is (the
  IF/THEN analyzer is genuinely cohesive); only the invocation path changes (it's now reached
  by the rule engine, not by `EntityVerbalizer`). `RuleAnalyzer`/`RuleStructure`/
  `AntecedentInfo`/`ConsequentBinding` can stay in `rule_analysis.py` and be imported by the
  rule.
- Delete `entity_verbalizer.py` and `rule_verbalizer.py`.

### D. Kill the concrete duplication

1. **OrderedBy / GroupedBy rendered twice.** `OrderedByRule.transform` and the entity
   `_ordered_by_clause` are near-verbatim; `GroupedByRule` is a simpler copy of
   `_grouped_by_clause`. Make **one** helper per clause (the richer entity version, which
   already branches internally on "aggregated selections and not SetOf" — keep that
   precondition). `OrderedByRule` / `GroupedByRule` (bare-node dispatch) and the query-body
   assembly both call the same helper. One definition each.
2. **Operator-phrase selection duplicated 5×** (`ComparatorRule`, `CalculationEqualityRule`,
   `NotComparatorRule`, `NotCalculationEqualityRule`, `restriction.py::_predicate_op_frag`).
   Extract one helper `comparator_phrase(comparator, ctx, build, *, negated=False)` that:
   detects calc-equality and temporality, selects the operator via
   `Operators.from_callable(...).select(negated=, compact=ctx.compact_predicates, temporal=)`,
   and assembles `left op right`. All five sites call it. This folds the `Calc*` and `Not*`
   comparator variants into flags: `ComparatorRule` handles affirmative (calc-eq detected
   inside the helper), and the `Not(Comparator)` case calls the helper with `negated=True`.
   Reduces four comparator rules toward one rule + one negation entry point while keeping
   output identical.
3. **"Already seen → the TypeName" early-return** copy-pasted in `verbalize_query`,
   `verbalize_nested`, `as_noun`, `as_inline_noun`, `_verbalize_instantiated_natural`. Add
   `VerbalizationContext.seen_reference(expr) -> Optional[VerbFragment]` returning the
   `"the <label>"` phrase when `expr._id_ in self.seen`, else `None`. All sites use it.
4. **`_word` / `_phrase` / `_role` one-liners** redefined in ~6 files. Add factory
   classmethods to the fragment classes (e.g. `PhraseFragment.of(*parts, sep=" ")`,
   `RoleFragment.plain(text, role, ref=None)`) or a single `fragments/factory.py`, and delete
   the per-file copies. Prefer the existing `Vocabulary.*.as_fragment()` for fixed words.

### E. Push structural facts into EQL core (remove duplicate representation)

Verified duplications/relocations:

1. **`walk_chain` / `chain_root` duplicate `MappedVariable._access_path_`**
   (`core/mapped_variable.py:177`). Add a `_chain_root_` property on `MappedVariable` (the
   non-MappedVariable child at the end of `_access_path_`) if not already derivable, then
   replace `walk_chain(expr)` usages with `(expr._access_path_, expr._chain_root_)` and
   `chain_root(expr)` with `expr._chain_root_` (guarding the non-chain case). Remove
   `walk_chain`/`chain_root` from `chain_utils.py`.
2. **`_aggregated_expressions_` overlaps `Query._aggregators_and_non_aggregators_in_selection_`**
   (`query/query.py:501`). Reuse the existing core split and filter by group-key roots, rather
   than re-classifying selected variables in the verbalizer.
3. **`flatten_same_type`** (currently a method on `VerbalizationContext`) is a generic
   operand-flatten algorithm. Move to EQL core as a structural utility (e.g. a function in
   core operating on a binary-operator type, or a method on the operator base). Verbalization
   calls the core version. Keep a thin `ctx`-free call site.
4. **`subquery.py` classifiers** (`selected_aggregator`, `is_aggregation_subquery`,
   `is_constrained_query`, `is_collapsible_aggregation_subquery`, `aggregation_leaf_attribute`,
   `aggregation_source_root`) are structural questions about an `Entity`/`Query`. Move the
   genuinely structural ones onto `Entity` / `Query` / `Aggregator` as properties/methods in
   core. Keep `is_calculation_value` where it makes sense (it asks "is this an aggregator
   value"). English-specific glue stays in verbalization.
5. **`is_temporal`** (currently `ChainVerbalizer.is_temporal`) is a type fact — base it on the
   expression's `_type_` and expose it in core (e.g. a small helper or property), since
   comparator rules and restriction both need it.

Guard rails for core moves: only relocate things that are structural facts about
expressions and have ≥2 consumers or already overlap core. Match core naming conventions
(`_dunder_` style on expression classes). Do **not** move English/wording logic
(`build_path_parts`, `verbalize_plural`, `range_fold`, restriction rendering) into core —
those stay in verbalization.

### F. File-layout outcome

- **Deleted:** `entity_verbalizer.py`, `chain_verbalizer.py`, `rule_verbalizer.py`.
- **`rules/`** becomes the heart: `chains.py` (full chain rendering), `query.py`
  (Entity/SetOf/GroupedBy/OrderedBy), `inference_rule.py` (IF/THEN), plus existing
  `logical.py`, `comparator.py`, `aggregators.py`, `quantifiers.py`, `variables.py`,
  `registry.py`. Shared rule helpers in `rules/_fragments.py` (or fragment classmethods) and
  `rules/_comparator.py` (the `comparator_phrase` helper).
- **`chain_utils.py`:** `walk_chain`/`chain_root` removed; `build_path_parts` /
  `verbalize_plural` stay (verbalization-specific) — optionally moved next to `rules/chains.py`.
- **`context.py`:** gains `seen_reference`; loses `flatten_same_type`.
- **`restriction.py`, `range_fold.py`:** stay (already clean declarative mini-systems);
  updated to call `comparator_phrase` and core classifiers.
- **`subquery.py`:** shrinks to whatever English-specific helpers remain after E.4.

---

## Migration sequence (incremental, each step keeps tests green)

Run `pytest test/krrood_test/test_eql/test_verbalization` after **every** step — byte-identical
output is the contract, so green tests = correct refactor.

0. **Baseline.** Run the verbalization tests; confirm green. (Activate env: `workon cram2`.)
1. **Extract shared helpers, no behavior change.** Add `comparator_phrase`,
   `ctx.seen_reference`, fragment factories. Rewire existing rules/verbalizers to call them.
   Remove the 5 operator copies, 5 seen-copies, 6 word/phrase/role copies.
2. **Move structural facts to EQL core** (E.1–E.5). Replace `walk_chain`, the aggregator
   re-classification, `flatten_same_type`, sub-query classifiers, and `is_temporal` with core
   reuse. Verbalization imports the new core locations.
3. **Absorb `ChainVerbalizer`** into `rules/chains.py`; delete it; remove `delegate._chain`.
4. **Absorb `EntityVerbalizer`** into `rules/query.py`; add `InferenceRuleRule` and move the
   IF/THEN cluster to `rules/inference_rule.py`; delete `entity_verbalizer.py` /
   `rule_verbalizer.py`; remove `delegate._entity` / `delegate._rule`. Convert the
   `can_handle` glue into the new rule's `applies()`.
5. **Collapse comparator rule family** using `comparator_phrase` (fold `Calc*` / `Not*`
   variants into flags where it stays byte-identical).
6. **Docs + memory.** Rewrite `krrood/doc/eql/developer/verbalization.md` to describe the
   single-system model (drop the "specialized verbalizers" section; the extension story is
   now "write one rule, register it"). Update the auto-memory note
   `project_eql_verbalization_phase2.md` to reflect the consolidated architecture.

Keep each step a separate commit so any output regression is bisectable.

---

## Critical files

- `verbalization/verbalizer.py` — shrink `EQLVerbalizer`, add `Verbalizer` protocol.
- `verbalization/rule_engine.py` — unchanged dispatch; `delegate` retyped to `Verbalizer`.
- `verbalization/rules/registry.py` — register `InferenceRuleRule`; reorder as needed.
- `verbalization/rules/chains.py` — absorbs `chain_verbalizer.py`.
- `verbalization/rules/query.py` — absorbs `entity_verbalizer.py` body/noun/clause logic.
- `verbalization/rules/inference_rule.py` (new) — absorbs `rule_verbalizer.py`.
- `verbalization/rules/comparator.py` + `rules/logical.py` — use `comparator_phrase`.
- `verbalization/context.py` — `seen_reference`; drop `flatten_same_type`.
- `verbalization/restriction.py` — call `comparator_phrase` + core classifiers.
- `core/mapped_variable.py` — `_access_path_` reuse (+ `_chain_root_` if needed).
- `query/query.py` — reuse `_aggregators_and_non_aggregators_in_selection_`.
- EQL core (operators / base) — new homes for `flatten_same_type`, sub-query classifiers,
  `is_temporal`.
- `krrood/doc/eql/developer/verbalization.md` — rewrite to one-system model.

---

## Verification

1. **Regression lock:** `pytest test/krrood_test/test_eql/test_verbalization` (both
   `test_verbalization.py` ~1576 lines and `test_verbalization_rendering.py` ~724 lines) must
   stay 100% green through every step. These assert exact strings, so they *are* the spec.
2. **Whole-suite check:** `pytest test/krrood_test` to catch any breakage from the EQL-core
   moves (other subsystems may consume the relocated helpers).
3. **Manual spot-check:** run `verbalize_expression(query)` on a handful of representative
   shapes (a `SetOf` with `grouped_by` + `having`, an aggregation sub-query, a bool-attr
   negation, a deep possessive chain, an IF/THEN inference rule) and diff against current
   output (capture current strings before starting, in step 0).
4. **Extensibility smoke test:** after the refactor, add a throwaway dummy rule for a fake
   expression type in <10 lines + one registry entry, confirm it dispatches, then remove it —
   proves the "one place to extend" goal.

---

## Risks / notes

- **Byte-identical is strict.** The grouped-by clause and the inference-rule path have
  context-dependent shapes; preserve their existing internal preconditions verbatim — the aim
  is removing *duplication and dual-dispatch*, not removing every legitimate `if`.
- **EQL-core moves can affect other consumers.** `_access_path_` and the aggregator split are
  already used by evaluation code; reusing them is safe, but new helpers added to core must
  follow core conventions and be covered by the full `test/krrood_test` run.
- **`InferenceRuleRule` ordering.** It must out-rank `EntityRule` via MRO depth (subclass of
  `EntityRule`) so the engine tries it first; verify with the extensibility smoke test.
- Pre-commit runs **black**; format touched files before committing.
