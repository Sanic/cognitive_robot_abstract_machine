# Warsaw SDT evaluation

This package holds what a run is judged and looked at by:

| Module | What it is |
| --- | --- |
| `graph.py` | `EvaluationGraph`: a run's bodies, their classes, and every mount it attempted, accepted or refused, independent of the database |
| `size.py` | The size of a body, which the graph records |
| `export_world_mesh.py` | A finished run's world as a GLB, one named node per body, for Blender |
| `label_vocabulary.py` | Matching our class names to another dataset's label vocabulary (habitat comparison, lexical baselines) |

The Articulate3D evaluation lives in `experiments.warsaw.articulate3d`. Its scoring
(`articulate3d/score.py`) reads a run's `EvaluationGraph`.

A run treats itself as an immutable experimental unit, and keeps these files in its
directory:

| Path | Purpose |
| --- | --- |
| `provenance.json` | Pipeline settings, input hashes, Python identity, and Git state |
| `python_environment.txt` | Exact installed distribution versions |
| `source.patch` | Local source changes relative to the recorded commit |
| `model_calls/<stage>/<question>/attempt_N.json` | Every prompt attempt, raw response, validation result, model, and latency |
| `evaluation_graph.json` | Database-independent final nodes and accepted or refused semantic edges |

The answer directories hold the latest answer per question. `model_calls` is the audit
trail, and a failed attempt stays in it when a later correction succeeds.

## The earlier evaluation against modelled worlds

Until commit `1490bff0bb`, this package also compared a run against a hand-modelled
world, for example the IAI apartment kitchen. That comparison was removed once the
Articulate3D evaluation replaced it. The code is still in that commit, and it can be
restored with:

```bash
git checkout 1490bff0bb -- experiments/src/experiments/warsaw/evaluation test/experiments_test/test_warsaw_evaluation
```

It worked in five stages:

1. **Ground truth.** It read a modelled world (from a URDF with inferred semantics, or
   from a Python world provider) into a `GroundTruthGraph`: its entities, classes,
   sizes, and every part, containment and support relation its annotations held.
   Corrections a person made by hand were kept in `ground_truth_corrections/`, and the
   written graphs in `ground_truth_graphs/`. Modules: `ground_truth.py`,
   `export_ground_truth.py`, `publish_ground_truth.py`.
2. **Alignment.** Corresponding points, picked by hand on the reconstruction and on the
   modelled world and stored in `landmarks/`, gave a closed-form rigid (optionally
   scaled) transform with its RMSE and per-landmark residuals. Modules: `alignment.py`,
   `export_ground_truth_mesh.py`, `overlay.py`.
3. **Scope.** Only fixed apartment structure was compared. Movable objects, people and
   robot bodies were left out, and a modelled entity counted only when enough of its
   surface lay near the aligned reconstruction. The scopes were in `scopes/`. Module:
   `scope.py`.
4. **Matching.** Reconstructed bodies were paired with modelled entities, by position
   once aligned and by class and size without alignment, before any per-object measure.
   Modules: `matching.py`, `placement.py`, `measure_run_bodies.py`.
5. **Measures:**
   - `label_metrics.py`: the earlier count-based HM3D label metric, kept for
     compatibility. It ignored object identity.
   - `composition.py`, `structure.py`: how much structure each graph had, per class and
     per relation pattern, without pairing objects.
   - `relation_correctness.py`: whether each relation a run asserted is one the modelled
     world holds.
   - `parenthood.py`, `siblings.py`: which whole each part ended up in, and why some
     ended up in none.
   - `ablations.py`: what each part of a run was worth, by leaving it out and scoring
     again.
   - `evaluate.py`: gathered all of these into one report.
   - `overlap_case.py`: cut one overlap out of a scan for a figure.

Back then, the model was also offered containment (`add_object`) and support
(`add_supporting_surface`) besides part-whole, and a relation's kind was carried through
these records. The pipeline now offers and mounts only part-whole relations, so the kind
is gone.
