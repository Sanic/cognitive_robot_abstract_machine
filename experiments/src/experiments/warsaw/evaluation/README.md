# Warsaw SDT evaluation

This package holds what a run is judged and looked at by:

| Module | What it is |
| --- | --- |
| `graph.py` | `EvaluationGraph`: a run's bodies, their classes, and every mount it attempted, accepted or refused, independent of the database |
| `size.py` | The size of a body, which the graph records |
| `export_world_mesh.py` | A finished run's world as a GLB, one named node per body, for Blender |
| `label_vocabulary.py` | Matching our class names to another dataset's label vocabulary (habitat comparison, lexical baselines) |
| `results.py` | What finished runs come to, as LaTeX tables per condition and per run |
| `own_ground_truth.py` | Writing the ground truth of a scene of our own, from its labelled mesh and a person's decisions |

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

## Tables of results

Any finished run can be reported, whatever scene it was made of:

```bash
python -m experiments.warsaw.evaluation.results <run> [<run> ...] --output tables/ \
    [--rename "<condition>=<name for the reader>" ...]
```

Runs are grouped into **methods** by themselves: runs with the same settings and the same
code are one method. A method is called by a short name saying only what tells it apart
from the others, such as `Qwen3.8-27B, images`, or `v2` for a later version of the code.
`methods.tex` defines each name, and `shared_settings.json` holds every setting the methods
have in common. The scene, where a run was kept and whether its renders were drawn in a
hidden window do not make a method. The command prints the names it found; pass `--rename`
to call them something else in the tables, which can be done again at any time. A run that
did not finish is refused rather than averaged in, so leave it out and say so in the text.

| Table | What it says | Columns | Needs a score |
| --- | --- | --- | --- |
| `classes` | whether the objects and their classes are right | parts found, class accuracy, hierarchical F1 | yes |
| `links` | whether the part hierarchy is right | link precision, recall and F1, ancestor F1, triplet F1 | yes |
| `structure` | how much a method asserted, whether right or not | labels mapped, new classes, bodies annotated, bodies mounted, mounts accepted | no |
| `effort` | what was asked of the model and how long it took | questions, corrected and unusable answers, thousand tokens, cost, model and total minutes | no |

`classes` and `links` compare a run with the ground truth. `structure` does not know what
is right: a method mounting every part into the wrong whole looks perfect there. It is what
a scene without a ground truth can report, and it shows whether a method got its score by
asserting little.

Each exists per method as `conditions_<table>.tex` and per run as `runs_<table>.tex`, and
`all_tables.tex` holds them all. What every row of a table shows the same is said once in
its caption rather than in a column, including how many runs and scenes it covers; a column
comes back where the methods differ, as with unequal numbers of runs. Each table's `.json`
keeps every column. A row covering one run gives its value, and a row covering several gives
the mean and the standard deviation over the runs, so over several scenes that spread holds
the differences between scenes as well. The cost is left out on a service that names no
price, such as the server on denver.

The document needs:

```latex
\usepackage{adjustbox}
\usepackage{booktabs}
```

```latex
\input{tables/methods}
\input{tables/conditions_links}
```

### Scoring a run of any scene

The scores come from `articulate3d_score.json` in the run's directory, which
`python -m experiments.warsaw.articulate3d.score <run> ... --ground-truth <directory>`
writes. Run it before writing the tables; a run without a score reports `--` in the score
columns, and without any scored run those tables are left out.

The command judges any run with a ground truth file. A scene that was not converted from
Articulate3D is called by its directory's name, so a scan in `dataset/kitchen2/` is judged
against `<directory>/kitchen2_ground_truth.json`. Write that file once, before looking at
any run of the scene:

```bash
python -m experiments.warsaw.evaluation.own_ground_truth draft <scene directory> \
    --output kitchen2_draft.json
# decide everything in kitchen2_draft.json, then:
python -m experiments.warsaw.evaluation.own_ground_truth build <scene directory> \
    kitchen2_draft.json --output <ground truth directory>
```

The draft lists every label and every object of the scene's mesh, with how many faces it
covers, how high it sits and what it meets. Overwrite these words:

| Where | Starts as | Write |
| --- | --- | --- |
| a label's `semantic_class` | `undecided` | the class of the ontology it means, or `not judged` where it covers different kinds of things |
| an object's `semantic_class` | `as its label` | another class, only where this object is not what its label means |
| an object's `whole` | `undecided` | the name of the object it belongs to, or `no whole` |
| an object's `field_name` | `null` | the field holding it, only where `build` says the whole has several |

Delete an object from the file to leave it out, such as a wall nobody means to judge;
whatever a run makes of its faces is then not judged. `build` names everything still
undecided and refuses a class the ontology does not hold, a whole the file does not list,
and objects belonging to one another in a circle. Keep the draft, since it is what the
ground truth was decided from; `draft` does not write over an existing file.

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
