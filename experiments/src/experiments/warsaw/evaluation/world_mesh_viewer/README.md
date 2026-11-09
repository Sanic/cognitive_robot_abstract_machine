# Pipeline world inspector

Browse a pipeline's final geometry, classes and object hierarchy without PostgreSQL,
ROS, ORM generation, or new model calls. This builds on the local world label inspector
from the sibling checkout and reuses its bundled Three.js 0.180.0 and two-pass rendering.

Serve this checkout from its repository root:

```bash
/home/pmania/venvs/cram-env2/bin/python -m http.server 8001 --bind 127.0.0.1
```

Open the kitchen2 Qwen3.8 run:

<http://localhost:8001/experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/?run=../../pipeline_runs/2026-11-06_162811/>

Use `?run=<directory URL>` for another run, or `?export=<directory URL>` for a standalone
`world_mesh` export. **Open folder** accepts either a full run folder or its `world_mesh`
folder; those files stay in your browser and are never uploaded. **Open run** accepts a
run directory URL relative to the viewer, or an absolute URL on the same HTTP server.

- Search by object name or its own class. Class filtering does not assign an enclosing
  cabinet's class to its handles.
- Click an object in the scene or list to see its name, own classes, parent and children.
  Relationship buttons select and frame the corresponding object.
- The selection is pink, its descendants cyan and its immediate parent amber. Emphasis
  remains visible through surrounding geometry; picking prefers those visible highlights.
  Surroundings keep their full opacity by default; reduce the slider to see through them.
  Highlighted objects have a thin bounding frame rather than internal triangle edges,
  so dense reconstructed surfaces keep their bright selection colour.
- **Isolate selection and its parts** shows the whole selected assembly. Without a
  selection, it isolates the current search/class results.
- **Hide object** hides only that object's mesh, so hiding a cabinet leaves its drawers
  visible. **Show whole world** restores hidden objects, filters and the full camera view.
- **Fit object** frames only the selected body; **Fit assembly** includes all its parts.
- Object IDs and objects without geometry remain inspectable from the list.

The viewer reads the existing schema version 2 `manifest.json` and its `scene.glb`.
Classes come from the body's direct semantic annotations. Parent/child navigation comes
from the final exported hierarchy, excluding the synthetic world root. For pipeline
worlds this is the result of mounting parts; for arbitrary exports these are kinematic
parent/child connections. These links do not claim support, proximity or containment,
and rejected proposals are not displayed as relationships.

Finished runs already have the required exports and can be opened directly. Serve the
checkout that contains both the viewer and the run; a server rooted in a different
checkout cannot see this run.

## Bundled library

The existing viewer's Three.js vendor files retain their MIT license in `vendor/LICENSE`.
No package installation, build, CDN, or internet connection is required. JavaScript
modules require HTTP serving; opening `index.html` as a file is not supported.

## Prediction and GT comparison

The [comparison page](comparison.html) displays two linked world inspectors. With the
same repository server on port 8002, open the prepared kitchen comparison:

<http://localhost:8002/experiments/src/experiments/warsaw/evaluation/world_mesh_viewer/comparison.html?comparison=../comparisons/kitchen2_qwen38_2026-11-06/comparison.json>

The GT is `kitchen_environment_2026-11-05`. The accepted kitchen2 landmark alignment was
recovered from commit `1490bff0bb`; the bundle keeps a copy of it and both export
manifests and GLBs. The prediction is displayed in metric GT coordinates, removing the
pipeline's source-frame transform before applying the landmark transform.

- Click a prediction to highlight its saved GT candidate. Click a GT object to highlight
  *all* assigned predictions, including several fragments of one object.
- Filter unmatched predictions, unassigned GT, class disagreements, parent disagreements,
  tied best candidates, or several predictions assigned to one GT.
- Read the total cost, metric distance, size difference and margin to the runner-up.
  Each prediction retains three ranked candidates. Clicking an alternative previews it
  without editing the saved association.
- Class and parent flags are judged *after* geometric association. A parent is left
  unjudged when its correspondence is missing. The same parent and part navigation as
  the individual world inspector remains available in the review panel.
- Opacity and isolation apply to both panes. An unmatched counterpart is empty under
  isolation rather than being replaced by a nearby object.

These are **provisional inspection candidates**, not the current evaluator's shared-face
correspondence and not accuracy measurements. Distance is from the prediction's bounding
box centre to the GT box, normalized by 0.5 m and capped at one, plus 0.3 times the
oriented-box size difference. A cost above 0.8 is unmatched. Those geometric weights and
limits come from the earlier spatial comparison; its semantic-class penalty is omitted.
This can confuse surfaces, parts and enclosing objects in the same place. Use the
alternatives and margins to inspect that uncertainty; the cost is not a probability.
GT outside the scanned area remains unassigned, so it must not be counted as missed.

Generate another bundle with the configured Python interpreter:

```bash
python -m experiments.warsaw.evaluation.export_comparison \
  <run directory> <GT export directory> <raw-scan-to-GT alignment.json> <output directory>
```

For programmatic policy changes, pass `ComparisonSettings` to `ComparisonExporter`.
No pipeline run, model request, ROS session or database connection is needed. The source
worlds and their saved evaluation results remain unchanged.
