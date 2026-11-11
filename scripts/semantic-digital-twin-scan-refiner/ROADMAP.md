# Agreed scope and follow-up reminders

## Agreed direction

- Reusable infrastructure for Python-defined and imported URDF SemDT worlds.
- General environments: kitchens, living rooms, laboratories and factory floors.
- Automatic alignment as the default, reviewed in the web interface.
- Manual landmarks, if needed, picked directly in the browser as a fallback.
- One registered capture for now. Repeated scans are outside the current scope.
- Local tools first; an LLM can use the same operations as a human or CLI.
- Work only inside `scripts/semantic-digital-twin-scan-refiner` for now.
- Undo is a core requirement: expose an Undo button and record reversible
  project actions from the start, including actions initiated by an LLM.

## Undo requirement

Before adding project-changing actions, give them a persistent before/after
record. Cover registration acceptance/replacement, selections and exclusions,
constraints, object changes and source application. Undo must restore the prior
project state and invalidate or refresh affected viewers, heatmaps and exports.
Retain immutable scan/model baselines and earlier registration candidates;
rerunning alignment must become a new revision rather than erase prior decisions.
The current CLI still replaces registration candidates, so action history and
the visible Undo button are pending implementation. Include redo where practical.

## Remember: fitting NEW objects is requested follow-up work

After inspection and existing-object refinement are usable, remind the user to
add missing geometry from a selected scan region. Start with boxes, cylinders
and simple objects, allow dimensions/pose to be locked or fitted, and turn the
accepted result into a SemDT body with user-reviewed semantics. Preserve the
selected evidence and constraints. This is explicitly wanted, not discarded.

## Next implementation increments

Implemented on 2026-11-10: bounded disk indexing, isolated multi-station E57
reader, cached overview, exact regional batches, bounded original-record samples,
body-based region selection and web detail requests. The rendering payload was
reduced without changing libraries. Continuous camera-driven loading is deferred.

1. Extend body/region selection with clip boxes, fixed plan/elevation sections, measurements
   and saved annotated views. Avoid repeating the earlier skewed-view problem.
2. Web-triggered registration and regional heatmap jobs with progress/cancellation, automatic gravity
   handling, masks, anchors and better partial-overlap/repetitive-layout search.
3. Coverage-aware comparison in both directions, per-object summaries and
   tolerances. Missing scan returns must remain distinguishable from errors.
4. Versioned proposals, constrained pose/dimension refinement and shared alignment
   constraints. Parent moves invalidate affected child fits and artifacts.
5. Reviewable source application or persistent overrides, actual world reload,
   semantic/articulation checks, export and portable review bundles.
6. Missing-object fitting as described above; exercise non-kitchen examples.
7. Profile whether camera-driven loading, compressed storage or another viewer
   is needed beyond the current regional access. Evaluate established tools
   before new dependencies; dependency additions require user approval.

## Later, only when needed

Repeated scans/change detection, registration of separate scanner stations,
robot mapping/SLAM sessions, learned object recognition, arbitrary mesh editing,
multi-user editing and enterprise hosting.

