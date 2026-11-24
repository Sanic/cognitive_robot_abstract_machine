# Workflow-oriented refiner interface

Original staged direction accepted 2026-11-15; updated 2026-11-20. Prepare, Align,
Inspect and Refine are implemented, including single-body proposals and validated
source-copy export inside Refine. Many broader capabilities below are design
ideas, not implemented features. See README.md for the implemented scope.

## Current direction: agent-assisted inspection and review

The agent stays in the existing conversation or Codex CLI. The viewer is for
finding and inspecting discrepancies, selecting the target and reviewing a
proposal. No separate chat system or expanded manual fitting wizard is planned.
The agent investigates evidence, selects constraints and tests corrections using
the existing shared service. Detailed numerical controls belong under Advanced.

**Copy investigation context** is available in Inspect → Region for the selected
object or a crop. It provides
compact project/revision/region/frame/registration context to paste into the
conversation with a natural-language request. The review panel supports explicit
grouped translations as one reversible action. It shows affected bodies,
reasoning, assumptions, linked scan evidence and Accept/Reject. Feedback remains
in the conversation. Paired heatmaps and the RGB overlay share the same 3D viewport.

Show proposal review in the existing workspace. Keep selection and context copy
in Inspect, then open a submitted proposal in the Refine inspector alongside the
same 3D scene. Keep orbit, pan, zoom, perspective and orthographic views, along
with the existing scan and geometry controls. Switching between Original and
Proposed preserves the current camera, crop, scan points and layer settings.
Reviewers can change the angle at any time and compare both states from that
angle. Add split view only if the shared viewport and toggle prove insufficient.

The ordered plan and real handle acceptance case are in
[ROADMAP.md](ROADMAP.md#current-priority-and-handoff-2026-11-20). That section
supersedes the original six-stage/manual-control migration below. Existing stage
navigation can stay; separate Review/Export stages are not required.

## Main structure

The original design proposed a persistent workspace with six stages:

**Prepare → Align → Inspect → Refine → Review → Export**

Allow returning to earlier stages and resuming a saved project. Preserve the
selected object, region and camera when changing stages. Stage navigation shows
whether its outputs are missing, provisional, accepted or stale.

The central scene stays in place. The surrounding controls change with the task.
Undo/Redo, history, project save status and running jobs remain available globally.

```text
Apartment project · saved                         Undo  Redo  History  Jobs
Prepare   Align   [Inspect]   Refine   Review   Export
─────────────────────────────────────────────────────────────────────────
Objects / issues       Scene and section workspace       Inspect selection
Search…                [RGB] [Distance] [Layers]         Cooktop
                                                        Scope: selected object
Apartment              Perspective / Plan / Elevation    Region + context
  Kitchen island                                        Scan density
    Cooktop            3D scene                         Slice / Measure
    Cabinets                                            Save annotated view
  Kitchen run          Optional second viewport
  Living area          Equal-scale section / comparison  [Evaluate region]
─────────────────────────────────────────────────────────────────────────
Placement: accepted · frame: World · units: m · comparison: current
```

The example hierarchy illustrates the intended adapter output. The current
frozen snapshot retains body names and geometry roles; it does not yet retain
all SemDT assembly and semantic relations needed for this tree.

## Stage responsibilities

| Stage | Main question | Main controls and output |
| --- | --- | --- |
| Prepare | What are we comparing? | Choose a URDF or Python world and scan; inspect units, bounds, available attributes and index progress. Produce immutable baselines and indexed evidence. |
| Align | Where does the scan belong? | Run automatic registration; inspect alternatives, anchors and exclusions; compare overlays and accept a placement. Store its transform and evidence. |
| Inspect | What needs attention? | Navigate objects/issues, load local scan detail, compare distances and coverage, use sections and measurements, save annotated views. Produce a scoped issue or refinement target. |
| Refine | What may change? | Choose pose/dimensions/joint state, lock parameters, declare shared constraints, choose fitting evidence and generate a proposal. Keep the baseline visible for comparison. |
| Review | Is the proposed change justified? | Compare before/after using the same evaluation evidence and scale, inspect parameter changes and geometry/semantic/articulation checks, accept or reject a proposal. |
| Export | What should others receive? | Generate a portable review bundle, labelled world meshes or a supported source patch/override; show included files, asset dependencies and revision identity. |

## Prepare

Use a compact input summary rather than exposing configuration fields throughout
the viewer. Show the world source, geometry availability, scan bounds/units,
station count where available, attributes and import/index status. Keep detailed
coordinate and provenance information in an expandable panel.

Indexing has visible progress, memory/storage expectations and cancellation when
supported. A complete index can be reused across sessions. Show which operations
are available on a preview while a full index is unavailable.

## Align

Put automatic placement generation and its job progress in the right panel.
Show alternatives as selectable cards with descriptive diagnostics, not invented
confidence percentages. The scene offers overlay, plan/elevation and separated
regions for checking a placement across the environment.

Optional browser-picked anchors and masks are available when automatic alignment
needs help. Gravity/levelling and partial-overlap diagnostics belong here.
Acceptance fixes the registration revision used by subsequent comparisons.

Registration remains distinct from moving an individual object in Refine.
Unregistered inputs can still be viewed, with an explicit placement status.

## Inspect

The left panel shows a searchable **World tree** based on SemDT's kinematic
connections. Clicking a branch frames the geometry of that entity and its
descendants and sets the inspection crop. The mesh icon on a branch selects
only that row's body. Expanding rows exposes geometry-free frames as well as
individual objects. The **Groups** section lists semantic assemblies and
explicit selections created with **Save a named assembly**. Groups may overlap
with one another and with the branches shown in the world tree.

Keep only a small viewport toolbar:

- Scan colouring: RGB, distance or hidden.
- Geometry: visual, collision or both, with the active comparison role visible.
- View: perspective, plan or elevation.
- Select region, section and measure.

Move point size, opacity, raw XYZ bounds and diagnostic IDs into an expandable
Display/Advanced panel. Picking an object should offer **Focus** and **Load local
scan**, with a visible context margin. Support drawing and adjusting a box in
the scene or section instead of requiring typed coordinate bounds.

Keep display density distinct from fitting/evaluation evidence. Show the loaded
record count and scan region; changing rendering quality must not silently change
the records used to fit or evaluate a proposal. Automatic detail loading can be
added later, after profiling the explicit loading workflow.

Sections should open as a resizable companion viewport, next to or below the
3D scene, with equal metric axes. A visible frame selector offers World, Object
and Scan coordinates once supported. This avoids forcing users to work with
large scanner coordinates or a rotated apartment plan. Available choices must
match the implemented section and inspection frame contracts.

Issues can be created from a selection, measurement and saved view. Distinguish
surface-distance discrepancies, missing model geometry and unknown/unobserved
scan coverage. A point-to-surface distance alone must not declare an object wrong.

## Refine

The original manual design proposed the following advanced capabilities. Under
the current direction, the agent chooses these settings; the ordinary user reviews
the resulting proposal rather than completing a parameter wizard:

- Pose, dimensions and articulation parameters supported by its world adapter.
- Explicit locks and permissible ranges.
- Relations such as centred-on, same-height, supported-by and parallel.
- Fitting region and exclusions, separate from the display clip box.
- A bounded **Fit proposal** job and visible diagnostics.

Use scene manipulators and unit-labelled fields together. For a group of handles,
for example, expose a common height or axis constraint rather than independent
noisy fits. For an assembly, show which children move with it and which previous
fits need rechecking.

New geometry remains an explicit follow-up: **Add object from scan**, beginning
with boxes, cylinders and simple objects. Fitting geometry and assigning its
semantic role are separate decisions. Do not infer a precise shape from
unobserved surfaces without a declared constraint.

## Review

Provide before/after switching and an optional split view with shared camera and
colour scales. A change list shows affected bodies and parameter values, linked
to the evidence, constraints and measurements that motivated each proposal.

Run the checks relevant to the change: surface fit and coverage, attachment and
articulation, semantic membership, and visual/collision policy. Allow different
visual and collision geometry where intentional; do not automatically make them
identical because a scan matches one representation better.

An accepted parent move marks dependent child fits and comparisons stale. Keep
those consequences visible. Accepting a project proposal does not silently edit
an external URDF or arbitrary Python source.

## Export

Offer deliverables by purpose:

- Read-only review: viewer, selected evidence, annotations and revision metadata.
- Downstream use: labelled world geometry and its manifest.
- Continued modelling: supported source patch/override and required assets.

Preview the file list, geometry role and included scan data. Keep full raw scans
out of the default small review bundle. Generated artifacts identify their input
revision; outdated results remain visibly stale until regenerated.

## Cross-cutting behaviour

Undo/Redo and a persistent action history cover human, CLI and LLM actions.
Display selection, fitting selection and evaluation selection are separate
concepts even when they initially use the same region.

Long operations use one global job drawer with scope, progress, cancellation and
result status. Completed jobs do not replace a newer selection or publish results
against a changed registration unnoticed. Avoid leaving overlapping controls active
while an operation is restoring state.

The external conversation/Codex CLI uses the same service operations and history
as the viewer. No separate assistant chat panel is planned. Proposal review must
not introduce a second writer or a separate route for changing source files.

## Original migration sketch (superseded by ROADMAP.md priorities)

1. Reorganize existing inspection controls into the shared shell, a small toolbar
   and a contextual panel. Preserve current APIs, Undo and data formats. Show
   planned stages clearly as unavailable until implemented.
2. Add selection focus, graphical region editing and resizable companion sections.
   Preserve named views across navigation. Keep frame choices explicit.
3. Add bounded web registration/comparison jobs and a global job drawer. Add
   semantic hierarchy and coverage summaries as their adapters become available.
4. Implement the Refine → Review proposal cycle, then source application/export.
   Add new-object fitting as the recorded follow-up.

Reuse the installed renderer initially. Avoid a rendering-library or framework
migration as a prerequisite for the workflow improvement.


## Foreground loading feedback (2026-11-19)

Startup and view restoration show a centered status panel over the viewport. The
embedded figure remains inert until this panel has painted. Undo/Redo name the
operation and show restoration steps; scan loading has its own status. Nested
operations retain the outer title and keep the panel visible until all steps finish.
Conflicting controls are inert while work runs, with their prior state restored
afterward. Use indeterminate feedback, not an estimated percentage. Startup errors
show a readable message and a Reload viewer action. Background analysis jobs keep
their separate Jobs progress and do not block scene navigation.


## Foreground operation audit (2026-11-19)

The same panel covers Orient axes and placement transforms, geometry/layer redraws,
focus/show-all, crop plans and sections, measurements, return to overview, loading
saved heatmaps, refreshing the shared project, proposal previews/application and
source review/export. It paints before expensive synchronous work, and nested
redraws retain the outer action's title. Mouse camera movement and automatic resize
stay quiet. Background registration/comparison/fitting continues to use Jobs.
