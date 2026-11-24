# Agreed scope and follow-up reminders

## Current priority and handoff (2026-11-20)

This section supersedes the implementation ordering below. Dated increments
describe existing capabilities, not a requirement to expand a manual refinement
wizard.

**The agent stays in the existing conversation or Codex CLI.** The web viewer is
the shared workspace for close inspection, communicating the target and reviewing
proposals. The user expects to delegate most investigation and correction in plain
language: “These handles look wrong sideways; investigate fixes, but preserve
their heights.” Do not build a separate viewer chat system or a general autonomous
reasoning framework. Follow KISS; add capabilities when real cases require them.

The intended loop is **request → agent investigation → proposal → review**. The
agent chooses hypotheses, evidence regions, constraints and fitting operations.
The user should not need to configure numerical fitting settings. Keep existing
detailed controls available under Advanced rather than deleting useful tools.

### Next increments, in order

1. **Connect viewer selections to the conversation (implemented).** “Copy
   investigation context” handles a selected body, world-tree branch, named group,
   or crop. It captures project identity, revision,
   qualified body identities, region/frame, registration and relevant inspection
   settings in a compact structured handoff. The user pastes it with their request;
   the agent retrieves detailed evidence through the existing service/client/CLI.
   Do not put large point arrays into the prompt or grant mutation authority
   through copied context. Detect stale context before running proposals against
   it or applying changes. Done when another model can inspect exactly the
   selected scene state without guessing filenames, frames or body names.
   The copied `region_query` is accepted directly by `create_region`; requests
   carrying a stale revision are rejected. See the agent guide for the handoff.
2. **Support coherent multi-object change sets (translation slice implemented).** Extend the single-body proposal
   path with explicit changes to multiple bodies, a complete affected-body preview
   and one reversible Apply/Undo action. Initially the agent can enumerate edits;
   do not build a general assembly solver. Moving a cabinet and its parts must
   preserve intended relative positions without double-transforming descendants.
   Show membership explicitly; do not infer attachments silently from names.
   Preserve revision checks, scope limits, frozen baselines, analysis invalidation
   and source-export validation. Accept reasoned explicit edits, such as centring
   handles, without requiring every change to originate in a scan optimizer.
   Grouped translation is supported. Grouped box dimensions and rotation remain
   follow-ups as needed.
3. **Make proposal review the main refinement UI (interactive 3D heatmaps implemented).** Show what changes, affected
   objects, explanation, assumptions and uncertainty. Reuse matched-camera
   original/proposed views, scan overlays and comparable heatmaps. Provide a
   before/after toggle, Accept and Reject. Show review in the current Refine
   inspector and reuse the interactive scene. Preserve the camera, crop and
   visible layers across toggles so reviewers can compare from any angle.
   Feedback can remain in this conversation.
   Residual improvement alone is not proof of physical correctness. Investigation
   and previews need not apply edits; mutation follows existing user authorization
   and configured service scope, with Undo available.
   Review now lives in Refine, with Original/Proposed, linked regional evidence,
   Accept/Reject and Advanced single-body tools. A paired job calculates both
   residual arrays from identical scan records and applies a shared colour scale.
4. **Prove the loop on real apartment handles.** Investigate sideways placement
   using drawer geometry, original positions, scan quality and previous changes.
   Preserve current heights unless instructed otherwise. Distinguish registration,
   assembly placement and individual-handle errors. Test a coherent correction,
   inspect neighbouring geometry for regressions and present one reviewable
   proposal. This is the next end-to-end acceptance case. Add centring/shared-height
   or attachment helpers only where justified; do not independently fit every noisy
   handle patch or assume all handles have the same mounting relationship.
5. **Package accepted results for sharing.** Reuse native source-copy export and
   reload validation. Add a concise change report and supporting comparison views
   tied to the accepted revision. Do not export unapplied previews or overwrite
   original inputs as part of this flow.
6. **Extend from demonstrated needs.** Add reusable group, centring, common-height
   or attachment constraints when they remove repeated work. Retain the requested
   new-box/cylinder/object fitting follow-up below. Repeated scans remain out of
   scope. Richer registration and coverage reasoning are later improvements, not
   prerequisites for this agent-assisted workflow.

### Starting point for the next model

- Work on branch `semdt-world-refiner`; inspect Git state first and preserve
  existing staged and unstaged work. Do not commit or push unless requested.
- Keep changes inside `scripts/semantic-digital-twin-scan-refiner`. Follow the
  repository AGENTS.md, including configured Python, tests and dependency rules.
- Read [HANDOVER.md](HANDOVER.md), [the agent interface](doc/AGENT_INTERFACE.md)
  and the existing viewer/inspection-state code before starting the handle case.
  The typed service/client/CLI already exposes inspection, rendering, registration,
  fitting, comparison, decisions, history and source export. Reuse it; do not
  create a second project writer or duplicate worker ownership.
- Current fitting supports one-body constrained translation and verified-box
  dimension changes. An explicit change set can group root-frame translations
  for several bodies. Include attached parts by name when they should follow.
  Centring and shared height still require the agent to compute explicit edits.
- Main workspace is normally <http://127.0.0.1:8815/>; verify the listener rather
  than starting a second writer. Server mode/scope controls agent authority.
- [The apartment use-case video](doc/APARTMENT_USE_CASE_VIDEO.md) demonstrates
  registration, inspection, one-panel correction, history and export on real scan
  evidence. Its panel error is deliberately introduced in a disposable copy; it
  does not demonstrate automatic assembly reasoning or an outstanding production
  defect. Do not use the recording copy as the production refinement baseline.
- Reuse the guided-video conventions below when demonstrating the new workflow.
  A recording is not a substitute for the real handle acceptance case.

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
Implemented 2026-11-15 for registration and inspection: persistent undo/redo,
retained candidates/acceptance, camera/query/clip/section settings, measurements
and annotated views. Geometry overrides now also use the journal and invalidate
derived analyses. Multi-object changes must extend that journal as one reversible
action.

## Remember: fitting NEW objects is requested follow-up work

After inspection and existing-object refinement are usable, remind the user to
add missing geometry from a selected scan region. Start with boxes, cylinders
and simple objects, allow dimensions/pose to be locked or fitted, and turn the
accepted result into a SemDT body with user-reviewed semantics. Preserve the
selected evidence and constraints. This is explicitly wanted, not discarded.

## Implementation history and remaining technical capabilities

Implemented on 2026-11-10: bounded disk indexing, isolated multi-station E57
reader, cached overview, exact regional batches, bounded original-record samples,
body-based region selection and web detail requests. The rendering payload was
reduced without changing libraries. Continuous camera-driven loading is deferred.

1. Implemented 2026-11-15: scan clip boxes, fixed metric plan/elevation sections
   with exact model outlines, point-pair measurements and saved annotated views.
   Large point loads use bounded render batches. VirtualGL-specific QA remains.
2. Implemented 2026-11-15: web-triggered upright registration and regional heatmap
   workers, progress/cancellation, explicit proposal application and stale-result
   protection. The staged Prepare/Align/Inspect shell, world/yaw-oriented crops,
   graphical plan selection and frame-consistent sections are available.
   Still pending: automatic gravity handling, masks, anchors and improved
   partial-overlap/repetitive-layout search. Registration jobs use the frozen
   overview preview; comparison jobs can use regional high-resolution evidence.
3. Implemented first slice 2026-11-15: labelled scan-to-triangle residuals,
   reproducible model-to-scan evidence support, area-weighted regional summaries,
   per-body median/P90/sample counts, explicit tolerance and independent budgets.
   The Results inspector filters/sorts bodies and downloads a provenance-bearing
   report. Missing returns and unsampled regions remain unknown, not errors.
   Still pending: scanner visibility/free-space reasoning, density-normalized
   support and per-body tolerance overrides. Current support uses sampled returns
   and is intentionally not a full observability/coverage determination.
4. First slice implemented 2026-11-15: versioned selected-body translation
   proposals, explicit root-axis locks/bounds, robust correspondence updates,
   observability/convergence checks, temporary ghost preview and Undoable
   snapshot overrides. Both roles move together. Any override change invalidates
   all derived analyses; submitted worker geometry is frozen.
   Added 2026-11-15: verified root-box dimension proposals with independent locks,
   center/minimum/maximum anchors, required-face evidence, bounded positive sizes,
   common role deformation and pose/size composition. Eligibility checks whole
   body representations. Snapshot overrides, jobs and browser reload preserve
   both geometry revision types.
   Added 2026-11-16: verified oriented-box dimensions, canonical geometric axes,
   full root-space stretch mappings, labeled preview axes and reload/Undo support.
   Still pending: source-parametric dimensions, rotation, shared
   alignment constraints, captured hierarchy and assembly propagation. Current scope is explicitly one body;
   attached bodies stay fixed rather than guessed from names.
5. Implemented first source-copy slice 2026-11-18: explicit review of accepted
   overrides, stale-source/revision guards, native URDF copies with OBJ assets,
   provider recipes, fresh live-world reload, labelled visual/collision exports
   and portable source bundles. Checks cover geometry, graph identities, joint
   types/axes/limits and semantic membership. Originals remain unchanged.
   Still pending: in-place adoption/rebasing, arbitrary Python source rewriting,
   moving-hinge/inertia policies and stronger semantic/physical validation.
6. Missing-object fitting as described above; exercise non-kitchen examples.
7. Profile whether camera-driven loading, compressed storage or another viewer
   is needed beyond the current regional access. Evaluate established tools
   before new dependencies; dependency additions require user approval.

## Later, only when needed

Repeated scans/change detection, registration of separate scanner stations,
robot mapping/SLAM sessions, learned object recognition, arbitrary mesh editing,
multi-user editing and enterprise hosting.


## Reusable demonstration workflow

The user requested that future feature videos reuse the accepted guided style:
yellow control outlines, a visible pointer and click pulses, exact option captions,
and pauses on results. Scan-density demonstrations must hide model meshes and
show the local point cloud. See [doc/VIDEO_WORKFLOW.md](doc/VIDEO_WORKFLOW.md).

## Historical apartment eligibility finding

The read-only oriented audit accepts 16 of 102 frozen geometric bodies, compared
with 0 under the original root-axis rule. Skewed waterfall and beveled cooktop
meshes were rejected by that audit. No apartment geometry overrides were applied
by the audit. Source-copy application was subsequently added as described above;
source primitive parameters and assembly propagation remain follow-ups. Continue
retaining the requested new-box/cylinder/object fitting
follow-up after existing-object refinement.

## Performance finding from video QA

Implemented 2026-11-18: the overview restores the frozen preview directly instead
of issuing a global indexed-region query. Regional requests remain explicit.
Browser QA restored 20,000 overview records after a 50,000-record region with zero
additional dense queries, preserving Undo/Redo and reload behaviour. Profiling
more sophisticated camera-driven loading remains optional later work.


## Agent interface increment (2026-11-18)

Implemented the typed service/client/CLI around existing workers, reusable regions,
bounded evidence images, revision-checked scoped decisions, owned Undo/Redo and
validated source export. The browser shares the worker manager/journal and refreshes
after agent changes. See [the interface guide](doc/AGENT_INTERFACE.md).

A protocol adapter is deferred; no MCP dependency was added. Next work should
exercise real scene cases and improve constraints where evidence requires them.
Shared height/centering constraints and fitting a new box/cylinder/object remain
explicit follow-ups. Repeated scans remain outside the current scope.
