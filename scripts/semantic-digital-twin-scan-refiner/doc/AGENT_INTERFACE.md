# Agent interface

The local workspace server owns the project, worker manager, action lock and Undo
journal. An agent connects to that running server through `AgentClient` or
`agent_cli.py`; it does not open the project as a second writer. All implementation
and entry points live in this refiner directory.

## Start and connect

Direction agreed 2026-11-20: the agent runs in the existing conversation or Codex
CLI, using this interface. The viewer supplies inspection context and proposal
review. The selection handoff and grouped translation review are available.
Follow the current priority section in
[ROADMAP.md](../ROADMAP.md), rather than adding another protocol or manual wizard.

In **Inspect → Region**, select a body, a world-tree branch, a group, or a drawn crop, then choose
**Copy investigation context** and paste the result into the existing Codex
conversation with your request. If clipboard access fails, the viewer displays
selectable text. The handoff carries the viewer URL, frozen input identities, the
selected body, the viewed and accepted registration, and the exact crop transform.
For a branch or group the handoff includes `selected_assembly` with its identity,
label, origin and qualified `body_names`. A branch has origin `kinematic` and
provides the qualified `root_name`. The nested `region_query` contains the
combined crop. Membership defines the scope to inspect. Include explicitly
justified edits in a later change set.

For an agent receiving `semdt-investigation-v1`, use `project.url` for the
existing local service. Save the nested `region_query` object as a plain JSON
request to `agent_cli.py create_region --url <project.url> --request <file>`, or
pass it as a `RegionQuery` through the typed client. This retains the oriented
crop or resolves the named body's contextual bounds. Use the returned region ID
with `inspect_region` or `render_region`, and pass the copied candidate index
explicitly to renders and proposals to preserve a provisional placement. A crop
can contain several bodies even though the viewer selects one object at a time.

Keep the copied `expected_revision` when proposing changes. If the service
reports that the revision changed, ask for a fresh selection before treating
those bounds as the user's current target. Copying does not change project
history, transfer scan points or grant permission to apply changes.

From the repository root, using the configured interpreter:

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/serve_agent.py \
  --project scripts/semantic-digital-twin-scan-refiner/projects/apartment \
  --port 8815

/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/agent_cli.py \
  inspect_project --url http://127.0.0.1:8815
```

The default mode permits inspection and unapplied proposals, while Apply,
registration acceptance and history traversal are disabled. For scoped changes,
start the server explicitly with, for example:

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/serve_agent.py \
  --project /absolute/path/to/project --port 8815 \
  --mode refine --body Apartment/cabinet1 --maximum-translation-m 0.03
```

Repeat `--body` for additional objects. In refine mode an empty body list permits
all objects, so specify the intended bodies for a focused session. Registration
changes additionally require `--allow-registration`. Translation and extent-change
limits default to 0.05 metres; the point budget defaults to 150,000. These are
server configuration, not permissions a tool request can grant itself.

The browser uses the same URL. Its regular job poll checks a cheap revision token
and refreshes geometry/registration after local operations finish. An agent change
therefore appears in the open browser without adding a duplicate history action.

## Grouped translation proposals

Submit a request with `submit_change_set`. The JSON below illustrates a three-body
request. Use current revision and region identities and measured or explicitly
reasoned edit values for a real proposal.

```json
{
  "expected_revision": "<current project revision>",
  "region_id": "<retained region ID>",
  "reasoning": {
    "explanation": "Move the three panels as a unit.",
    "assumptions": ["Preserve height and spacing. Keep cabinets and countertop fixed."]
  },
  "edits": [
    {"body_name": "apartment/island_back", "translation": [0, -0.012, 0]},
    {"body_name": "apartment/island_waterfall_left", "translation": [0, -0.012, 0]},
    {"body_name": "apartment/island_waterfall_right", "translation": [0, -0.012, 0]}
  ]
}
```

List 1–50 distinct bodies, including any attached parts that should follow.
Each root-frame XYZ delta applies once to both roles and composes with accepted
overrides. A linked region is optional and must come from the observed revision.

In **Refine**, choose the proposal under **Agent proposal**. Use **Refresh
proposals** to fetch new submissions. A proposal with a linked crop can run
**Compute paired heatmaps**, then **Show paired heatmaps** when the worker finishes.
**Original / Proposed** keeps the camera, visible layers, sampled records and
0–40 mm colour scale while switching the geometry and distances together.
The paired worker compares to all visual surfaces, including fixed neighbours,
under the displayed registration. It uses the dense index if present and the
preview otherwise. The scan menu can switch between Distance, RGB and hidden.

`list_change_sets` reports the current queue. In refine mode, **Accept** or
`decide_change_set` rechecks the body/movement scope and creates one history action
and an agent receipt. **Reject** records a decline in the proposal queue. Use the
existing Undo/Redo controls for geometry history. Inspect mode allows submitting
and previewing proposals. Grouped edits currently support translations. Single-body
box fitting is available under Advanced.

## Operation contracts

`scan_refiner/agent/models.py` defines validated keyword-only dataclasses.
`interfaces.py` groups the small inspection, proposal, decision and source
interfaces. `AgentService` implements the local operations; `AgentClient` implements
the same contracts over loopback HTTP.

| Operations | Result |
| --- | --- |
| `inspect_project`, `find_bodies` | Revision, authority, frozen input identities, geometry and optional verified source metadata |
| `create_region`, `inspect_region`, `render_region` | Reusable crop identity, bounded sample provenance, HTML and optional PNG artifacts |
| `propose_registration` | Existing upright registration worker and unapplied candidates |
| `propose_refinement` | Existing constrained translation or verified-box dimension worker |
| `submit_change_set`, `list_change_sets`, `decide_change_set` | Grouped translations and one accepted history action |
| `compute_change_set_heatmap` | Revision-bound worker for matched original/proposed distances, with one binary point array and two distance arrays |
| `propose_comparison` | Existing regional residual and model-support comparison worker |
| `inspect_job`, `cancel_job`, `compare_proposal` | Progress, measured results, explanations, optional matched-camera before/after images |
| `apply_proposal`, `accept_registration` | Revision-checked decision within the configured authority |
| `undo_action`, `redo_action` | Exact agent-action history traversal; intervening decisions are preserved |
| `review_source`, `export_source` | Existing bounded source-copy validation and independent export bundle |

The CLI accepts a plain JSON request file with `--request /path/to/request.json`.
It validates the input and uses the typed client. Its output uses the repository's
`DataclassJSONSerializer` format, including type and collection metadata. Requests
sent directly to HTTP must use that serializer; use the CLI or Python client rather
than constructing those envelopes manually.

`Reasoning` keeps an explanation and assumptions separate from worker measurements.
Every proposal and Apply records an explanation. Unsupported fields, including
unimplemented shared-height/centering constraints, are rejected instead of ignored.

## Evidence and coordinate contract

- All geometry, crop bounds, tolerances and movements are in metres.
- Geometry summaries and fitted translations use the model root frame.
- `RegionQuery.frame_T_scan` maps scan coordinates into an oriented inspection
  frame. Region lower/upper bounds are in that frame. Body-derived crops require
  a current placement candidate.
- Regions retain their frozen scan digest and creation revision. Reusing a region
  does not silently follow a subsequently moved body.
- Dense sampling requires the attached high-resolution index. Preview sampling
  is explicit and may return fewer points after cropping the coarse preview.
- Samples retain original point records in NPZ artifacts. Responses contain paths,
  counts and provenance, not large arrays in the agent context.
- Renders reuse the viewer's scene builder, compensate the display origin and offer
  perspective or orthographic top/front/side cameras. Rendering does not edit the
  working project's inspection settings.
- `placement_job` renders unapplied registration candidates. `comparison_job`
  renders the comparison's original sampled records and distances with a
  green-to-red heatmap, requiring the same crop/frame and geometry role.
- Before/after proposal images use identical records, crop and camera. Fitting
  residuals are in-sample diagnostics, not independent accuracy estimates.

PNG generation uses an existing Chromium executable (`--browser` can select it).
HTML generation works without Chromium. Sandboxed Chromium installations may need
an ordinary home-directory project path; Snap Chromium cannot read the host's
`/tmp` files. The server process also needs permission to launch the browser.

## Revision and history contract

Mutation requests carry the revision returned by `inspect_project`. Its digest
includes geometry, registration, source identity and accepted-candidate selection.
The service rejects stale requests before work or mutation, and Apply rechecks the
actual movement against the current server scope. Existing worker baseline checks
remain in place.

Agent actions additionally retain complete before/after journal fingerprints.
Undo refuses to cross a later human action, including a changed inspection setting.
Redo requires the exact next history state. Reducing the server's body/movement
scope also restricts traversal of earlier agent actions.

The HTTP boundary translates only request-validation errors and `InvalidEvidence`
into structured errors. Core operations still raise, and unexpected programming or
process failures are not hidden. The listener is loopback-local; it is not a
multi-user remote service or authentication layer.

Source review/export uses the same child task as the browser, with its 16 GiB memory
ceiling and 240-second timeout. Export publishes accepted geometry as a new copy;
it does not edit the original URDF/provider or apply a pending proposal.

## MCP and follow-up work

No MCP dependency was added. The service and CLI can already be invoked by an agent
with local command execution; that is the agreed integration for the next steps.
Only if a concrete need arises, a thin MCP adapter could register the typed
client methods from `AgentRegistry` and return rendered PNGs as image content. It
must attach to the running workspace rather than create a second worker owner.
Installing an optional protocol SDK remains a separate dependency decision.

This increment does not add new fitting algorithms or claim semantic constraints
that the numerical workers do not implement. Shared alignment constraints and
fitting a new box/cylinder/object remain roadmap work. Repeated scans are outside
the present scope.


Workers now request a 16 GiB address-space ceiling. A lower inherited hard limit
from the launcher is respected, including memory-capped tests. The shared policy
lives in `scan_refiner/process_memory.py`; running workers retain their existing
limits, while new analysis/source processes use the updated default.
