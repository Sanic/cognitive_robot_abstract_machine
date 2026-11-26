# SemDT scan refiner

Reusable local tools for inspecting an approximate SemDT against a measured
scan. Work is isolated to this directory on `semdt-world-refiner`, created from
`origin/main` at `d47537cf910577321df4757fa76253785bd63c59`.

## Current working slice

### Agreed direction (2026-11-20)

The agent stays in the existing conversation or Codex CLI. The viewer is the
shared workspace for inspection and proposal review; most investigation and
correction is delegated in plain language. Detailed fitting controls are advanced
tools, not the required user workflow. No separate viewer chat is planned.

See [ROADMAP.md](ROADMAP.md#current-priority-and-handoff-2026-11-20) for the
authoritative roadmap. In **Inspect → Region**, select an object, assembly or crop and
click **Copy investigation context**. Paste it into the Codex conversation with
your request. Submitted grouped translations appear under **Refine → Agent
proposal** for interactive Original/Proposed review and one reversible decision.
The [Cabinet 3 handle case](doc/APARTMENT_HANDLE_CASE.md) demonstrates this
workflow from registration review through grouped Accept and Undo.

In the left pane, **World tree** follows the SemDT joints, including frames
without meshes. Expand a row to browse its children, click the branch name to
focus its combined geometry, or use the mesh icon to focus only the body at
that row. **Groups** shows semantic assemblies and named selections saved in
the Region panel. The search field covers both sections.

- Load a trusted Python world provider or URDF through SemDT.
- Freeze actual visual and collision geometry independently in root coordinates.
- Import a PLY/NPZ preview, retaining RGB bytes and preview-record identities.
- Search upright yaw/translation automatically and retain ranked alternatives.
- Review candidates in one self-contained browser page with geometry layers,
  orthographic top/front views and an optional scan-to-triangle heatmap.
- Compare both directions with per-body residuals and sampled nearby-evidence
  support, explicit budgets/tolerance and downloadable provenance-bearing reports.
- Accept a reviewed transform explicitly and save `scan_T_root.npy`.
- Undo/redo registration and inspection decisions with persistent action history.
- Clip displayed scan points, inspect metric plan/elevation sections, measure
  distances and save annotated views.
- Index all E57 stations through an existing isolated reader, retaining raw
  record/station IDs, float64 coordinates, byte RGB and available intensity.
- Click a model body to select surrounding scan evidence and load 10k/50k/150k
  original regional records through the local server.

The viewer uses the installed Plotly bundle and needs no external scripts.
There are no added dependencies or package installation steps. The tools use
the project's configured interpreter. Providers execute trusted local Python.

This is an implementation increment, not the complete planned system.
Registration can run through the CLI or the browser job drawer. Regional detail
loads on demand; continuous camera-driven level of detail remains future work. Initial PLY/NPZ import loads a bounded
preview into memory; dense E57 indexing streams bounded batches.

## Staged workspace

The shared page now offers **Prepare**, **Align**, **Inspect** and **Refine**.
Validated source copies and exports are available inside Refine. Separate
Review/Export stages were an earlier design option, not a prerequisite for the
current agent-assisted roadmap. Refine supports selected-body translation and verified box
dimension proposals, including rotated boxes. Rotation, source-parametric dimensions and assembly
constraints remain later work.
The scene and selection survive navigation. The object list searches actual body
identities; it does not invent semantic assembly membership.

- **Prepare** shows baseline sources, hashes, geometry counts and scan-index status.
- **Align** reviews placements, accepts one through Undo history, and starts an
  automatic upright search. Confirm both Z axes are vertical. Jobs publish
  proposals separately: **Review candidates** applies them as an undoable revision.
- **Inspect** provides an object list, a compact layer/camera toolbar and Region,
  Results, Section, Measure and Views panels. History and Jobs are global drawers.
- **Refine** creates bounded translation or verified box-dimension proposals,
  previews them against current geometry, and applies reviewed snapshot overrides
  through Undo.

### Oriented crops

When registration exists, a fresh project starts with **World axes**. Both scan
and model are expressed in those axes for display. Choose **Original scan axes**
or an additional yaw, then **Orient axes**, to change the inspection basis.
This does not rotate or rewrite the stored scan or change registration.

Pick a body to fill bounds with 15 cm of surrounding context, or use **Draw crop
in plan** and drag a rectangle in the companion view. Edit the vertical bounds
under **Edit XYZ bounds** if needed. Then **Load region**. **Focus selection**
zooms the 3D axis limits to the chosen box; **Show all** restores automatic limits.
Sections use the same inspection frame and yaw as the crop.

The backend transforms the box conservatively into scan cells, then tests each
original record against the actual oriented box **before** applying its sample
budget. It preserves original float64 coordinates, RGB and station/record IDs.
A loaded query retains its original frame matrix when the display axes change.
Undo and saved views can therefore reproduce the same evidence selection.
The existing `/api/region` contract remains scan-aligned when no optional
`frame_T_scan` matrix is provided.

### Analysis jobs

**Compute regional heatmap** uses the selected box, an independent evaluation
budget, explicit visual/collision geometry and optional selected-body scope.
The worker reads indexed evidence when available and uses the frozen preview
otherwise. A completed job reports median/P90 unsigned surface distance and the
fraction of evaluated points within the declared tolerance. **Show heatmap**
loads its exact evaluated records; it does not colour a different display sample.

Jobs run in a separate process, one at a time, with a 4 GiB address-space ceiling
and single-thread numerical-library defaults. The drawer shows phase/progress
and cancellation. Registration search is indeterminate; triangle comparisons
report actual processed/total points. Closing the server stops its owned workers.
Failures remain inspectable in the job log. Interrupted jobs are not resumable.

Running a job does not replace registration or accept a placement. Changed
registration marks incompatible results stale and prevents their application.
Named views/history retain comparison job IDs so valid heatmaps can be restored
without recomputation. Keep the `jobs/` artifacts when sharing a project that
references those results. Display-region edits are not registration masks or
exclusions; those remain future work.

## Try the demo

Run from the workspace root:

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py demo \
  --project scripts/semantic-digital-twin-scan-refiner/projects/demo

/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py view \
  --project scripts/semantic-digital-twin-scan-refiner/projects/demo --heatmap

/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py index \
  --project scripts/semantic-digital-twin-scan-refiner/projects/demo --preview

/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py serve \
  --project scripts/semantic-digital-twin-scan-refiner/projects/demo
```

Open <http://127.0.0.1:8765/>. Alternatively open the project's `index.html`
directly. Initialization refuses existing directories: use a new demo path on
subsequent runs, or use `view` to reopen the existing project.

## Use your own world

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py init \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example \
  --scan /path/to/preview.ply \
  --urdf /path/to/environment.urdf
```

Alternatively replace `--urdf` with
`--world-provider package.module:WorldClass`. The class must expose a no-argument
`create_world()`. Normal ROS/package resolution must already work in the shell
for URDFs referencing `package://` assets. Providers preserve their own world
construction; bare URDF imports do not infer additional semantic annotations.
For a directly modelled world, use
`--world-provider semantic_digital_twin.predefined_maps.kitchen_environment:KitchenEnvironment`.
Both import paths capture the kinematic tree from SemDT. Existing snapshots
without parent links still show **All bodies**. Import into a new project to
capture the tree while keeping the existing project's history.

Coordinates are metres unless `--scan-scale` explicitly converts them. NPZ must
contain `points` with shape `(count, 3)` and optionally uint8 RGB in `colors` or
`colours`. Missing RGB gets a neutral display colour. Normalized float RGB must
be converted explicitly. Existing `source_indices`, `station_indices` and
`intensity` arrays are retained. Without supplied identities, record IDs refer
to the imported preview and station zero.

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py align \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example --upright

/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py view \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example --candidate 0 --heatmap

/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py accept \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example --candidate 0
```

`--upright` explicitly confirms that both input Z axes already represent gravity.
This first search fits yaw and XYZ translation, preserving scale and roll/pitch.
It initializes height from the low coordinate quantiles, then refines height.
That heuristic is not a floor detector and can fail with incomplete coverage.
Automatic levelling and region/anchor selection remain planned work.

`--geometry visual` is the default for registration; `--geometry collision` is
available. There is no implicit collision fallback. Bodies without visual
geometry are absent from visual registration and its heatmap. In particular,
the apartment's collision-only walls are not silently added to the visual model.

Choose `--cell-width` and `--yaw-step` for the scene if necessary. A grid allocation
limit rejects excessively large spatial extents; use a suitable preview/crop or
coarser cells. The initial search samples model surfaces by area, so large or
wrong surfaces can dominate. Strong clutter, repetitive layouts and unequal
coverage require further candidate and exclusion work. Do not automatically
accept the top score.

## Dense scan access

Build the index once, using the already available reader environment:

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py index \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example \
  --e57 /path/to/scan.e57 \
  --reader-python /home/pmania/.cache/semantic_digital_twin/apartment-e57-reader/bin/python
```

No reader environment is installed or modified. `--reader-python` must point
to an existing interpreter with pye57 and NumPy; its virtualenv symlink is
preserved. The worker does not import SemDT or inherit `PYTHONPATH`.

The E57 must share the preview's declared scan coordinate system. All station
poses are applied exactly once. Original record IDs remain station-local;
the station/record pair identifies evidence. Cartesian and spherical coordinates
are supported, invalid/nonfinite positions are excluded, and station transforms,
source counts and point-field names are saved in `stations.npz`. Unavailable RGB
is displayed neutrally; unavailable intensity is NaN. This record format supports
byte RGB only and refuses unsupported colour conversion. It does not preserve
every possible E57 attribute; the raw container remains the authoritative source.

Default indexing uses 250,000-record batches and one-metre storage cells. Storage
cells do not quantize the positions. The binary evidence layout costs 47 bytes
per retained point, plus a bounded 200k-record overview and small cell metadata.
For 546 million records this is approximately 25.7 GB. This simple first index
favours exact evidence and existing dependencies over compressed storage.

`scan-index/index.json` is published only after successful completion. Existing
or partial index directories are never overwritten. An interrupted import must
be moved aside before retrying; resumable construction is not implemented.

Serve the project, click a body to fill its scan bounds with 15 cm of context,
then click **Load region**. Set the point budget before loading. **Overview**
uses cached original records instead of rereading the entire dense cloud.
Numeric bounds are inclusive and expressed in the chosen inspection frame,
in metres. The raw API defaults to original scan coordinates.
These are on-demand regional samples, not automatic camera-driven streaming.

CLI `crop` and `SpatialScanIndex.batches` bounds remain in original scan axes.

The Python API also exposes `SpatialScanIndex.batches(region)` for every original
record in bounded batches. The CLI exports a bounded regional sample:

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py crop \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example \
  --lower 0 0 0 --upper 1 1 1 --maximum-count 50000 \
  --output scripts/semantic-digital-twin-scan-refiner/projects/example/region.npz
```

Both paths preserve original coordinates, IDs and available attributes. Loading
a different selection disables its predecessor's heatmap; compute a regional
comparison for the new selection instead of reusing mismatched colours.

## Reversible inspection

Use the local `serve` command for persistent **Undo** and **Redo**. History is
stored in `history.json`: registration candidates/acceptance, display choices,
loaded region descriptors, camera, clip settings, sections and annotations.
It stores metadata only, never duplicate scan/model arrays. The journal is
authoritative; compatibility metadata and the accepted transform are derived
from its current revision. A new action after Undo clears the redo branch.
Concurrent editing from several clients/processes is outside this increment.

The CLI also supports `undo`, `redo` and `history`, each with `--project`.
CLI Undo/Redo regenerates the preview; use `view --heatmap` to restore its heatmap.
Standalone HTML retains basic geometry/camera controls; the persistent
inspection tools require the local server.

Open **Sections, measurements and saved views** for:

- **Clip scan**: display only loaded points inside the oriented XYZ box. This does not
  fetch additional evidence or clip model triangles. Load the region first when
  more density is needed. Clip bounds and loaded-query bounds are independent.
- **XY / XZ / YZ section**: choose its position in original scan metres and its
  total scan-slab thickness. **Use box centre** fills a convenient position.
  **Apply section** shows an equal-scale 2D plot beneath the 3D pane, with exact
  model/plane intersections at the slab centre. The 3D scan also shows the slab.
- **Measure two scan points**: click endpoints in either view. The annotation
  reports their Euclidean 3D distance in millimetres; point selection and scanner
  accuracy still limit the measurement. Clear measurements is undoable.
- **Save annotated view**: store a name, observation, camera, geometry, loaded
  query/budget, clip and section settings. Restore and removal are undoable.

Bounds are inclusive in the selected inspection frame, independent of the local
render origin. Loaded evidence retains its query frame across orientation changes. Increasing 50k to 150k with unchanged bounds increases the display
sample budget in the same box; sparse regions can return fewer points. Large
loads are split into at most 50k points per WebGL trace. Loads and history
traversals disable overlapping region/history controls until they finish.

## Small rendering improvements

The viewer shares one scan-position buffer between RGB and heatmap modes, uses
compact float32 typed arrays around a recorded local origin, omits duplicate
mesh vertices, and updates affected layers only. The stored scan/model remain
full precision. Point size, hover and mesh opacity are adjustable; hover starts
off and meshes start opaque. No rendering-library dependency was added.

Left-button dragging starts in free rotation. Top, Front and Perspective restore
rotation after other Plotly drag modes; Rotate restores it without changing the
current viewpoint. This avoids remaining in pan/zoom mode after a view change.

Scene controls activate after Plotly finishes creating the plot. Encoded numeric
buffers are decoded before candidate transforms or body selections, including
initially hidden collision meshes. Renderer updates are serialized; geometry
switches preserve the camera and do not overlap scan/candidate updates.

The apartment page decreased from 39,333,815 to 20,045,029 bytes (49%) with the
same displayed scan selection and heatmap. This is a payload measurement, not a
browser frame-rate benchmark. Later checks used software-rendered headless Chromium; VirtualGL-specific performance has not been measured.

## Interpretation and persistence

The browser's support value is the fraction of sampled model points within the
configured distance of sampled scan points. It is a registration diagnostic,
not confidence, independent accuracy or a complete quality report.

The heatmap starts green at zero distance, passes through yellow and ends red
at the configured maximum (10 cm by default). It measures unsigned scan-to-triangle distances for one displayed
candidate and geometry role. Switching candidates disables that heatmap;
generate another with `view --candidate N --heatmap` or a browser comparison job.
The initial embedded heatmap uses reproducibly selected preview records; regional
jobs can query the high-resolution index. Coverage-aware model-to-scan comparison
remains a follow-up.

Transforms follow SemDT conventions:

```text
p_scan = scan_T_root @ p_root
p_root = inverse(scan_T_root) @ p_scan
```

These are homogeneous column vectors. The browser transforms the model into
scan coordinates, then subtracts a recorded common display origin before using
float32 buffers. The footer reports that origin. Accepted transforms are stored
separately from proposals. Running `align` again clears previous acceptance and
replaces candidates as an undoable revision; previous candidates and acceptance
remain in `history.json`.

`world.npz` and `scan.npz` are frozen baselines. `project.json` records source
identity, scale and hashes; candidate JSON retains settings. Derived files are
written inside the project. The tool does not edit the imported URDF, meshes or
Python world. Inspection and registration history is implemented. Geometry refinements and
source copies are now available through the reviewed workflow below.

The local server serves artifacts, regional queries and reversible inspection
actions on loopback. It exposes the selected
project directory, including snapshot files; it is not an authenticated sharing
service. Stop it with Ctrl+C. Generated projects are Git-ignored.

## Development

The focused tests include registration recovery, baseline persistence, actual
URDF geometry origins, point-to-triangle distances, regional boundary/provenance
checks, HTTP byte-layout decoding and bounded overview access. E57 integration
tests skip unless `SCAN_REFINER_E57_PYTHON` names an existing reader interpreter.
Browser-control regression tests use an existing Node executable from PATH or
`SCAN_REFINER_NODE`; they install no JavaScript packages. They exercise actual
Plotly-encoded figure data and delayed renderer operations without a graphical
browser. They skip when no Node executable is available.

```bash
free -m
ulimit -v 8388608
/home/pmania/venvs/cram-env/bin/python -m pytest \
  -c scripts/semantic-digital-twin-scan-refiner/pyproject.toml \
  scripts/semantic-digital-twin-scan-refiner/tests --orm-build never -q
```

For the optional local E57 integration tests, prefix the pytest command with
`SCAN_REFINER_E57_PYTHON=/home/pmania/.cache/semantic_digital_twin/apartment-e57-reader/bin/python`.

Tests and fixtures live here as requested. Existing package CI selects `test/`
directories explicitly, so adding this test command to CI is pending until
changes outside this directory are in scope. Format Python using the repository's
`krrood-format` pre-commit hook and check with `krrood check --all-lines`.

See [doc/VIDEO_WORKFLOW.md](doc/VIDEO_WORKFLOW.md) for the reusable guided-video
recording method and [UI_DESIGN.md](UI_DESIGN.md) for the accepted UI direction.

See [DESIGN.md](DESIGN.md) for the architecture and [ROADMAP.md](ROADMAP.md) for
accepted priorities and deferred work.


## Directional comparison results

After a new comparison completes, **Jobs → Show heatmap** opens **Inspect →
Results**. The panel provides two complementary summaries:

- **Scan → model:** exact unsigned point-to-triangle distance, assigned to the
  nearest body's triangle. Per-body median/P90 and record count describe only
  those assigned records. Nearest geometry is not a semantic correspondence.
- **Model → scan:** reproducible area-distributed surface samples within each
  body, queried against the exact evaluated scan records. Samples outside the
  frozen oriented crop are excluded. Each body receives a share of the bounded
  model-sample budget; the global support fraction weights samples by the area
  they represent. It estimates supported surface area within the crop.

Use **Model surface sample limit** independently of the scan evaluation limit.
Small crops can retain few or no model samples. Counts are shown explicitly.
Both budgets and the declared tolerance affect support; this is not a visibility
or occupancy test. Missing returns can result from occlusion, material, sparse
sampling, or wrong/missing geometry. The report does not classify those causes.
No assigned scan records means unknown residuals, and no model samples means
unknown support; neither is presented as zero error.

Filter/sort body results, or include unsampled bodies to inspect scope exclusions.
**Download report** saves a typed JSON report with body metrics and frozen input
fingerprints, crop frame, placement and budgets. Numeric arrays and original
record identities remain beside it in the job directory. Source baselines remain
unchanged. Stale reports are blocked; Undo and saved views restore valid cached
reports together with their comparison. Jobs created before this increment keep
working as scan-only heatmaps; recompute to obtain directional results.

## Constrained body translation

Select a body and a tight scan crop in **Inspect**, then open **Refine**. Choose
visual or collision fitting geometry, explicitly unlock model-root X/Y/Z axes,
and declare per-axis movement bounds, initial association distance and record
budget. These axes are independent of the inspection frame's extra yaw.
Rotation and dimensions stay fixed.

**Propose translation** runs in the existing bounded/cancellable worker. The job
freezes the effective geometry revision, placement, crop and constraints. It
associates original records within the initial point-to-body gate, then updates
closest-surface correspondences over that same record set. Robust point-to-plane
fitting preserves axis locks and absolute movement bounds. Exact unsigned
point-to-triangle distances evaluate before/after improvement on the same records.
These are in-sample metrics, not independent accuracy or semantic correspondence.
Too few records, unconstrained axes, saturated bounds, unfinished optimization
and insufficient improvement prevent application.

In **Jobs → Review translation**, inspect the delta and metrics. The temporary
preview shows the proposed geometry in orange and the current pose as a blue
ghost, with the associated original scan records. Toggle current/proposed or
dismiss it. Previewing does not create a model revision. Leaving Refine or using
Undo clears the preview; applying and restoring project state serialize against
history traversal. Scan colouring from an incompatible model revision is disabled.

**Apply snapshot override** moves both geometry roles of the selected body and
creates one Undo revision. It does not move other bodies, including attached
bodies; the frozen snapshot currently lacks parent/child topology. It does not
edit the source URDF/assets or rebuild the live SemDT. Inspect the scope before
application. All geometry-derived analyses and proposals are invalidated on any
override change. Undo/Redo restores the prior effective geometry and validity of
compatible cached analyses. Worker inputs retain their submitted overrides even
if the user edits and undoes the model while a job runs.

Baselines stay in world.npz; cumulative root-frame translations and their source
job IDs live in the project journal. Each job retains its request, fingerprint,
submitted overrides, proposal.json, associated record IDs and resulting distances.
Source application, geometry export, actual world reload and joint/semantic
validation remain future work. New box/cylinder/simple-object fitting remains an
explicit requested follow-up.

## Verified box dimensions

In **Refine → Proposal type → Box dimensions**, the UI checks every present
geometry role of the selected body. Eligibility uses the actual combined surface
for each body/role, not whichever mesh part happens to be last. Closed boxes are
supported in model-root or verified geometric box axes. The latter recover the
actual surface directions and preserve orientation during resizing. Beveled,
skewed, open and compound shapes are rejected; an enclosing bounding box never
substitutes for the surface. Both roles must share compatible box directions.
GET `/api/body-dimensions?body=<qualified-name>&role=visual&frame=box` returns
typed eligibility, current bounds and their rigid box-to-root frame. Omitting
`frame` retains the original root-axis interpretation.

Unlock X/Y/Z extents and choose a reference independently for each axis:

- **Keep center** preserves the current box center and requires enough assigned
  returns on both opposing faces.
- **Fix minimum face** changes the maximum face and requires its scan returns.
- **Fix maximum face** changes the minimum face and requires its scan returns.

A fixed face is an assumption the user declares, not an observed landmark inferred
by this tool. Choose **Dimension axes → Verified box axes** (the UI default) or
**Model root axes**. Geometric axes are selected consistently toward the closest
root directions; they are not assumed to be the original URDF-local axes. Preview
labels identify X/Y/Z in red/green/blue. The frame is independent of registration
or the inspection frame's extra yaw. Locked extents and their face
positions stay fixed. The bound is the *total extent change*: with a fixed center,
a 40 mm extent increase moves each face by 20 mm.

The worker retains an initial point-to-box association, estimates face positions
with medians, and compares exact before/after triangle distances on those same
original records. Required-face counts explain rejections. Positive size floors,
change bounds, evidence counts and a minimum capped-mean residual improvement
prevent application when insufficient. These in-sample checks are not a proof of
semantic correspondence, visibility or independently measured accuracy.

Review and preview use the existing job and ghost workflow. Applying a dimensional
proposal stores a positive stretch/offset mapping, shared by both roles.
Rotated-box stretches retain the complete root-space matrix; legacy root-axis
scales remain compatible.
Existing visual/collision differences remain; their margins scale with the body.
Resizing and prior/later root translations compose explicitly. Undo/Redo and
reload reconstruct baseline geometry plus the stored mappings without cumulative
render-buffer scaling. Other bodies remain fixed. The geometry fingerprint and
submitted worker inputs retain both pose and dimensional revisions.

**Apartment audit:** 16 of 102 frozen geometric bodies pass the verified oriented
box checks; see `projects/apartment/oriented-dimension-eligibility.json`. The
earlier root-only audit remains in `dimension-eligibility.json` (0/102). The
waterfall is slightly skewed and the cooktop has non-box face directions, so both
remain ineligible. No apartment overrides were applied. Source-parametric edits,
assembly propagation and in-place source updates remain follow-up work.

Guided demonstrations are indexed in [doc/FEATURE_VIDEOS.md](doc/FEATURE_VIDEOS.md).


## Fast overview and reviewed source copies

**World overview** and **Return to scan overview** restore the embedded frozen
preview directly. They clear the dense region/clip/section/comparison state and
save an Undo revision. They do not query the dense scan index. Local regional
loads retain their existing original-record budgets and coordinate frames.

In **Refine → Review & export source copy**, review accepted changes, then write
and download a validated independent source bundle. URDFs receive native copied
parameters and OBJ assets; Python-defined worlds receive a reloadable provider
recipe. Source reload must match the accepted geometry and preserve topology,
joint parameters and semantic membership. Source originals and project geometry
history are retained. See [doc/SOURCE_APPLICATION.md](doc/SOURCE_APPLICATION.md)
for outputs, CLI commands, supported edits and validation limits.

## Agent access

A typed local API and CLI now share the browser's project, worker manager and Undo
journal. Start `serve_agent.py` (inspect mode by default), then use `agent_cli.py`
or `scan_refiner.agent.client.AgentClient`. See
[AGENT_INTERFACE.md](doc/AGENT_INTERFACE.md) for operations, authority, coordinate
frames, evidence images and the tested Apply/Undo/Redo/source-export workflow.
No MCP dependency is required; the optional host adapter is deferred.


## Complete apartment workflow video

[Watch the 2 min 16 s apartment use-case](http://127.0.0.1:8815/videos/apartment-usecase.mp4)
or open the [video gallery](http://127.0.0.1:8815/videos/). It demonstrates actual
registration, dense scan inspection, before/after heatmaps, a constrained correction,
Undo/Redo and validated source export. It uses a disposable model with a clearly
disclosed panel offset. See [the use-case guide](doc/APARTMENT_USE_CASE_VIDEO.md).
