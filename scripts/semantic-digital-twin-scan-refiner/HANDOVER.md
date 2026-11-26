# Current handoff — revised walkthrough on apartment handles (2026-11-26)

The [Cabinet 3 case](doc/APARTMENT_HANDLE_CASE.md) begins with the whole apartment
scan and model unaligned, then records the automatic placement search before
dense crop, Codex proposal, paired 3D heatmaps, Accept and visible Undo.
A real pair of door handles is proposed to move together
by 20 mm along world +Y, with both heights and the fixed doors preserved.
The pending local review is on port 8838. The working apartment source has
not been edited.

# Previous handoff — kinematic outliner (2026-11-23)

The [apartment outliner demo](http://127.0.0.1:8831/) uses the existing accepted
registration and dense scan index. Expand **Apartment Root → Kitchen Root** or
search for **Side A** or **Side B**. Click a branch label to focus its descendants,
or use the mesh icon to focus its own body. **Groups → Kitchen island** still
provides the named selection of 39 bodies.

URDF imports and Python providers both freeze parent links in `world.npz`.
The directly modelled `KitchenEnvironment` is covered by the import tests.
Earlier snapshots show the existing body list. To capture topology, initialize
a new project from the original source. The apartment demo was made in a separate
project after checking the reloaded source against the prior frozen geometry.

The copied context for a branch includes its root, exact geometric members, and
combined crop. Chromium checked selection and Undo on the apartment, and the
agent retrieved 1,000 indexed points from the copied Side A crop.

# Previous handoff — selectable inspection assemblies (2026-11-22)

Try **Assemblies → Kitchen island** at <http://127.0.0.1:8829/>. Selecting it
auto-focuses the combined bounds of 39 bodies and prepares their scan crop.
The arrow expands the explicit member list. **Copy investigation context**
includes the whole assembly and the same directly reusable region query.

New provider imports freeze rooted semantic branches alongside their geometry.
This apartment project loads the bare URDF, so its saved group was seeded
from the verified KitchenIsland membership in the matching apartment provider.
For another plain URDF, create or remove groups in the Region panel. These
decisions use inspection Undo/Redo and do not change geometry revisions.

Chromium verified full group focus, expanding and selecting a member, creation
and Undo of a custom group, and returning 1,000 dense scan points through the
copied crop. A real provider snapshot round-tripped its semantic groups. Existing
agent action fingerprints also matched after reopening pre-assembly history.

# Previous handoff — paired heatmaps in the 3D review (2026-11-22)

The disposable apartment viewer at <http://127.0.0.1:8827/> has the pending
“Scan-fitted island panels · 3D heatmap walkthrough” proposal and its completed
comparison. In **Refine**, select it under **Agent proposal**, then choose
**Show paired heatmaps**. For a new linked crop, **Compute paired heatmaps**
starts the bounded worker. Toggle **Original / Proposed** to switch geometry and
point colours in the regular 3D scene. Orbit, zoom and choose geometry layers.

The worker samples the crop once, compares to all visual triangles for both
states and publishes one record array, two residual arrays and a short summary.
The viewer uses a shared 0–40 mm colour scale. It includes fixed neighbours in
the distance calculation, so the median 12.2→4.1 mm and 85.7→89.4% within 20 mm
on this copy differ from the earlier island-only video. The proposal stays pending.

Chromium verified real orbit, unchanged camera and ranges across the toggle,
50,000 matching point identities, changed colours and restoration on Original.
The focused Python/Node suite passed 41 tests, excluding the older live-camera
fixture whose cross-VM comparison still awaits the previously requested change.

# Previous handoff — grouped translation and interactive review (2026-11-21)

[Live functionality video (1:11.5)](http://127.0.0.1:8815/videos/agent-review-live.mp4)
demonstrates selection handoff, a real CLI submission, camera orbit, comparison
on the same 50,000 points, grouped acceptance and one Undo. Recording exposed
a stale layout camera after focus and orbit. Geometry redraw now reads the live
camera, and the final take verified the correction in Chromium. The test viewer
on port 8823 was regenerated with the fix.

Roadmap increments 2 and 3 now have a working translation slice. Codex submits
explicit root-frame body deltas with `submit_change_set`. The revision and every
body/delta are rechecked before acceptance. The shared journal applies the group
as one action and supports Undo/Redo. An optional region links to dense evidence.
Kinematic membership and shared constraints are expressed through explicit edits.

In **Refine → Agent proposal**, select a submission. Review its affected bodies,
reasoning and assumptions. Load the linked crop, orbit or change camera presets,
then use **Original / Proposed** in the same scene. The viewport routes the same
RGB points and visible layers through both states. Accept/Reject requires the
configured refine scope. Previous single-body fit controls are under Advanced.
Paired residual heatmaps and grouped dimensions/rotations remain follow-ups.

The disposable apartment review is at <http://127.0.0.1:8823/>. The pending
“Walkthrough: move the three island panels together” proposal is illustrative.
Port 8823 is scoped to the three named island panels. The original apartment
is on port 8815 and the selection-only demo is on port 8822.

The full refiner suite passed 190 tests. The focused group, redo and native
source reload cases passed after the final history-status update. Chromium
verified matched top-view ranges and scan identities, orbit preservation on
toggle, three single displacements, and one Apply/Undo. The URDF fixture also
verified export of a fixed parent/child group without double movement.

Next investigate the real apartment handle-alignment case. Use selection
context, inspect scan quality and drawer geometry, compute proposed sideways
translations with heights fixed, and submit one group for review.

# Previous handoff — investigation context (2026-11-20)

In roadmap increment 1, **Inspect → Region → Copy investigation context** now
copies the selected body or oriented crop into a `semdt-investigation-v1` handoff
for the existing conversation. Its nested `region_query` works directly with
`agent_cli.py create_region`. The revision follows the displayed model and the
server rejects stale copied requests before retaining a region. A selectable
text field appears if the browser denies clipboard access.

All 187 refiner tests passed with Node and E57 enabled using the configured
interpreter and an 8 GiB test memory cap. Real Chromium QA copied
`apartment/handle_cab6_t` and the agent client reopened its crop and retrieved
1,000 original records from the dense index. Try the disposable apartment view
at <http://127.0.0.1:8822/> while the local test server is running. It shares
the frozen scan index. The main listener on 8815 still serves its earlier build.

# Agreed direction — agent-assisted roadmap (2026-11-20)

The user accepted keeping the agent in this conversation or Codex CLI. The viewer
is for inspecting errors, selecting targets and reviewing proposals. Most fitting
and investigation should be delegated in plain language. Follow KISS: no separate
viewer chat, expanded manual wizard or general assembly solver as a prerequisite.

Read the **Current priority and handoff** section in ROADMAP.md first. After
the selection handoff, the roadmap calls for explicit multi-object proposals,
focused review and a real apartment-handle acceptance case.

README.md, DESIGN.md, UI_DESIGN.md and doc/AGENT_INTERFACE.md now point to this
direction. Older design sketches and dated handover entries are historical;
ROADMAP.md sets implementation priority. The November 9 handover is an archive.
Preserve existing staged and unstaged Git work.

# Previous increment — complete apartment use-case video (2026-11-20)

Published a 135.75-second captioned apartment walkthrough, with highlighted
controls, pointer cues and 63 clickable chapters. Watch
<http://127.0.0.1:8815/videos/apartment-usecase.mp4> or use the gallery at
<http://127.0.0.1:8815/videos/>. The eight-video offline archive is
<http://127.0.0.1:8815/refiner-feature-videos-20261119.tar.gz>.

The recording follows real Apartment-Lab.e57 evidence through initially missing
XY/yaw registration, automatic alignment review/acceptance, dense scan inspection,
regional heatmaps, constrained panel correction, Apply/Undo/Redo and validated
native source export. A deliberate +30 mm world-Y panel offset in a disposable
URDF is disclosed in captions and a persistent badge. Production apartment source,
geometry and refinement history were not modified.

For the demonstrated local crop, median distance falls from 21.345585 mm to
0.827276 mm using the same original point records, crop, registration and geometry
role. These are local fitting diagnostics, not independent whole-world accuracy.
The native export reload validates 115 bodies and 114 joints.

Final H.264 video is 1440×1000 at 8 fps, 5,263,133 bytes. Full encoded-stream
decode completed without diagnostics; eight decoded frames were inspected.
Actual gallery chapter playback and eight video entries were checked in Chromium;
video/archive HTTP responses are 200. No application implementation changes or
new application test run were needed for this recording/documentation increment.

See doc/APARTMENT_USE_CASE_VIDEO.md for scope and reproduction artifacts.
Recorder, publisher, setup, verification and disposable project are retained in
the ignored projects/apartment-usecase-20261119-v1/ directory. Previous seven
feature videos remain in the gallery. Main viewer server stays on port 8815.
After resuming, the old listener returned empty responses. Restarted that owned
server in inspect mode with output in /tmp/refiner-main-8815.log (session 94436).
Video and gallery now return HTTP 200. Temporary recording browser is stopped.

# Previous increment — worker memory and foreground action audit (2026-11-19)

User requested a larger memory ceiling, loading feedback for Orient axes and an
audit of other slow actions. New analysis and source-copy workers now request
**16 GiB** instead of 4 GiB. ProcessMemory in scan_refiner/process_memory.py owns
the shared default and applies it within any lower inherited hard RLIMIT_AS.
This is necessary for workers launched by the memory-capped pytest process: an
unprivileged child cannot raise its inherited hard ceiling. Existing maximum_memory
constructor fields remain available. Running workers retain their existing limits.
A standalone process smoke check reported soft/hard limits (16,16) GiB.

Foreground loading now covers InspectionTools.change (Orient axes/crop-plan/
heatmap operations), orientFrame (including placement changes), scan/section/
measurement render, geometry/layer redraws, adding preview layers, return to
overview, focus/show-all, project refresh, refinement preview/application and
source review/export. Heatmap evidence download moved inside its visible change
operation. PlotUpdates.enqueue keeps expensive renderer work ordered and uses
ViewerActivity.run; ordinary mouse camera updates and automatic resize stay quiet.
Nested operations retain the outer title. Error/finally paths release their tokens.
Background registration/comparison/fitting still uses Jobs instead of blocking
inspection. No changes to geometry, registration algorithms, crops or source models.

Added tests for inherited/infinite/higher hard limits, shared defaults and
paint-before-work/error cleanup. **185 refiner tests passed** with Node/E57,
CRAM_ORM_BUILD=never and an 8 GiB test address-space cap. New/modified Python passed
repository krrood-format and Ruff; JavaScript syntax and git diff checks passed.
Existing Matplotlib Axes3D warning remains.

Real software Chromium QA on the disposable focus project: Orient axes immediately
showed its panel and retained the outer title through nested redraws; geometry
switching showed Switching geometry; subsequent real mouse drag changed the camera
and produced no loading activity or unhandled errors. Screenshot
/tmp/refiner-orient-loading.png was visually inspected. Script/log:
/tmp/verify_foreground_actions.py, /tmp/refiner-foreground-qa.log. Temporary browser
9237 and server 8820 stopped. Main http://127.0.0.1:8815/ serves regenerated controls;
no restart needed for new worker scripts. No production history edits, new
dependencies, commits or pushes. Current memory behavior is documented in
AGENT_INTERFACE.md and SOURCE_APPLICATION.md; UI_DESIGN.md records the action audit.

---

# Current fix — focused metric proportions (2026-11-19)

User correctly reported compressed height after the previous focus correction.
Equal numeric XYZ focus ranges were still rendered with Plotly's full-scene
aspectmode=data proportions. Browser measurement reproduced unit scales
[0.69879449, 0.57363953, 0.13094453], so Z was compressed by over 5x relative to X.

Focus now explicitly sets aspectmode=cube alongside its equal XYZ ranges; the
rendered unit scales are [0.37441965, 0.37441965, 0.37441965]. Show all explicitly
restores aspectmode=data with autorange; measured full-view unit scales were
[0.09425947, 0.09425947, 0.09425947]. Geometry and crop coordinates are unchanged.
Upstream aspect selection logic was checked against the primary source:
https://raw.githubusercontent.com/plotly/plotly.js/master/src/plots/gl3d/scene.js

Extended the passing focus regression with aspect-mode requirements before
fixing production code; the new assertion reproduced the failure. **21 related
viewer/frame/loading tests passed** across two sequential focused runs. The prior
full 180-test suite was not repeated for this two-line presentation correction.
git diff --check passed; no Python source/test files changed in this increment.

Real Chromium verified the old distortion, equal focused scales and correct Show
all scaling on the disposable focus QA project at world yaw 35 degrees, with zero
captured unhandled errors. Screenshot /tmp/refiner-focus-proportions.png was
visually inspected. Script/log: /tmp/verify_focus_proportions.py and
/tmp/refiner-focus-proportions.log. Browser 9237 and QA server 8819 stopped. Main
http://127.0.0.1:8815/ now serves the regenerated controls. No production geometry
or history edits, dependencies, commits or pushes.

---

# Current fix — complete selection focus (2026-11-19)

User reported selected objects being cut by Focus selection when the world is
rotated relative to view axes. Existing focus only assigned crop-axis ranges and
retained the old panned camera center/zoom. Updated controller.focusSelection to
wait for queued coordinate updates, include all current visual/collision vertices
of the selected body in displayed coordinates, and fit a cube based on the largest
XYZ extent with 10% additional span. Crop context is retained in the focus bounds;
other bodies are not included in the extent calculation. Scan crop inputs and
point selections are not rewritten.

Focus also resets the camera center to the volume center, fits its distance for
the viewport aspect ratio, and retains view direction/up/projection. The internal
camera relayout temporarily preserves InspectionTools.applying, preventing an
extra Move camera history action; the caller keeps its normal Focus selection
Undo action. Existing restoration/frame orientation callers retain their outer
applying state. Show all still restores automatic axis ranges.

Regression test uses a rotated long object, stale undersized crop, nonzero display
origin, a remote neighbor and a close/panned orthographic camera. It asserts full
containment, cubic ranges, camera recentering/zoom fitting, projection/up retention,
unchanged crop and suppression of duplicate programmatic-camera saves. Also checks
region-only focus without a selected body.

Real Chromium QA used a disposable apartment copy (projects/focus-qa-20261119).
Cabinet2's complete visual/collision geometry was contained in original scan axes
and world axes with 35-degree yaw; no captured unhandled errors. Screenshots
/tmp/refiner-focus-fixed.png and /tmp/refiner-focus-rotated.png were inspected.
Script/log: /tmp/verify_focus_selection.py, /tmp/refiner-focus-qa.log. The copied
project's startup was changed to cached preview after the first full-apartment
150k-point restoration exceeded the diagnostic polling timeout; production history
was not modified. Temporary browser 9237 and server 8818 are stopped.

Main viewer http://127.0.0.1:8815/ serves the updated embedded controls. No source
URDF/mesh changes, dependency changes, commits or pushes. All **180 refiner tests
passed** with Node/E57, CRAM_ORM_BUILD=never and an 8 GiB address-space cap.
Repository krrood-format, Ruff and git diff --check passed. The existing Matplotlib
Axes3D import warning remains. Port 8818 was confirmed unavailable after shutdown.

---

# Current increment — visible foreground loading (2026-11-19)

User found startup and Undo indistinguishable from another frozen viewer. Added a
centered viewport status panel for startup, Undo/Redo, saved-view restoration and
indexed-region loading. It names the outer operation and shows the current nested
step. Conflicting controls are inert until all foreground work finishes, retaining
their prior inert state. Startup failure shows a message and Reload viewer button.
Feedback is indeterminate; no artificial progress percentages are shown.

`ViewerActivity` lives in web/plot_buffers.js; the controller supplies it to
InspectionTools. Existing try/finally paths release operation tokens; aborted or
superseded regional loads also release their own tokens. The initial figure is
embedded in an inert HTML template. initializeViewer paints the loading panel
before prepareViewerFigure inserts the embedded scripts and starts WebGL. Viewer
creation still uses the existing Python generator; no Python API changed.

New tests cover nested status/locks, paint-before-work, rejection cleanup, startup
error feedback and deferred initial markup. **179 refiner tests passed**, with
Node and isolated E57 reader configured, CRAM_ORM_BUILD=never and an 8 GiB address
space cap. New Python test passed repository krrood-format and Ruff; diff check
passed. The existing Matplotlib Axes3D warning remains.

Real Chromium QA on a disposable apartment project observed startup feedback
before the scene existed, followed by normal unlocked interaction. With injected
network latency, Undo showed its panel immediately, retained it throughout the
restore and unlocked on completion, without captured unhandled errors. Screenshot
/tmp/refiner-undo-loading.png was visually inspected; script/log are
/tmp/verify_viewer_loading.py and /tmp/viewer-loading-qa.log. Original apartment
geometry/history were not changed by QA. Temporary Chromium 9237 and QA server
8817 are stopped (8817 confirmed unavailable). An initial attempt to bind 8816
failed because that port was occupied; it was not used for this verification.

Main workspace **http://127.0.0.1:8815/** now serves the rebuilt HTML with one
loading panel and one deferred figure, preserving existing figure/evidence data.
Rebuild helper /tmp/rebuild_refiner_page.py strips the inert template wrapper when
reusing an already-generated figure, avoiding nested templates on future rebuilds.
No commits, pushes or dependencies added.

---

# Current fix — 3D dragging and picking (2026-11-19)

User reported Firefox allowing initial scroll zoom, then freezing on left drag and
showing its slow-page warning. Reproduced on a disposable apartment copy in
software Chromium: a drag stalled the page; removing plotly_click listeners
prevented the stall. The selection callback could redraw while a drag began.

Added shared `SceneClicks` in web/plot_buffers.js, registered through
`PlotUpdates.registerClick`. It retains Plotly picks until mouse release, discards
a gesture after >5 CSS pixels of movement and cancels on window blur. Delivery is
deferred until the release event has finished. Both object selection and scan
measurement use it. Point picking on ordinary clicks remains available; camera
moves retain the existing Undo journal behavior.

Regression test covers premature Plotly clicks, drags, late click events,
release-time events, ordinary clicks, both registered consumers and blur. User
approved adding the new dispatcher method to the existing history test mock;
its history assertions are unchanged. All **177 refiner tests passed** with Node,
isolated E57 reader, CRAM_ORM_BUILD=never and an 8 GiB address-space cap. Formatter,
Ruff and git diff --check passed. Existing Matplotlib Axes3D warning remains.

Real browser QA on copied apartment: repeated timed mouse drags changed the live
camera, and wheel zoom after dragging worked for visual/collision/both/scan-only/
visual modes; zero captured unhandled rejections. Ordinary mesh clicks selected
an object. Firefox itself has not been automated. The disposable project is
projects/rotation-qa-20261119; it reuses the dense index by symlink, with independent
history. Diagnostic scripts/logs are in /tmp/*rotation*. Temporary Chromium 9237
and QA server 8816 are stopped. Main workspace 8815 remains available.

Rebuilt the main embedded HTML shell around its existing figure/evidence data;
GET / on 8815 serves the new guard. Original apartment geometry, registration and
history are unchanged. A frozen Firefox tab may need closing and reopening, or a
hard refresh, to load the updated script. No commits or pushes.

---

# Current increment — shared agent operations

2026-11-18, branch `semdt-world-refiner`. User accepted implementing the agent-facing
interface. Everything remains under this refiner directory; no commits/pushes or
original apartment source edits.

**Current viewer and agent API: <http://127.0.0.1:8815/>**, server session 31257.
The old 8813 process was no longer present. The server defaults to inspect mode;
no apartment geometry, registration or history decisions were changed.
The existing bundle and video gallery are available at the same paths on 8815.
The HTML shell was rebuilt around its existing figure and numeric scan distances.

Read [doc/AGENT_INTERFACE.md](doc/AGENT_INTERFACE.md) for exact usage and limits.
New entry points: `serve_agent.py` and `agent_cli.py`. Typed interfaces, DTOs, client,
service, strict repository-serializer codec, finite registry, renderer and HTTP
handler live in `scan_refiner/agent/`.

Implemented: project/object inspection (optional verified source metadata), opaque
reusable oriented regions, explicit preview/dense evidence, bounded HTML/PNG views,
unapplied registration-candidate and heatmap renders, existing registration/fit/
comparison workers, proposal review, revision-checked scoped Apply/acceptance,
explanations and exact-action Undo/Redo, source review/export. Browser polling
refreshes agent geometry changes after local operations finish without another
journal action. The API shares the server's existing JobManager and action lock.

Native URDF end-to-end test: sample a known 20 mm discrepancy, fit through HTTP,
review measured proposal, Apply, Undo, Redo, review source and export a validated
URDF/bundle/archive while preserving original source bytes. Rendering also passed
manual Chromium QA with 2,000 visible scan points and an orthographic overlay;
artifacts are in ignored `projects/agent-qa-2/`. Snap Chromium cannot read host
`/tmp` files and needs a normal home-directory project path. Its subprocess needs
permission to run outside this session's sandbox.

Approvals: user allowed narrow request-validation/InvalidEvidence handling at HTTP
boundary, and corrected a new test's expected exception to Pydantic ValidationError
(the schema was already rejecting unknown constraints). User asked why MCP SDK is
needed; explained it is only for automatic host protocol integration and deferred
it. **No dependency added, no MCP adapter claimed or installed.** CLI/API are usable
from this agent's local command tools now. Optional SDK/host integration is later.

Source validation/export now shares `SourceProcess` between browser and agent. It
uses the existing 4 GiB child task and 240-second timeout. Original source loading
for optional metadata remains synchronous and opt-in.

Known scope: translation and verified box dimension fitting only; shared-height,
centering and new box/cylinder/object fitting remain follow-ups. Unknown constraint
fields are rejected. Fit residuals are in-sample diagnostics. Agent Undo refuses
intervening human inspection actions as well as geometry changes. This is a local
loopback interface, not a multi-user server. PNG artifacts currently retain their
Chromium profile directory; artifact retention/pruning is future housekeeping.

Validation: all **176 refiner tests passed** with Node and the isolated E57 reader
configured, CRAM_ORM_BUILD=never, one package invocation and an 8 GiB address-space
cap. Repo pre-commit krrood-format passed; targeted Ruff and git diff --check passed.
The only pytest warning is the existing Matplotlib Axes3D import warning. Live typed
client inspection on 8815 found `apartment/cooktop` and loaded an explicit dense
10,000-point region (`ea27e26a1ae64b9b8468d7844b34dae0`) without changing history.
Current main agent revision: 38fca329e9b32537eb7ed833f2c13bce3ff23602ba5a73ae6b96cb8adc460a57.
No temporary servers or test workers remain; only the main workspace is intended
to remain running. MCP integration was not installed/configured in this session.

---

# Current increment — cached overview and reviewed source-copy application

2026-11-18, branch `semdt-world-refiner`. User requested roadmap items 1 and 2:
fast overview and applying accepted refinements to reloadable source descriptions.
Work remains inside this directory. Original URDF/assets and project baselines
are unchanged; no commits or pushes. Native source copies are deliberately separate.

**Current main viewer: <http://127.0.0.1:8813/>**, exec session 10484.
**Validated apartment bundle:**
<http://127.0.0.1:8813/source-reviews/19e1f3525cbc4dce8028b66f8cf3c98f/world.tar.gz>.
This is about 24.5 MB; its bundle has 115 bodies and 114 joints. No additional
apartment overrides were accepted: the main history remains four actions, zero
translation overrides and zero dimension overrides. Its URI source still exactly
matches the frozen baseline. Parser-only URDF loading supplies zero annotations;
this is reported rather than silently running a new reasoning pipeline.

Temporary QA server 8814 (session 63614) and Chromium 9235 are stopped after QA.
The older 8810 session was no longer registered. Leave only main 8813 running.
The existing video gallery remains served on 8813/videos/; recordings predate this
increment and do not demonstrate the new source-copy controls.

## Implemented

- `InspectionController.loadOverview` restores the embedded preview directly,
  cancels an obsolete pending regional fetch, clears loaded region/frame,
  comparison/clip/section/selection/focus state, renders and saves an Undo action.
  It never calls `/api/region`. Both overview buttons share this operation.
  Clearing saved regional bounds also prevents global dense reads on reload/Redo.
- Refine has **Review source changes**, **Write validated copy & export** and a
  completed bundle download. Only accepted overrides are included. Source-review
  applicability follows accepted geometry revisions, independently of display
  settings. Restoring reviewed geometry restores its completed download.
- `source_application.py` captures a frozen review/source digest and source graph,
  reloads the original and compares all labelled geometry against the baseline,
  applies the accepted geometry to a fresh live world and emits a source copy.
  Publication checks the review and source again. Outputs do not overwrite originals
  or rebase project history. A new reviewed source revision needs a new bundle.
- URDF copies retain original names and non-geometric XML, update native box
  parameters and independent OBJ assets, and adjust fixed/prismatic placements.
  Outgoing offsets compensate unedited children. Moving incoming hinges, translated
  world roots and resizing inertial bodies are explicitly unsupported.
- Python-defined worlds get a `ReviewedWorldSource` recipe around their original
  provider; arbitrary Python code is not rewritten. Original provider code and
  dependencies remain required. Exported recipes reject changed entry-module
  content/recipes or changed kinematic/semantic membership on later reloads.
- Live-world reload compares geometry, frame names, graph edges, joint types,
  axes, DOF names/limits and semantic classes/referenced frames. A real
  NaturalLanguageDescription fixture verifies semantic data survives the recipe.
  These are consistency checks, not ontology/physics/collision validation.
- Bundle: refined.urdf (URDF sources), assets/, refined_world.py, refinement.json,
  review.json, labelled baseline.npz/world.npz, visual scene.glb, collision.glb,
  manifest.json. Directory publication and archive replacement are atomic after
  validation. This uses native trimesh GLBs; the manifest is not Warsaw's schema.
  The generated live-world provider can be fed to Warsaw when that exporter exists.
- `source_task.py` provides explicit review/publish CLI actions. Local POST
  `/api/source/review` and `/api/source/publish` invoke it with the configured Python,
  a 4 GiB worker limit, a same-origin guard and the existing project action lock.
  No arbitrary source or destination path is accepted over HTTP.

## Dependencies and user approvals

The initial new overview test had a VM setup error. User approved correcting its
setup; assertions remained intact. The real semantic fixture initially could not
import because the configured environment lacked dependencies already declared in
probabilistic_model/pyproject.toml. User questioned simplifying it; the real test
was retained. The installation command was approved and installed array-api-compat
1.16.0, array-api-extra 0.11.5 and causal-learn 0.1.4.8, plus their missing transitive
requirements (statsmodels, momentchi2, patsy, formulaic, interface-meta). No dependency
files changed. User then approved correcting the fixture's qualified-name lookup;
it uses PrefixedName(prefix='test', name='box'). The real annotation test now passes.

## Verification

157 tests passed in the full focused refiner run with E57 and Node enabled.
An additional provider-source provenance regression was reproduced and fixed;
the two provider tests pass with the reload guard. Source/overview/control checks
also passed separately after atomic-archive and cleanup changes. Final checks are
recorded below when complete. Every test run used CRAM_ORM_BUILD=never, a memory
cap and the configured interpreter. Existing Axes3D warning is unrelated.

Browser QA in disposable projects/source-qa-20261118/ used real apartment evidence:
50,000 regional records → 20,000 cached overview records, zero new dense queries;
Undo restored 50k via a local query and Redo restored the preview with zero queries.
Source review/publish/download worked: 115 bodies/114 joints, no browser errors.
The main project's four-action history and empty override lists were preserved.
Failed development export attempts remain in ignored staging directories under
source-reviews; only completed bundles/results are advertised as ready.

Final verification: the post-guard source/overview/control run passed **15 tests**;
combined with the earlier full run this covers **158 distinct tests**. Pre-commit
formatting, Ruff and diff checks passed. Main history remains four actions with
empty geometry override lists. QA browser and temporary server are stopped;
main 8813 is running. The output download and source controls return HTTP 200.

## Usage and next scope

See doc/SOURCE_APPLICATION.md for the UI, CLI, generated provider import and
relative-resource resolver. The copied URDF is portable with its assets; use the
bundled RefinedWorld provider, or explicitly pass bundle_resolver to URDFParser.
Python recipes remain dependent on their original provider code/environment.

Remaining: source adoption/rebasing/in-place transactions, source hierarchy and
assembly propagation, moving-hinge/mass policies, richer semantic/physical checks,
rotation/shared alignment constraints, gravity/masks/coverage improvements.
Remember requested new missing box/cylinder/simple-object fitting. Repeated scans
remain outside scope. Dedicated Review/Export stages are still planned; the first
source-copy workflow is available inside Refine.

---

# Historical increment notes — earlier ports and limitations below

# Current increment — verified oriented boxes and complete video collection

2026-11-16, branch `semdt-world-refiner`. Work remains entirely in this directory.
User asked to continue the roadmap and make videos covering implemented features.
No dependencies, commits, pushes, source URDF/mesh edits or apartment overrides.

**Apartment workspace: <http://127.0.0.1:8810/>**, exec session 99252.
**Video gallery: <http://127.0.0.1:8810/videos/>**.
**Portable video archive:**
<http://127.0.0.1:8810/refiner-feature-videos-20261116.tar.gz>.
Older apartment backend 8808 is stopped. Temporary services 8809, 8811, 8812 and
software Chromium 9234 are stopped after QA. Leave only apartment 8810 running.
Apartment history still has four actions, with no translation/dimension overrides.

## Implemented

- Dimension frame selector: verified geometric box axes (UI default) or model
  root axes. Existing API requests retain root axes unless `frame_mode="box"`.
- `OrientedBoxInspection` verifies actual orthogonal faces, closed surface,
  area and volume in recovered axes. It uses existing trimesh oriented bounds,
  then verifies the real shape; an enclosing box never establishes eligibility.
  Canonical signed axes are chosen consistently toward root directions, preserving
  identities across triangle reorder and extent-order changes.
- Every whole geometry role must be a box with compatible directions. Non-box,
  skewed, open, beveled and compound surfaces remain excluded. The recovered
  frame is geometric, not assumed to be a source URDF-local frame.
- `DimensionFitter` fits face medians in verified box axes, retaining the existing
  extent locks, declared center/min/max anchors, required-face evidence, positive
  sizes, movement bounds and same-record triangle-distance improvement checks.
  Proposal/frame/bounds are frozen. Root-space deformation keeps the full stretch
  matrix, including off-diagonal terms, preserving physical orientation.
- Both role geometries receive the common stretch; their size differences remain.
  Prior/later translations compose. Frozen job geometry, Apply, Undo/Redo and
  viewer reload retain complete stretch matrices. Legacy diagonal fingerprints
  remain compatible by omitting empty optional matrix data from signatures.
- Metadata endpoint accepts `frame=root|box`; invalid or repeated values return
  400. The inspector reports actual frame extents. Review restores frame choices
  and renders red/green/blue X/Y/Z labels on the temporary proposal.
- No source application/export or assembly propagation added. Attached bodies
  remain fixed; only the selected body receives a snapshot override.

## Actual apartment audit

Read-only `projects/apartment/oriented-dimension-eligibility.json` accepts **16 of
102 geometric bodies**, versus the historical root-only audit's 0/102. Countertop,
several wall/panel/door surfaces and one handle qualify. The waterfall is slightly
skewed and the cooktop has non-box face directions: both remain excluded. No
apartment sizing proposals were automatically applied. Do not loosen these checks.

## Verification

140 tests passed in the focused refiner run with E57 and Node enabled. Three
additional HTTP frame tests passed in the 19-test service run (143 tests total
across the focused suite). Pre-commit `krrood-format`, Ruff and diff checks passed.
New cases cover oriented locks/anchors, frame stability, incompatible role axes,
non-box rejection, invalid stretch mappings, full browser matrix application,
job/apply/Undo composition and frozen worker geometry after later Undo.

Real software Chromium verified a rotated synthetic width discrepancy:
visual 1000→1040 mm; collision 980→1019.2 mm; center, depth, height and orientation
retained. Regenerating an applied page then reloading did not double-scale:
max vertex distance was 1.26554336 m, then 1.23288281 m after Undo, matching the
1040/1000 mm box diagonals. Geometry modes gave 1/3/0/2 visible meshes on the
oriented demo; no empty meshes or browser errors; actual mouse drag rotated it.
VirtualGL remains user-side QA.

## Videos

Seven clips total 350 seconds, H.264, 1440×1000; archive about 9.1 MB. Four new
recordings cover representations/navigation, automatic registration/cancellation,
current dense inspection and oriented dimensions. Three earlier verified clips
cover oriented crops/heatmap jobs, directional results/report export and constrained
translation. The crop clip is trimmed before its obsolete roadmap closing caption.
All use visible highlights, pointer/click pulses and captions. Synthetic scenarios
are labeled; apartment demonstrations are inspection only. Native gallery playback
and chapter seeking worked without browser errors. See doc/FEATURE_VIDEOS.md.

Scripts, chapter metadata and decoded frames remain in `projects/*-tour-20261116/`.
The gallery builder is archived at `projects/apartment/build_video_gallery.py`.
The inspection caption at 9.5 s was corrected to match its actual scan-axis crop;
the first take is retained separately. Unfinished navigation takes were replaced
and are not published. The finished navigation take uses the small synthetic scene.

## Next scope

A whole-scan overview request currently reads the dense index across global bounds;
the apartment can therefore be slow. Prefer tight local crops. A small next fix
should restore the frozen preview directly for overview; see ROADMAP.md.
Then continue source-backed/local-parametric dimensions, captured hierarchy and
assembly propagation, reviewable source application, actual SemDT reload and
semantic/articulation validation, export/portable review bundles. Rotation, shared
constraints, gravity/masks and visibility/density-aware coverage remain pending.
**Remember requested missing-object box/cylinder/simple-object fitting** after
existing-body refinement; repeated scans remain out of scope.

---

# Historical increment notes

The following sections describe earlier snapshots; current scope/ports are above.

# Current increment — verified box-dimension proposals

2026-11-15, `semdt-world-refiner`; all source changes remain in this directory.
User asked to continue without review for now. No apartment overrides were applied.
No dependencies added, source URDF/mesh edits, commits or pushes.

**Current apartment workspace: <http://127.0.0.1:8808/>**, session 23862.
The older 8804 backend is stopped. Temporary servers 8806/8807 and software
Chromium 9233 are stopped; only apartment 8808 remains running.
Apartment history remains four actions;
translation and dimensional override lists are empty. Existing comparison
`e51faebc475f4f3ca98b187de6f61e23` still succeeds under the extended fingerprint.
The prior rejected Cooktop translation diagnostic remains unapplied.

## Implemented

- Refine → Proposal type selects Translation or Box dimensions. The existing
  four-stage test contract is unchanged in this increment.
- `dimensions.py`: verified closed, root-aligned boxes; independent X/Y/Z extent
  locks; fixed-center or minimum/maximum-face anchors; required-face return counts;
  median face estimates; absolute extent-change bounds and positive size floors.
  Before/after exact triangle metrics use the same admitted original scan records.
  These are in-sample diagnostics, not independent accuracy or visibility proof.
- Centered sizing requires both opposing faces. A fixed minimum/maximum face
  requires its moving opposite face. The fixed face is an explicit user assumption.
  Anchors use root coordinates independently of scanner yaw and inspection yaw.
- Eligibility checks each whole body/role surface. `WorldSnapshot.body_mesh`
  combines all parts so a compound body cannot pass by checking its last box part.
  Rotated, beveled, open or compound surfaces are not replaced by their AABBs.
- GET `/api/body-dimensions?body=<qualified-name>&role=visual` returns typed
  `BodyDimensionInfo` with actual bounds or the reason sizing is unavailable.
  The UI shows eligibility before enabling a dimensional request.
- `JobKind.DIMENSIONS`, typed constraints/proposals, existing bounded job process,
  review/preview/application endpoints and staleness guards. Both role surfaces
  receive one common positive diagonal deformation, preserving their differences;
  role-specific margins scale rather than being forced equal.
- `BodyDimensions` stores cumulative scale/offset relative to baseline, before
  the separate root translation. Applying a current-frame resize composes both
  earlier dimensions and translations. Later root translations still act after it.
  Immutable world.npz is unchanged; Undo/Redo captures both geometry revision types.
- Fingerprints and frozen worker inputs include dimensional overrides. Empty
  fields are omitted from signatures to preserve existing result compatibility.
  A fingerprint reads one complete state snapshot, never mixed pose/size revisions.
- Browser baseline coordinates subtract embedded pose and invert embedded dimension
  changes once. Current overrides and temporary dimension previews compose before
  registration. Heatmaps/statistics are hidden for incompatible model revisions.
  Preview uses the same blue-original/orange-proposal workflow and original records.
  Review restores saved fit role, limits, budget and axis/anchor choices into controls.

## Actual apartment finding — important limitation

The read-only audit found **0 eligible bodies out of 102 frozen geometric bodies**.
See `projects/apartment/dimension-eligibility.json`. The actual assets are rotated,
beveled, open or compound. For example, the right waterfall is a rotated thin box;
Cooktop surfaces include additional non-axis face directions. This is not treated
as scan evidence for a new bounding box. The UI explicitly rejects those shapes.
Do not claim the new dimension fitter can refine this apartment yet.

Next dimension work needs verified oriented box frames and/or captured source
primitive/local-frame parameters. Do not loosen eligibility into arbitrary AABB
replacement. New missing box/cylinder/object fitting remains requested follow-up.

## Verification and demonstration

**126 focused tests passed**, including E57 and Node. Formatting through the
pre-commit hook, Ruff and diff checks passed. New tests cover opposing-face
requirements, both anchors, positive extent floors, rotated scan registration,
whole-role compound rejection, pose/resize composition, worker geometry freezing,
coherent fingerprints, HTTP metadata/proposal access and preservation of a smaller
collision. Existing registration, point-cloud, history and rendering tests remain.

Real software Chromium verified a synthetic width error under rotated scan
registration: visual 1000→1040 mm; collision 980→1019.2 mm; Y/Z unchanged. Apply
creates one revision; Undo/Redo restores both roles. Regenerating the viewer from
applied geometry and reloading does not double-scale it; Undo gives 1000/980 mm.
Geometry switches retain 1/3/0/2 collision/both/scan-only/visual meshes and real
mouse rotation; no empty meshes or browser errors. VirtualGL still needs user QA.

Guided video: <http://127.0.0.1:8808/dimension-walkthrough.mp4>, 61.25 s, H.264,
1440×1000. It is explicitly synthetic. Decoded frames verify 6000 original records,
face counts, 1000→1040 mm proposal, unchanged Y/Z and retained smaller collision.
Source video, chapter metadata, decoded frames and recording scripts are in
`projects/dimension-tour-20261115/`; published copy is in apartment/.
`projects/dimension-demo-20261115/` contains the separate browser QA case.
Previous walkthrough files remain served from apartment/ on the new 8808 port.
See doc/VIDEO_WORKFLOW.md for the saved reusable recording method.

## Remaining scope

Oriented/source-parametric dimensions, rotations, shared alignment constraints,
source topology and assembly propagation; real SemDT reload, semantic/articulation
validation, source application/export and portable review bundles. Earlier gravity,
registration masks/anchors, partial-overlap search and visibility/density-aware
coverage remain open. Repeated scans remain outside scope.

---

# Current increment — constrained translation proposals and snapshot overrides

2026-11-15, `semdt-world-refiner`, work remains entirely inside this directory.
No dependencies, source URDF/asset edits, commits or pushes.

**Current apartment workspace: <http://127.0.0.1:8804/>**, service session 44301.
The previous 8802 service is stopped; a new port was used for the restarted backend.
Temporary synthetic servers 8803/8805 and software Chromium 9232 are stopped.
Only the apartment service on 8804 was left running.
The apartment still has its original four history actions and **no overrides**.
Existing heatmap `e51faebc475f4f3ca98b187de6f61e23` remains valid. A read-only
Cooktop Z-only trial is saved as `842735fdcc934a2a9f7fb686bc8176da`: 2542
associated records, tentative ΔZ −1.93 mm, **not_converged**, application blocked.
No apartment geometry was changed. This crop includes other nearby surfaces;
its diagnostic is not an independently validated cooktop correction.

## Implemented first slice of roadmap step 4

- Refine is the fourth enabled stage. The user explicitly approved extending the
  existing workspace test's expected stage set; its other assertions stayed intact.
- `refinement.py`: explicit root X/Y/Z unlocks, per-axis bounds, initial proximity
  association, bounded robust point-to-plane fitting with correspondence updates,
  normal-rank checks and optimizer/convergence guards. Associated original record
  identities stay fixed through fitting. Rotation and dimensions stay fixed.
- Before/after metrics use exact point-to-triangle distances on the same admitted
  records. They are in-sample diagnostics, not independent accuracy. Too few points,
  underconstrained axes, saturated bounds, unfinished optimization or insufficient
  capped-mean residual improvement prevent application. A thin-box synthetic case
  proves correspondence updates avoid the initial opposite-face ambiguity.
- `JobKind.REFINEMENT`, constraints in JobRequest, proposal.json, associated original
  records and after-distance arrays. Jobs retain their model overrides at submission.
  Worker geometry is loaded from baseline + those frozen overrides, so edits/Undo
  while a worker runs cannot relabel newer geometry under an older fingerprint.
- `BodyTranslation` snapshots are cumulative relative to immutable world.npz and
  retain their source job ID. `effective_snapshot` applies them in root coordinates.
  Both geometry roles of one body move together; other bodies stay fixed.
- Snapshot topology is currently flat. No parent/child or assembly motion is inferred.
  The UI states this scope. Any override change invalidates *all* derived jobs and
  caches, conservatively covering dependencies without claiming hierarchy support.
- Overrides live in the same durable Undo/Redo journal. Geometry-aware fingerprints
  omit an empty override field for compatibility with existing analysis signatures.
  Sections, registration, comparisons and generated pages use effective geometry.
- `web/refinement.js`: locks/limits, proposal diagnostics, temporary current/proposed
  preview with blue original ghost and orange proposal, dismissal and explicit Apply
  snapshot override. Preview records are original associated scan records. The ghost
  trace is reused; temporary preview settings do not create history revisions.
  Applying is serialized against Undo, and leaving Refine/using Undo clears preview.
- Browser coordinates are always rebuilt from baseline buffers plus current overrides.
  Embedded effective offsets are subtracted once at initialization. Undo therefore
  cannot accumulate or leave stale displacement. Geometry-dependent heatmaps and
  statistics are disabled/hidden for incompatible overrides or temporary previews.

## Verification

**108 focused tests passed**, including E57/Node. Twelve renderer/override/workspace
checks were repeated after the final UI guards; all passed. Formatting through the
pre-commit hook, Ruff and diff checks passed. Tests cover exact axis locks, ambiguous
planes, bounds, unfinished optimization, iterative thin-box recovery, bounded triangle
queries, immutable baselines, both geometry roles, other-body isolation, cumulative
Undo/Redo, stale result restoration, frozen job inputs and rejected HTTP application.

Synthetic browser scene: `projects/refinement-demo-20261115/`, with a deliberately
20 mm elevated panel, separate visual/collision and a fixed neighboring body. The
fit recovers ΔZ −20 mm with ΔX=ΔY=0 and numerical-zero residual. Application creates
one history revision; Undo/Redo restores both role poses. Browser errors were empty.
Real software Chromium checked stage/geometry switching and mouse rotation. The
user's VirtualGL setup still needs its own check.

Guided video: <http://127.0.0.1:8804/refinement-walkthrough.mp4>, 54.25 s, H.264,
1440×1000. The clip explicitly uses the synthetic example. Decoded frames verify
visible gray scan, blue current-pose ghost, orange proposal, exact deltas and Undo.
Artifacts, chapter list and archived recording scripts are in
`projects/refinement-tour-20261115/`; the published copy is inside apartment/.
The prior workspace and comparison videos remain on the new 8804 server.
See doc/VIDEO_WORKFLOW.md for the reusable recording method.

## Still pending

Rotation/dimension fitting, coupled height/alignment constraints, captured topology
and assembly propagation; source application, real SemDT reload, joint/semantic
validation and export. Automatic gravity/levelling, registration masks/anchors and
partial-overlap search remain earlier open work. New box/cylinder/simple-object
fitting remains explicitly requested follow-up; repeated scans stay out of scope.

---

# Current increment — directional comparison reports

2026-11-15, branch `semdt-world-refiner`. All changes remain inside this directory.
No dependencies added, source-world changes, commits or pushes.

**Current apartment workspace: <http://127.0.0.1:8802/>**, session 47965.
The prior 8799 service was stopped but that port could not be rebound; use 8802.
The four pre-existing apartment history actions were preserved. A derived 50k
comparison was added without accepting/replacing registration or changing views:
`e51faebc475f4f3ca98b187de6f61e23`. Open Jobs → Show heatmap to review it.

## Implemented

- `coverage.py`: labelled directional results, per-body median/P90/threshold
  fractions, reproducible bounded model samples and area-weighted support.
  Original scan records retain their identities. Model samples are stored in
  original scan coordinates with their body indices and distances.
- Nearest model triangle assigns each scan record to a body; this is geometric
  proximity, not semantic correspondence. Equal body sample allocations protect
  small bodies; sampling within each body is by area. Global support weights by
  the represented area, not by the number of bodies.
- Exact oriented-crop filtering is applied to model samples. The global model
  budget is allocated before filtering; tiny crops may retain few/no samples.
  Body and global evaluated areas are Monte Carlo estimates within that crop.
- Unknown scan residuals and unevaluated model regions stay null. Unsupported
  samples do not imply wrong geometry. No visibility/free-space, confidence or
  density-normalization claims; both point budgets and tolerance affect support.
- `comparison.py` retains scan_to_model compatibility, exposes nearest triangle
  ownership, and reports phases through an explicit progress observer.
  The worker uses the same EvidenceComparison entry point as synthetic checks.
- JobRequest adds bounded integer model_sample_count (default 20k, max 50k).
  Jobs retain report.json with typed metrics and provenance (placement, crop,
  budgets, representation and input fingerprints) plus labelled arrays in NPZ.
  GET `/api/jobs/<id>/report` requires a current successful comparison. Older
  jobs with no report remain valid scan-only heatmaps; recompute for both sides.
- `web/comparison_report.js`: dedicated Inspect → Results panel, filters, sort,
  unsampled-body toggle and report download. Show heatmap opens it. Stale results
  are hidden; valid cached reports restore with Undo/Redo and saved views.
  Inspector tab navigation resets previous scroll so results start at the top.

## Verification and demonstration

**89 focused tests passed**, including E57 and Node. Formatting through the
pre-commit hook, Ruff and diff checks passed. Synthetic cases cover missing
returns, crop exclusion, empty reverse samples, bounded reproducible samples,
rigid placement invariance and exact nearest-body ownership. HTTP tests guard
reports against stale registration; browser tests distinguish unknown from zero.

Real software Chromium compared the indexed Cooktop context at 10k and 50k.
The 50k / 20k model-budget job retained 826 in-crop model samples. Median scan
residual ~40.4 mm, P90 ~60.2 mm; model support ~46.6% at 20 mm. This includes
surrounding cabinet surfaces, not only the Cooktop. Its body residual is much
smaller. Changing placement hides the report; Undo restores it. Browser errors
were empty. Final geometry switches retained 100/198/0/98 collision/both/scan-only/
visual meshes and actual mouse rotation, with no empty meshes. The temporary
8801 server and 9231 software Chromium are stopped; only apartment 8802 remains.
VirtualGL-specific QA still belongs on the user's setup.

Guided video: <http://127.0.0.1:8802/comparison-walkthrough.mp4>, 55.5 s,
H.264, 1440×1000. Captions, pointer and control highlights follow the accepted
method. Decoded frames verify 50k points, correct green-to-red colours, per-body
metrics and the unsampled-body distinction. Source clip, chapters, recording
scripts and frames are in `projects/coverage-check-20261115/`. The previous
workspace video remains available on 8802 as `workspace-walkthrough.mp4`.

## Remaining scope

The roadmap marks this as the first bidirectional support slice, not full
observability. Gravity/levelling, registration masks/anchors and partial-overlap
search remain open. Next refinement needs versioned constrained proposals and
parent/child invalidation before geometry edits. New box/cylinder/object fitting
remains explicitly requested follow-up. Source application/export and actual
world reload/semantic/articulation checks are still later stages.

---

# Current increment — staged workspace and oriented evidence selection

2026-11-15. Work remains on `semdt-world-refiner`, entirely inside this directory.
No new dependencies, commits, external source edits or raw-scan modifications.

## Current interface

- Prepare/Align/Inspect stages share a searchable body list and one scene.
  Region/Section/Measure/Views controls live in the contextual inspector.
  Global Undo/Redo, history and jobs remain visible. Refine/Review/Export are
  explicitly planned. Snapshot body names are not a reconstructed semantic tree.
- World/Scan axes plus extra yaw orient both model and scan together. Fresh
  registered projects start in World axes; existing saved Scan views are retained.
  Choose World → Orient axes for a straight apartment plan. Draw crop in plan
  uses a real rectangle gesture, visible box and retained vertical bounds.
- `frames.py` performs an exact oriented-box filter over original records before
  sampling. Scan-cell lookup uses only a conservative AABB. Coordinates/IDs/RGB
  are never rewritten. HTTP region/section frames use optional `frame_T_scan`.
- `inspection_state.py` retains display axes, selected crop, frozen loaded-query
  matrix and comparison job IDs. Changing the display basis does not reinterpret
  a preceding query. Measurements retain original scan endpoints.
- `web/coordinate_frame.js`, `workspace.js`, `jobs.js`, `workspace.css` and the
  existing rendering/inspection controllers are embedded in the portable page.
  Renderer updates and resizes are ordered. Hidden stages skip pending resizes.

## Bounded analysis

- `jobs.py`, `job_worker.py`, `run_job.py`, `job_service.py`: one owned worker,
  4 GiB address-space limit, single-thread numerical-library defaults, explicit
  phases/progress, cancellation and atomic result publication.
- Web registration remains upright yaw/XYZ over the frozen preview, requiring
  explicit shared-vertical confirmation. Generated candidates are separate until
  Review candidates; replacement and placement acceptance are undoable.
- Regional heatmaps use dense indexed records when available, preview otherwise,
  explicit visual/collision geometry and optional body scope. Evaluation budget
  is independent of display settings. Reports median/P90 and within-tolerance
  point fraction, not confidence or model coverage.
- Result applicability fingerprints include world/preview identities, dense index
  identity and registration. Stale results cannot replace candidates or colour
  newer evidence. Keep jobs/ when sharing project histories with result references.
- Undo/Redo and saved views restore valid cached heatmaps by ID, with original
  point associations; no recomputation is required for that restoration.
- `comparison.py` owns numeric distance queries; inspection.py re-exports its
  existing SurfaceComparison import for compatibility. PointRecordLayout now
  uses plain string dtype field names so structured NPZ results load correctly.

## Verification and case artifacts

Synthetic tests cover exact rotated cropping, filtering before sampling, byte
provenance, metric frame round trips, nested browser matrix serialization,
job cancellation/application/staleness, hidden-stage resize and view-action
serialization. Existing E57 and renderer-control regressions are retained.

Real software Chromium checked world-aligned Cooktop bounds (~0.823×1.102 m
including context), original 50k records loaded in that frame, actual plan drag,
a 10k regional heatmap, acceptance/Undo, registration generation/cancellation,
explicit proposal application, stale old result and restoration by Undo. Frame
changes preserve the loaded evidence and Undo restores World + its heatmap.
Final suite: **76 tests passed**, including the isolated E57 reader and Node
regressions. Formatting hook, Ruff and diff checks passed. Real mouse rotation
also survives stage navigation and collision/both/scan-only/visual switches
(100/198/0/98 meshes), with no empty meshes or JavaScript errors.
VirtualGL-specific rendering remains to be checked on the user's setup.

`projects/ui-redesign-smoke-20261115/` and `ui-redesign-check-20261115/` contain
these disposable tests. The production apartment's prior four history actions
were preserved. Do not delete or reset that history. Earlier smoke results may
be stale after the fingerprint was extended to include dense index identity.

Final apartment server: <http://127.0.0.1:8799/> (session 74266). Port 8793 could
not be rebound after stopping the preceding server; use 8799 for UI and videos.
Recording uses `projects/workspace-tour-20261115/` with linked existing index.
Published guided video: <http://127.0.0.1:8799/workspace-walkthrough.mp4>
(60.5 s, H.264, 1440×1000). Captions, controls, pointer and crop gesture are
visible; decoded dense-scan and green-to-red heatmap frames were inspected.
Recording scripts, chapter metadata and frames are archived beside the source
clip. The final run reported no browser errors and a successful 10k comparison.
Temporary recording/test servers and software Chromium are stopped; only the
apartment service on 8799 was left running.
See doc/VIDEO_WORKFLOW.md for the tracked, accepted video-production method.
The original guided reference and scripts remain in the ignored revision-2
recording directory. The staged UI direction is recorded in UI_DESIGN.md.

A decoded recording frame caught an interactive palette bug: Plotly JavaScript
does not accept the Python `_r` palette suffix. Interactive restyles now reuse
the explicit preview stops, with the nested per-trace wrapper Plotly requires.
RGB-only pages also carry those stops for later regional comparisons. Tests
prove both contracts. One optional full-preview recomputation exceeded a 4 GiB
CLI cap; the existing apartment distances were retained and its UI assets
refreshed without touching history or recomputing evidence.

## Next scope

Finish automatic gravity/levelling, masks/anchors and partial-overlap search;
then coverage-aware comparison and constrained geometry proposals. Parent moves
must invalidate dependent fits. New box/cylinder/simple-object fitting remains
an explicitly requested follow-up, not discarded. Repeated scans remain deferred.
Source application/export and semantic/articulation validation are later stages.

---

# Latest increment — 2026-11-15

Current apartment viewer: <http://127.0.0.1:8793/> (serve session 1061).
Temporary smoke servers and headless Chromium were stopped.
Guided walkthrough v2: <http://127.0.0.1:8793/refiner-walkthrough-v2.mp4>
(71.5 s, 2.47 MB). Controls/clicks are highlighted; scan-only 50k and 150k
close-ups begin at 26.5 s and 34.25 s. Recording artifacts and decoded QA frames
are in `projects/functionality-video-v2-20261115/`. UI_DESIGN.md records the
accepted staged UI direction. The temporary recording browser/server are stopped.

The entries below this section describe earlier increments chronologically.
Current work remains inside this directory; no new dependencies or commits.

- Heatmap uses `RdYlGn_r`: zero green, intermediate yellow, capped errors red.
- `web/scan_cloud.js`: split detail into <=50k-point traces; clear spare batches
  on smaller loads. `PlotUpdates` serializes trace insertion and other updates.
- The 50k/150k API returns the correct counts. The user's VirtualGL blank view
  was not reproduced in software Chromium; bounded batches harden that path.
- Region bounds are inclusive original scan XYZ in metres. Budget changes
  density, not region extent. Body selection includes 15 cm of context.
- `history.py`, `inspection_state.py`: typed persistent metadata revisions,
  atomic journal, undo/redo. Scan/world baseline bytes are never rewritten.
  Registration reruns and acceptance retain prior candidates and decisions.
- `project.py`: journal is authoritative; legacy metadata and accepted transform
  are compatibility artifacts. New actions after Undo discard the redo branch.
- CLI `undo`, `redo`, `history`; browser buttons and recent action list.
- `web/inspection_tools.js`: persistent camera/view/query, independent display
  clip bounds, XY/XZ/YZ slabs, point-pair measurements and annotated bookmarks.
- `sections.py`: exact trimesh plane cuts, body/role labels in original scan
  coordinates. 2D plots use equal metric axes and selected-region bounds.
- New loopback `/api/state`, `/api/undo`, `/api/redo`, `/api/section` resources.
  POSTs serialize mutations within the serving process. This is a single-user
  local tool, not a concurrent editing service.
- Main initial RGB/heatmap sharing and ordered geometry updates are retained.
  Loading indexed evidence still invalidates the preview's heatmap; regional
  heatmap jobs are the next increment, not included here.

Browser smoke work uses the ignored `projects/inspection-smoke-20261115`, copying
only baseline metadata/NPZ/page and linking the existing index. Apartment history
is not filled with test actions. Temporary test servers/browser are stopped at
completion; use the final response's apartment URL or restart `serve`.

Verification: **50 tests passed**, including E57 and Node regressions; formatting
hook, Ruff and diff checks passed. Real software
Chromium verified 50k/150k batches, persistent undo/redo, 22 section traces,
metric equal axes, measurement values, clip 3038→31 points and saved-view restore
to 3038 points. Geometry switches retain 100 collision / 198 both / 98 visual
meshes and real mouse rotation; no empty meshes or JS errors. Tests also cover
concurrent scan/measurement trace insertion indices. Apartment history remained
untouched by these tests. The graphics results do not establish VirtualGL QA.

Follow-up remains explicitly requested: fit new boxes/cylinders/simple objects
from a selected scan region after existing-object refinement is usable.
Repeated scans remain out of scope. CI wiring still needs authorization to edit
outside this directory. Tests here remain part of the focused refiner suite.

# Scan refiner handover — 2026-11-10

## Scope

Branch `semdt-world-refiner`, originally created from fetched `origin/main` at
`d47537cf910577321df4757fa76253785bd63c59`. The remote has advanced since then;
the branch's base was preserved. No commits or pushes. All new work is inside
`scripts/semantic-digital-twin-scan-refiner`; older untracked case artifacts
and model/scan inputs remain untouched. Earlier checkpoint:
[HANDOVER-2026-11-09.md](HANDOVER-2026-11-09.md).

The user accepted automatic registration, good inspection and constrained
refinement as the initial direction. Repeated scans are deferred. New-object
box/cylinder/simple-object fitting is explicitly requested follow-up work;
keep that reminder in ROADMAP.md.

The user explicitly requires an Undo button for actions. Implement persistent
reversible action history before adding project mutations, including LLM actions;
restore dependent artifacts consistently. Current CLI registration replacement
is not yet undoable. See ROADMAP.md for the full requirement.

## Implemented

- `refine.py`: init, align, view, accept, serve, demo, index and crop commands.
- Frozen SemDT visual/collision geometry and independent original preview.
- Upright automatic yaw/XYZ registration candidates and explicit acceptance.
- Compact self-contained Plotly viewer with one shared scan trace, local
  float32 display coordinates, layer controls and a preview-distance heatmap.
- `read_e57.py`: bounded isolated-reader protocol, all stations, Cartesian or
  spherical coordinates, station poses applied once, invalid records excluded.
- `point_records.py`: float64 positions/intensity, uint8 RGB, uint64 original
  record IDs and uint32 station IDs. Missing intensity is NaN.
- `spatial_index.py`: unquantized binary physical cells, bounded exact regional
  batches, deterministic samples and a cached 200k-record overview.
- `scan_sources.py`: common source interface for previews and isolated E57.
- `web_service.py`: loopback-only static viewer plus scan-info/regional binary
  endpoints. Advertises binary offsets; compresses API replies when requested.
- Browser body picking fills a region with 15 cm context; Load region and
  Overview request 10k/50k/150k original records. Point-size/hover/opacity controls.
- Exported NPZ crops retain upstream station IDs, record IDs and intensity when
  reopened or used as preview sources.

No dependencies were installed or added. The reader uses the existing
`/home/pmania/.cache/semantic_digital_twin/apartment-e57-reader/bin/python`.
Do not resolve that interpreter symlink: doing so bypasses its virtualenv.
The CLI uses `.absolute()`; a regression test covers the actual CLI invocation.

## Current local projects and viewers

- Demo: `projects/demo/`, a 16k-record preview index and synthetic geometry.
- Apartment: `projects/apartment/`, 198 mesh roles (98 visual, 100 collision),
  original-coordinate 1,994,483-record PLY preview, and the existing refined
  apartment URDF from `scan_previews/apartment/handles-column-aligned/`.
- Full apartment E57 index: **546,488,256 records**, approximately **25.69 GB**,
  generated in about eight minutes with 250k-record decoder batches. No spatial
  position quantization. Source hash, complete index metadata and station data
  are inside `projects/apartment/scan-index/`.
- Current read-only servers: demo <http://127.0.0.1:8786/> and apartment
  <http://127.0.0.1:8787/>. Earlier viewer listeners were stopped during updates;
  their old URLs should not be used. Restart with `serve --port ...` if needed.

The best apartment registration remains unaccepted: yaw approximately -46.4138
degrees and translation (-5.3391,3.4415,-136.9284) m. It is close to the earlier
floor-anchored registration but does not fit its .052-degree tilt. This is not
independently surveyed accuracy.

## Evidence and performance checks

Original apartment page: 39,333,815 bytes. Updated page: 20,045,029 bytes, about
49% smaller for the same displayed points and heatmap. Browser frame rate was
not measured. Full-precision evidence and meshes were not reduced. The original
page was retained as `projects/apartment/index-before-streaming.html`.

`projects/apartment/verify_index.py` compares indexed records with the earlier
dense cooktop crop by original E57 identity, without decoding the E57 again.
Its `index-verification.json` records:

- 50k original records sampled within the crop's axis-aligned scan bounds.
- 40,497 shared reference identities: exact XYZ and byte RGB agreement.
- Approximately .959 seconds for the regional query and .0036 seconds for a
  cached 50k overview. These are local warm-cache timings, not service guarantees.

The old crop is oriented, so its axis-aligned bounds also contain other valid
scan points. The unmatched sampled IDs are not automatically errors.

Reproduce the case check from the workspace root:

```bash
PYTHONPATH=scripts/semantic-digital-twin-scan-refiner:. \
  /home/pmania/venvs/cram-env/bin/python \
  scripts/semantic-digital-twin-scan-refiner/projects/apartment/verify_index.py
```

The focused suite passed 30 tests including the optional E57 integrations with
the existing reader interpreter. Tests use the configured project Python,
`--orm-build never` and an 8 GiB cap. E57 integrations skip without
`SCAN_REFINER_E57_PYTHON`. No ORM regeneration. A known Matplotlib Axes3D warning
does not affect Plotly rendering.

HTTP regional queries and binary metadata were checked. Viewer controls were
exercised in a simulated DOM using real HTTP response data, including exact
byte-offset/identity decoding and stale-heatmap invalidation. No graphical
browser was available for visual/frame-rate QA. Node tool networking could not
reach loopback, so its mock fetch used replies captured by the Python HTTP check.

## Current limitations and next work

2026-11-13 geometry-switch correction: decode Plotly numeric buffers for all
trace coordinates and triangle indices before caching model positions. The
previous controller treated `{dtype,bdata}` objects as arrays, so candidate
changes silently replaced meshes with zero vertices; body picking also could
not reduce encoded coordinates. Initialize after the initial plot promise and
disable custom scene controls until ready. Serialize renderer operations;
geometry changes use one data/layout update while preserving camera state.
New files `web/plot_buffers.js` and `web/plot_ready.js` are embedded into each
self-contained page. Existing data and the persistent scan index are unchanged.

Two browser-control regressions failed before the fix (zero retained vertices;
four overlapping renderer operations) and now pass. Full focused suite: 33
passed with the optional E57 and Node checks enabled. The existing Node runtime
was used via `SCAN_REFINER_NODE=/usr/lib/chatgpt/resources/cua_node/bin/node`;
no dependencies were installed. Real headless Chromium confirmed collision=100,
both=198, visual=98 and scan-only=0 model meshes, with rotation working after each
switch. Candidate changes preserve vertices and body picking supplies valid
regional bounds. Repeated switches retain one canvas and report no JS errors.
Apartment and demo pages were regenerated. Current apartment server is
<http://127.0.0.1:8788/>; earlier server URLs above describe previous sessions.

Camera-control correction: initial drag mode is explicitly `orbit`. Every view
preset restores it, and a Rotate button preserves the current camera while
restoring rotation. Previously Perspective retained a selected pan/zoom mode.
The issue was reproduced in headless Chromium with software rendering, then
verified after the fix: initial/preset modes, restoration from pan, preserved
viewpoint and actual mouse-drag camera changes. Five focused inspection/rendering
tests pass, including the new initial-rotation regression. Both pages were
regenerated. The earlier lack of graphical QA above describes the indexing
increment; this later check used the user's approved Snap Chromium launch.

Regional data loads on demand, not automatically with camera motion. The
browser still initially loads a self-contained model/preview page. Switching to
new regional points disables the old preview heatmap. Regional heatmap jobs,
fixed plan/elevation sections, measurements and saved annotated views are next.
Registration jobs still run through CLI; gravity detection, masks, anchors,
registration history and model-to-scan coverage reasoning remain pending.

Source writing, constrained geometry proposals, articulation validation,
semantic-graph persistence and export integration remain future increments.
The geometry snapshot currently omits semantic annotations. Do not assume a
bare URDF reconstructs provider-specific assembly semantics.

The raw E57 remains authoritative for attributes beyond the current record
schema. Byte RGB is supported; unsupported conversion is rejected. Station
poses/counts/available-field names are retained, but arbitrary metadata is not
round-tripped. Indexing is not resumable: completion metadata is published last,
and existing/partial directories are refused. The initial failed reader launch
was retained as `scan-index-failed-reader-2026-11-10/` before retrying.

Persistent index files trade space for exact evidence and existing dependencies.
Compression and alternative indexes/viewers can be evaluated later. Keep useful
progress bounded and avoid a rendering-engine rewrite before actual profiling.
CI wiring is pending because the user currently restricts edits to this directory.
