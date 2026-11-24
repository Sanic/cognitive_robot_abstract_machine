# Reusable scan-based SemDT refinement: proposed system

Status: initial design, 2026-11-09. The user accepted the recommended direction.
See README.md for implemented scope and ROADMAP.md for current priorities.
See [UI_DESIGN.md](UI_DESIGN.md) for the proposed staged browser workspace.

Direction update, 2026-11-20: the agent runs in the existing conversation or Codex
CLI. The browser supports inspection, target selection and proposal review; it
does not need its own chat or a growing manual refinement wizard. The agent uses
shared deterministic tools to investigate and propose corrections. Explicit
multi-object change sets come before a general assembly solver. The current
priority section of ROADMAP.md supersedes the original implementation ordering
and manual-workflow emphasis in this broader design.

The goal is to make refining an existing, approximate semantic world against
measured evidence repeatable. Python world construction and URDF imports should
share scan access, registration, inspection, comparison and proposal machinery.
The system must also work for living rooms, laboratories and factory floors.

This proposal builds on the earlier workflow in
`semantic_digital_twin/doc/scan_based_map_refinement.md`,
including the SUTURO and apartment case studies. It proposes extracting useful
infrastructure rather than converting all historical scripts into public API.

## Product and workflow

Start with a local project directory, a Python API/CLI and one browser workspace.
Use local files and a single local service initially. Hosting, authentication,
concurrent editing and distributed scheduling can wait for a demonstrated need.
The same operations must work without the browser and without an LLM.

The typical session is:

1. Open a world provider or URDF and attach a scan.
2. Inspect metadata and prepare indexed scan data and a model baseline.
3. Align stable structures; inspect alternatives and accept registration.
4. Inspect the whole scene, then select a room, assembly, object or scan region.
5. Diagnose the discrepancy using RGB, sections, measurements and heatmaps.
6. Choose a permitted change and constraints; preview a fitted or manual proposal.
7. Review changes to geometry, semantics and articulation; apply or reject.
8. Rebuild the actual world and refresh affected comparisons and exports.

For example: select a sofa, permit floor-plane translation and yaw, lock its
dimensions, exclude a blanket from fitting, inspect before/after, then save the
accepted pose. The infrastructure is the same as for a cabinet assembly or
machine; their geometry and constraints differ.

## Architecture and responsibilities

| Part | Owns | Important boundary |
| --- | --- | --- |
| Project record | Input identities, revisions, frames, selections, decisions and artifact dependencies | References immutable raw data; no silent baseline replacement |
| Scan access | Metadata, streaming import, spatial index, level of detail, exact crops and attributes | Display resolution does not determine fitting evidence |
| World adapter | Build SemDT, preserve stable identities, expose meshes, joints, hierarchy and semantics | Loading a world does not imply arbitrary source code can be rewritten |
| Registration | Correspondences, anchors, candidate transforms, refinement and review | Registration versions are explicit inputs to every comparison |
| Comparison | Scoped distances, coverage, residual distributions and per-object summaries | A residual is not automatically an error or a correspondence |
| Refinement | Editable parameters, constraints, candidate generation and validation | Does not invent unobservable dimensions or change locked parameters |
| Source application | Reviewable patches or overrides, conflict checks and rebuild | Separate support for Python parameters, URDF and generated Xacro |
| Browser workspace | Inspect, measure, select, compare, review and save views | Uses the same operations and records as CLI/Python clients |
| Artifact runner | Bounded jobs, cancellation, progress, cache and atomic publication | Refreshes only outputs whose actual inputs changed |

Keep SemDT as the canonical world representation. Add refinement-specific
records beside it rather than creating a second semantic world implementation.
Initially these can be modules in SemDT with format adapters; package boundaries
should follow actual dependency requirements rather than this table.

## Durable project data

Use typed, versioned records with these responsibilities:

- **Scan source:** identity/hash, units, coordinate precision, bounds, stations,
  available attributes, timestamps and scanner poses when supplied.
- **World revision:** provider/source revision, resolved asset hashes, stable
  body identities, geometry roles, joint state and semantic snapshot.
- **Registration:** source/target frames, transform, anchors, candidates,
  evidence, diagnostics and review status. Follow SemDT transform conventions.
- **Selection:** geometry, frame, exclusions and source revision; distinguish
  display selection, fitting selection and evaluation selection.
- **Editable parameter:** owning object/source, units, bounds, permitted degrees
  of freedom and dependencies. Mesh bounds and link origins are distinct.
- **Constraint:** a relation such as equal height, centred, supported, fixed
  dimension, parallel, mounted or clearance, with provenance and affected objects.
- **Proposal:** parent revision, changed parameters, evidence, fit diagnostics,
  validation results and acceptance/rejection record.
- **Artifact:** input fingerprints, settings, software version, output paths,
  completion state and freshness.

Store small records separately from indexed point/mesh arrays. Use existing
project serialization facilities when implementing typed records. No database
is required for the first local workflow. Version record schemas from the start.

Use source identities for point provenance: scan source, station and record
index. Preserve available RGB/intensity and metadata. Spatial display indexes
may quantize coordinates or discard attributes; record this explicitly and
retain an exact evidence store when needed. Do not promise lossless E57 metadata
round-tripping through a browser-oriented point format.

## Scan infrastructure and scale

Ingest a large source once with bounded memory; create a persistent spatial
index and progressive display representation. Further inspection crops should
query the index instead of repeatedly decoding the complete E57.

Expose three distinct requests:

- Display: visible region and point budget, with progressive loading.
- Analysis: reproducible spatial sampling at a declared physical resolution.
- Evidence: original records within a region, optionally streamed in batches.

Build resolution levels in physical units, selected relative to the feature
and scanner quality rather than fixed kitchen presets. Support rotated boxes,
height slabs, polygons and semantic-object selections. Freeze point identities
and masks for before/after evaluation.

Start with E57 plus one simple interchange format already supported by the
environment. Retain station transforms rather than silently flattening every
input. Full registration of unregistered scanner stations is a separate possible
extension, not a prerequisite to refining against a registered scan.

Prototype an existing point-cloud viewer/index combination before selecting
storage. Potree is a strong viewer candidate; COPC is a storage candidate.
Check their compatibility, attribute/provenance preservation, precision,
conversion cost, licensing and deployment together. Do not build multiple
indexes or a new renderer without evidence that this is necessary.

## Registration and trustworthy comparison

Support manual landmark initialization and imported transforms first. Add
existing coarse registration and robust multi-resolution refinement behind the
same interface. Allow gravity, floor and known anchors, without assuming every
site is flat or Manhattan-aligned. Repetitive factory layouts need alternative
candidates and geographically separated checks.

Registration should use stable structures; movable furniture, open doors,
people and known wrong geometry can be masked. Keep source-to-world and
station-to-source transforms explicit. Large site coordinates may need a local
rendering origin while preserving original coordinates and precision.

Comparisons need both directions with distinct meanings:

- Scan to model: observed surfaces that disagree with, or are absent from, the
  selected model. Scope excludes deliberately unmodelled regions.
- Model to scan: model surface support, subject to visibility and scan coverage.
  No nearby returns alone does not prove the object is absent.

Show absolute distances initially; signed distances only where surface
orientation and geometry support a meaningful sign. Provide matching gates
(selected objects, distance, normals or explicit correspondences) so unrelated
nearby surfaces do not silently become evidence.

Per-object summaries should show residual distributions, spatial coverage,
sample counts, exclusions, evaluation resolution and tolerance. Treat
insufficient evidence separately from acceptable/deviating geometry. If scanner
poses exist, later add visibility reasoning; otherwise use conservative unknown
coverage and manual masks. Do not present uncalibrated confidence percentages.

Compare candidates using the same frozen registration and evaluation records.
Keep fitting evidence distinct from held-out checks. Report numerical fit
improvement without implying surveyed accuracy.

## Browser workspace

Use a shared scene and selection state for:

- Searchable body/assembly tree and semantic/context relations.
- RGB scan, visual geometry, collision geometry, baseline and proposal layers.
- Independent scan-to-model and model-to-scan heatmaps with fixed scales.
- Perspective, orthographic and fixed equal-scale plan/elevation views.
- Clip boxes, cross-sections, measurements, adjustable point size and density.
- Object-local axes, mesh bounds, joint frames and original-position outlines.
- Saved viewpoints, annotated issues and links that restore selection/view state.
- A proposal panel showing changed parameters, locks, constraints and validation.
- Job progress, cancellation, stale-result labels and export controls.

Reuse one application with project data loaded at runtime. Stop generating a
different large HTML application for every handle and heatmap. Retain portable,
read-only review bundles for colleagues; omit raw scans unless explicitly chosen.

## Refinement and source application

Begin with a small toolbox: rigid object/assembly placement, dimension changes
on declared parametric shapes, planar/box fitting, joint-state adjustment and
shared alignment constraints. Add cylinder fitting if industrial use needs it.
Known dimensions and human statements are explicit constraints with provenance.

Keep generic relations reusable: centred-on, equal-height, common-axis,
supported-by, mounted-to, parallel and minimum-clearance. Kitchen recipes can
compose these; a handle-specific optimizer does not belong in the universal core.
Deformable sofas, curtains and vegetation need different expectations from
machined parts; retain manual masks and approximate geometry options.

An accepted parent move invalidates dependent child fits. Show what must be
rechecked and refresh affected geometry, comparisons and exported snapshots.
Check articulation, clearances, visual/collision policy and semantic membership
separately. Geometric improvement alone cannot validate a KitchenIsland or a
handle-free Door annotation.

For persistence:

- Python worlds: expose selected construction parameters or save explicit
  refinement overrides applied to a named baseline. Source edits remain manual
  where no parameter interface exists.
- URDF: produce reviewable origin/scale/joint/asset changes with asset-local
  frame validation. Detect externally modified inputs before applying.
- Xacro: change the generating source when supported; otherwise clearly mark
  an exported URDF as a derived artifact, not a maintained source fix.
- Arbitrary meshes: start with pose changes; require an explicit asset operation
  for geometry edits, preserving provenance and portable formats as appropriate.

Expose an Undo button backed by persistent before/after action records. Cover
registration, selection/exclusion and constraint changes as well as geometry
and source edits, including LLM-initiated actions. Restore previous project
state and invalidate or refresh dependent artifacts together. Preserve earlier
registration revisions when rerunning alignment. Add redo where practical.

Support rejection, undo and replay against the same baseline. Portable exported
meshes and editable source releases are separate deliverables.

## LLM participation

Offer the same typed operations to a human, CLI or LLM. The LLM can inspect
summaries and images, suggest regions/constraints, run bounded analyses and
explain proposals. Its explanations are not evidence. Numerical operations,
records, source application and validation remain deterministic and inspectable.

No agent should need to assemble temporary crop scripts or rewrite viewer HTML
for routine work. Batch acceptance can be a policy later; human review should
remain available at registration and proposal boundaries.

## Lessons from related projects

These are documented capabilities and design inspiration, not benchmarked
claims or a commitment to integrate the products.

| Project | Documented capability | Proposed lesson |
| --- | --- | --- |
| [Potree](https://github.com/potree/potree) | Large point-cloud viewing, measurements, clipping, profiles, mesh overlays and annotations | Reuse established browser infrastructure and add SemDT editing/review |
| [COPC specification](https://copc.io/) | Compressed LAZ with clustered octree and spatial subset access | Evaluate indexed, partial access rather than repeated full scan decoding |
| [Open3D multi-scale ICP](https://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html) | Coarse-to-fine registration | Reuse numerical methods; add explicit evidence, constraints and review |
| [CloudCompare ICP](https://cloudcompare.org/doc/wiki/index.php/ICP) and [M3C2](https://cloudcompare.org/doc/wiki/index.php?title=M3C2_%28plugin%29) | Partial-overlap registration and robust directional cloud-to-cloud comparison | Respect overlap and noise; advanced repeated-scan comparison can follow later |
| [Verity analysis parameters](https://clearedge3d.com/support/verity/analyze-and-review/analyze/analysis-parameters/) and [analysis FAQ](https://clearedge3d.com/support/verity/analyze-and-review/analyze/analysis-faq/) | Tolerances, scanner-noise settings and insufficient-data cases | Inspection needs evidence quality and per-object outcomes, not heatmap colour alone |
| [EdgeWise workflow](https://www.clearedge3d.com/support/edgewise/) | Semi-automated extraction, repeated patterns and remainder clouds | Consider missing-object regions, reusable shape fitting and repeated assets |
| [NavVis IVION sharing](https://knowledge.navvis.com/docs/sharing) | Shared views, points of interest and measurements | Save and share exact inspection context rather than ambiguous screenshots |

## Delivery sequence and acceptance criteria

1. **Durable foundations.** Preserve useful temporary drivers; define a project
   record, baseline snapshot and world adapter; extract generic scan operations.
   Demonstrate Python and URDF imports with explicit frames and reproducible
   exact crops. Keep old case artifacts as evidence rather than public API.
2. **Unified inspection vertical slice.** One web workspace, progressive scan
   loading, model picking, sections, measurements, dense crops and both comparison
   directions. Reopening a project restores registration, selections and views.
   Region inspection must not reread the complete E57 after indexing.
3. **Registration and quality workflow.** Manual/imported transforms followed by
   automatic candidate/refinement tools, exclusions, tolerances and saved reports.
   Verify transforms on known synthetic cases and stability in partial overlap.
4. **Reviewable refinement.** Pose/dimension proposals, parameter locks and shared
   constraints; apply, rebuild, undo and refresh affected artifacts. Reproduce
   handle centring, cooktop placement and parent/child invalidation from our cases.
5. **Generalization check.** Exercise a non-kitchen room and an industrial example
   before declaring the API general. Include movable objects, repeated geometry,
   missing coverage and a large scene; use synthetic fixtures where scans are
   unavailable, without claiming real-world validation from them.

Benchmark memory, indexing time, regional query latency, browser interaction
and storage expansion on our actual scan/hardware before setting performance
targets. Validate caches against input changes and interrupted jobs. Keep data
conversion isolated when reader dependencies conflict with the project runtime.

Defer learned recognition, whole-scene reconstruction, temporal change tracking,
multi-station SLAM, advanced uncertainty models and enterprise collaboration
until priorities justify them. Keep timestamps/station provenance now so those
extensions remain possible.

## Questions to decide scope

The original questions and suggested defaults were:

1. Missing geometry: flag regions, fit simple new shapes, or propose semantic
   objects automatically? Suggested first step: flag and manually fit primitives.
2. Primary workflow: one registered scan, repeated captures, or unregistered
   stations/robot sessions? Suggested first step: one registered scan.
3. Extra inspection priority: measurements/sections/shared views, tolerance
   reports, or robot clearance/accessibility checks? Suggested first step:
   measurements and sections, with basic per-object summaries.

Further choices before implementation: local-only versus shared review server;
whether refined Python worlds should use persistent overrides or direct source
parameters; desired first non-kitchen example; whether articulation and semantic
editing belong in the first release. These need not block the architecture sketch.

The user accepted the recommended defaults, requested automatic registration,
deferred repeated scans and explicitly asked to retain new-object fitting as a
follow-up reminder. ROADMAP.md is the current scope record.
