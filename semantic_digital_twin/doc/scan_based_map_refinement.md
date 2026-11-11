(scan-based-map-refinement)=
# Improving semantic environment maps from point-cloud scans

This handover describes a repeatable workflow for refining a hand-modelled
environment against a measured scan. It applies to kitchens, laboratories,
offices and other partially modelled spaces. The output is a better semantic
map, a reproducible before/after comparison, and a record of what was measured
versus assumed.

The workflow was exercised on a Python-defined SemDT laboratory world and a
URDF-defined apartment loaded through `URDFParser` and `WorldReasoner`. The same
registration and geometry checks apply to both; applying the result differs.
The analysis scripts have environment-specific inputs, labels and bounds;
they are not a general-purpose command-line tool. Case handovers and safe
commands live in `scan_previews/alignment/` and `scan_previews/apartment/`.

The [reusable system proposal](scan_refinement_system_plan.md) describes how to
turn these case workflows into shared scan access, inspection and refinement tools.

## Intake for the next environment

Start with fresh evidence and an output directory for the new environment. The
kitchen's transforms, dimensions, labels and crops are case evidence, rather
than defaults for another room.

| Input | What to record |
| --- | --- |
| Map | Environment factory, source revision, body names, visual/collision meshes and joint state |
| Scan | Immutable source path/hash, units, stations, poses, attributes and capture date if available |
| Scope | Which modelled objects should match; which scan areas lie outside the model |
| Scene state | Open drawers/windows, movable furniture, known damage and changes since capture |
| Constraints | Floor support, adjacency, articulation, required clearances and collision behaviour |
| Review | Who can identify surfaces and confirm alignment/candidates |
| Storage | New artifact directory, baseline retention and storage for large scans |

Retain separate stages for inputs, registration, frozen observations, proposals
and applied results. Export the untouched model before editing source. Confirm
an RGB alignment overlay with the person who knows the room before local fitting.

## 1. Establish the inputs and preserve an immutable baseline

Record the source-map revision, scan identity/hash, capture date if known,
coordinate units, available point attributes, scanner poses and scan count.
Check whether multiple scans are already registered. A furniture move between
capture and modelling can be a real scene change rather than an alignment error.

Preserve the original scan container. A derived XYZ/RGB PLY is useful for
processing but does not preserve every possible scan attribute or container
metadata. Do not assume another E57 has the same attributes, coordinate types
or scan layout as the laboratory example.

Build the unmodified semantic world and export:

- Per-body visual and collision meshes, body names and root-frame transforms.
- A labelled model scene and a deterministic surface sample for registration.
- The source revision/hash, object/joint state and nominal parameter values.

Store original mesh vertices and faces independently of future source rebuilds.
This is essential: once the Python map changes, rebuilding it is no longer a
way to obtain the baseline. Key meshes by stable body names, not list order or
triangle indices. Later models may add or remove bodies.

Read the repository's instructions and spatial conventions before editing.
Use its configured interpreter and existing dependencies. Ask before adding
dependencies where required by the project.

### Python worlds and imported URDFs

For a Python-defined world, retain the construction parameters and a snapshot
of the built geometry. For a URDF world, also retain the XML, package resolution,
referenced assets, geometry origins/scales and the reasoner's semantic output.
Use the actual world provider for final checks: a provider can add semantic
composition that a bare URDF and generic reasoner do not reconstruct.

Compare meshes in a declared common frame. A link origin is not necessarily the
centre of its mesh or drawer front. The apartment's top cabinet6 drawer had a
12.5 mm joint offset partly compensated by mesh placement; centring its handle
on the joint would reproduce an existing error. Record geometry bounds in the
parent cabinet frame as well as link-local bounds.

Inspect both visual and collision geometry before choosing a fit target.
Collision can be a decomposition, filled aperture, proxy or simplified obstacle;
it is not generally an inset copy of the visual. Do not infer planning intent
from bounding boxes alone. Trace consumers when changing collision behaviour.

## 2. Make a preview before loading the entire scan

Inspect headers and record counts first. If the scan is too large for convenient
viewing, decode bounded chunks and write a preview. Read all relevant scans;
do not accidentally sample only the first chunk or first station.

The laboratory preview retained approximately two million original records
using a deterministic stride. This preserved sampled XYZ/RGB values exactly.
For a different scan, inspect acquisition order: a stride can miss repeated
structures or small features. Stratified sampling, reservoir sampling or a
spatial preview may be more appropriate. Keep the original-record preview
separate from voxel-averaged clouds used for fitting.

For each scan station, account for its coordinate frame, pose and invalid
records according to the reader and actual file contents. Preserve the mapping
back to source records where practical. Record any transformations explicitly.

Use a partial output file and publish it only after successful completion.
Record decoded/written counts and verify the output header and attribute types.
A viewer progress stall alone does not establish file corruption.

## 3. Separate acquisition, scoring and display crops

These selections serve different purposes:

| Selection | Purpose | Rule |
| --- | --- | --- |
| Acquisition/working crop | Retain useful raw evidence and alignment context | Keep a generous master selection; preserve the original scan |
| Fitting/evaluation regions | Identify corresponding surfaces and compare candidates | Freeze exact points and masks before comparing models |
| Display crop | Keep the interactive viewer understandable | May expand freely without changing fit or score points |

In CloudCompare or another point-cloud editor, crop the preview first. Retain
floor patches, corners and distinctive objects around the target. Excessively
tight crops make global alignment ambiguous and hide neighbouring constraints.
Preserve any global shift/scale information so exported coordinates remain
recoverable in the original frame.

Verify the exported file format from its header, not its extension. In this
case, a file named `.ply` was actually a CloudCompare BIN project; it needed a
separate real PLY export.

A bounding box enclosing a polygon selection does not reproduce that polygon
selection. It can retain most of the room. For detailed work, stream narrowly
bounded patches from the full-resolution source rather than materialising a
huge enclosing crop. Save the selection bounds or selection geometry.

When the viewer appears to be missing a wall, first inspect its display bounds
and subsampling. Missing displayed points need not be missing scan evidence.

## 4. Register the room before fitting individual objects

Use one explicit transform convention throughout; see the [style guide](style_guide.md).
`scan_T_root` maps model-root points to original scan coordinates:

```text
p_scan = scan_T_root @ p_root
p_root = inverse(scan_T_root) @ p_scan
```

These are homogeneous column vectors. NumPy row arrays require the corresponding
transposed rotation. Unit-test transform direction and inverse composition.

If the floor is visible, fit a verified floor plane robustly to establish
gravity and height. Check that the selected plane is the floor, rather than a
tabletop or another level. Otherwise obtain gravity/height from reliable
metadata, surveyed landmarks or explicit manual correspondences.

Search broadly over the remaining placement variables before local refinement.
The laboratory procedure used height-sliced horizontal occupancy correlation
over yaw and translation, followed by robust point-to-plane refinement of
several separated candidates. Do not rely on a single ICP initialization in
a repetitive room. Compare competing candidates and inspect named objects
throughout the room. Distant stable structures help reveal alignment drift.

Exclude irrelevant structures or heavily wrong model geometry from registration
when they would dominate correspondences. Do not force scan points in the
unmodelled part of a room onto the small modelled subset. Fix scale when units
are known; do not use arbitrary scale changes to disguise dimension errors.

Human confirmation of the initial overlay is especially valuable here.
Save all transforms, candidate scores, sampled points and an aligned scene.
Freeze registration for subsequent local before/after comparisons. If it must
be revised, version it and reevaluate both models under the revised transform.
Independently aligning each candidate can hide modelling errors.

## 5. Diagnose the mismatch before choosing parameters

Inspect RGB context, plan/elevation slices, local normals and depth histograms.
Look for repeated evidence across height and along a surface. Decide which
kind of error is present:

| Evidence | Likely action |
| --- | --- |
| Several stable faces shift together | Fit a rigid object or assembly pose |
| Fronts disagree while backs must meet a wall | Resolve pose, depth and rear-clearance constraints together |
| Multiple persistent depth levels | Model separate physical features or recesses |
| One drawer/door differs while its parent agrees | Check joint state and local geometry; avoid moving the whole assembly |
| Many distant objects disagree systematically | Recheck registration or units |
| Object footprint aligns but seat/back/handles do not | Improve shape representation separately from pose |

A dominant plane is not automatically the wall. The lab scan contained a
projecting poster, a cable channel, the actual wall face and recessed window
assemblies. Different patches initially suggested different depths. Exposed
strips and depth slices distinguished them.

Ask the person familiar with the room targeted questions: which side is the
room interior, whether a strip is skirting, whether a sill projects, whether
windows/doors were open, or whether furniture moved. This often resolves
ambiguities that more optimization cannot resolve.

Treat scan state and damage explicitly. In the lab, one island drawer was partly
open and the dishwasher handle was damaged and sloping. The closed drawer model
was retained and the damaged handle was left alone. Matching one captured state
is not always the desired nominal map.

## 6. Define observed faces and reserve evaluation data

For each usable face, record its semantic/body association, search bounds,
expected normal, candidate points, selection method and quality criteria.
Use spatial balancing so dense scan areas do not dominate solely through
sampling density. Normal filtering can help isolate surfaces, but must be
documented and fixed for evaluation.

Split spatial blocks into fitting and evaluation sets. A random point split
can put near-duplicate neighbours in both sets. The laboratory used alternating
150 mm tangent-plane blocks. Choose block size for the feature scale and
correlation length; there is no universal correct value.

Use robust losses or training-only inlier selection to fit faces. Retain all
reserved evaluation candidates when reporting errors, including residual
clutter. Never improve a comparison by dropping its difficult evaluation points
after seeing the candidate result. Save the actual arrays/masks, rather than
only a recipe that might select different points on the next run.

If geometry or patch bounds were chosen while looking at the entire preview,
say so. Spatially reserved residuals then provide useful local diagnostics,
but are not independent validation of all modelling decisions. Stronger
validation needs untouched patches, another scan or independent measurements.

## 7. Fit only what the evidence determines

Start with the smallest supported parameter set. A horizontal object often
needs X, Y and yaw; heights and dimensions can remain fixed. One face does not
determine every translation or dimension. Two nonparallel observed sides can
constrain a horizontal placement much better than one side alone. Symmetric
boxes can leave orientation ambiguous by 180 degrees.

State fixed gauges and priors explicitly. For example, a single observed
cabinet front cannot independently establish cabinet depth, rear clearance and
wall position. If backs are constrained flush to a measured wall, the resulting
depth is a constrained effective dimension, not an independently measured back.

Fit assemblies coherently and preserve child transforms and joint conventions.
A diagnostic front-plane translation is not automatically a valid whole-body
move: it may push the cabinet into the wall. Likewise, an unconstrained sofa
fit may penetrate the island even when its observed faces align well.

Useful constraints include floor support, wall adjacency, nonpenetration,
minimum clearance where specified, shared edges and valid joint ranges. For
a box behind a separating plane `n·x = d`, its minimum support is:

```text
n·center - sum(half_extent[i] * abs(n·rotation_axis[i]))
```

Require this to be at least `d + required_clearance` when that plane is a valid
separator for the objects. This is a useful local constraint, not a replacement
for general collision checking. A constraint at zero clearance establishes
nominal contact, not a measured physical gap.

For a window wall, preserve actual wall material around apertures and connect
recess reveals to the recessed assembly depth. Add separate sills and projecting
channels. Missing glass returns do not demonstrate an open passage; document
any conservative closed-window collision panel and nominal backing thickness.

Do not shrink unseen geometry merely because the scan lacks points there.
Distinguish measured faces, inferred continuation and retained nominal values.

### Thin handles and other small features

Check whether the preview resolves the feature before fitting. Retaining one
source record in hundreds can miss a 15 mm handle. Stream a bounded dense patch,
preserving original coordinates/colours and enough mounting-face/edge context.

Use local depth, transverse and longitudinal coordinates. Separate protruding
rails from denser mounting surfaces, inspect coherent bands along their length,
and examine depth sections. Mixed foreground/background returns can form streaks
that should not become solid geometry. Balance sampling spatially and vary
reasonable selection thresholds to assess sensitivity. Neither raw point counts
nor threshold stability constitute independent confidence bounds.

Separate a handle's position from its projection relative to its parent. Matching
depth offsets of handle and drawer suggest checking the front first. Begin with
supported translations; retain thickness, length and angle until cross-sections
and endpoints support changing them. Preserve the parent connection.

Use physical mounting constraints where the scan is weak. A drawer pull can be
centred across its drawer front without fitting noisy lateral returns. Handles
in a horizontal row can share a height, and a drawer column can share a lateral
coordinate. Define the constraint in a common cabinet frame, then move along
each drawer's own mounting direction to preserve depth. Slight frame rotations
mean independent bounding-box centres need not lie on one common line.

Keep these constraints separate from measurements: exact row alignment is a
model requirement, not evidence of submillimetre scan accuracy. Leave door
handles outside drawer-centre rules unless their mounting convention is known.

### Dark surfaces and known dimensions

Known physical dimensions can resolve ambiguity that sparse geometry cannot.
The apartment cooktop was modelled as 57 × 49 cm; the user's 80 × 52 cm
specification was applied before fitting its position. Scaling about an asset
origin can shift an off-centre mesh, so compensate the geometry origin when
the intended physical centre should remain fixed. Match collision bounds
explicitly if requested; equal bounds do not imply equal surface geometry.

For the black glass cooktop, a dense RGB raster and its outer dark-rim
transitions were more useful than an unconstrained surface fit. Reflective
streaks and missing returns inside the glass were not treated as physical
parts. A fixed-size rectangle with translation only improved placement while
retaining height and yaw. Save raster resolution, edge samples and losses;
this colour-based diagnostic is not independent geometric validation.

### Replacing an inadequate shape

Establish pose before increasing shape complexity. A correctly placed box can
still fill a sofa's seating cavity or omit its backrest. Identify physical parts
from RGB/depth sections and model broad surfaces with a few primitives/profiles.
Retain observed asymmetry where supported; avoid representing every wrinkle.

Two seat cushions, two back cushions and arms explained most of the lab sofa's
remaining error. Boolean union with the existing manifold engine removed internal
overlapping faces and produced one connected watertight collision surface.
Concealed backing remained conservative. The taller back required a new worktop
clearance check beyond the old footprint constraint. Its semantic support region
was placed over the seat rather than the back top.

## 8. Measure fit against actual rebuilt geometry

Rendering the model as a sampled point cloud is useful for registration and
viewing. For final surface errors, prefer closest distances to mesh triangles:
the result then does not depend on arbitrary model sampling density and respects
finite face footprints and edges. Use the actual model rebuilt from edited
source, not only the optimizer's proposed transform.

Report two complementary views:

- Corresponding observed-face errors isolate whether a particular feature
  improved. Infinite-plane errors are useful diagnostics but ignore footprints.
- Fixed broader-region scan-to-mesh errors retain surrounding surfaces and
  shape mismatch. They prevent a few chosen faces from being the entire story.

For each, report point count, units, distance direction, median, P90 and fractions
within application-relevant tolerances. Include signed plane offsets when the
direction of a correction matters. Save full CSV results and residual maps.

Reuse exactly the same registration, point arrays, spatial balancing, bounds,
masks and distance implementation before and after. Do not recenter a scoring
crop on each candidate. Nearest-triangle errors may select an old wall's back
face, so they need not equal its room-facing plane offset.

Model-to-scan errors also require care: hidden backs, internal cabinet surfaces
and undersides should not be counted as missing observations. A symmetric
F-score is meaningful only with declared observation/visibility masks, adequate
sampling and a common modelled scope. A partial scene crop and absent scanner
poses may prevent a defensible visibility-aware score. Do not label unmasked
nearest-neighbour scores as precision/recall or scanner accuracy.

Inspect both central error and tails. In the sofa correction, the median
improved strongly while P90 slightly worsened: a correctly positioned solid box
still cannot represent a recessed seat. Diagnose large errors instead of
assuming every residual is clutter or claiming uniform improvement.

## 9. Apply and verify the semantic map

Put supported measurements into documented configuration/dataclass fields where
configuration is useful. For small one-off adjustments, the developer preferred
final mounting insets directly beside construction over a separate offset
hierarchy. Fold accepted corrections into the final values, rather than keeping
nominal values plus historical deltas. Follow the target project's rules and
the developer's preferences for this choice.

Reuse existing geometry and semantic factories. Keep
environment-specific construction in the existing environment file unless a
separate abstraction has a concrete benefit. Avoid introducing a generic fitting
framework merely to complete one environment.

The current SemDT API uses `WorldSource.create_world()` and spatial types from
`krrood_spatial_math.spatial_types`. Build final unbound shapes in a
`BodySpecification`, wrap them in `SemanticAnnotationWithRootSpecification`,
then spawn once. The convenience `get_annotation_specification(...).spawn(...)`
API uses the same mechanism. Nested part bindings can describe doors, drawers
and handles together with their connections.

Generated sofa and wall meshes can therefore be final at spawn time, avoiding
placeholder bodies and later shape replacement. A virtual seating region can
be a root child specification; explicitly associate the spawned region as the
sofa's supporting surface. Child geometry does not populate every semantic field.

Preserve stable names, semantic annotations, articulation, collision/visual
relationships and the world-root convention. Test window apertures in collision
geometry as well as appearance. Keep a physically closed window represented as
an obstacle even if its visual panel is transparent.

Check floor contact, neighbour intersections, endpoint connections and unrelated
body geometry. Use vertex sets or geometric matching rather than assuming mesh
triangle order is stable across rebuilds. If a change needs generated meshes,
use storage appropriate to their lifetime; a serialized world must not retain
paths to files automatically deleted when the producing process exits.

### Recheck children after moving a parent

An assembly correction moves every descendant, including previously fitted
hardware. After raising the apartment oven cabinet by 50 mm, its two drawer
pulls needed their shared row heights restored and its upper door pull needed
its scan-supported height restored. A correct cabinet top did not establish
correct handle positions.

Keep a list of accepted local constraints and rerun it after parent pose,
dimension or joint edits. Check all relevant descendants, including features
previously deferred for inspection. Change only the components supported by
the new evidence; preserve accepted lateral positions, depths and rotations.

Check structural meaning as well as geometry. Cabinet12 was incorrectly copied
as a multi-panel door; the real cabinet needed one full-height, handle-free
door. Handle presence must not be the only criterion for recognizing a Door.
Likewise, a KitchenIsland can be explicitly composed from existing kinematic
members without flattening its cabinets or changing articulation. Verify direct
and context annotation membership in the export, not only annotation counts.

### Mesh assets and editable-source delivery

Keep generated experimental assets persistent. GLB was convenient for local
refinement, but the target iai_maps package used visual DAE and collision OBJ.
When porting back, preserve units, axis conventions, triangle topology and
disconnected collision components. Restore asset-local coordinates and URDF
geometry origins together; otherwise the same correction can be applied twice.
Compare the reloaded root-frame geometry, not only the converted asset bounds.

If Xacro is authoritative, update it together with the generated URDF and check
regeneration equivalence. Verify package-relative resource resolution and
installation rules. Overwrite existing asset paths only in the authorized
target branch; experimental checkpoints and a parked repository port can
represent different revisions and must be identified separately.

### Regression and runtime checks

Write meaningful regression tests before changing the source: measured face or
pose placement, retained dimensions, aperture material removal, closed collision
panels and adjacency/nonpenetration constraints. Follow local policies if a new
test itself turns out to have an incorrect expectation.

Prefer behavioural/geometric invariants and authoritative measurement fixtures
over copying a long list of fitted constants into tests. The developer later
removed the lab's scan-specific numeric regression file. Keep numerical snapshot
audits in the analysis artifacts where appropriate; do not automatically restore
that old test file for the next room.

Use the configured interpreter, check available memory, cap resource use and
run relevant test subsets. Follow repository formatting and generated-file rules.
Do not regenerate or edit unrelated interfaces as part of geometry fitting.

For construction-only refactors, compare body meshes and region poses with a
pre-refactor snapshot. In this case, 118 visual body meshes and 11 region poses
were checked after the final construction/inset cleanup; displacement was at
floating-point roundoff.

Run mesh-producing helpers sequentially where temporary directories are shared
across PID namespaces. SemDT's abandoned-root cleanup identifies owners by PID;
an overlapping helper removed another live helper's temporary meshes during
this work. Keep persistent generated map meshes in appropriate storage and check
memory before starting each helper or test run.

## 10. Make the result reviewable and transferable

Provide an offline interactive overlay with original scan colours, labelled
before/after geometry, feature toggles, axes/units and optional local context.
Keep sufficient context for a human to identify objects. Make display clipping
explicit and expand it as work moves to another part of the room. Draw actual
recess geometry rather than substituting a filled bounding box over openings.

### Dense overlays and trustworthy viewing angles

Retain the original XYZ/RGB and source IDs in dense feature patches. Rendering
limits are separate from evidence limits: the apartment's missing-looking cloud
became visible with 2 px markers and traces of at most 60,000 points, without
downsampling. This is an observed Plotly/browser workaround, not a universal
point limit. Keep scan-only controls to distinguish rendering from registration.

For mounting alignment, provide original-position reference boxes with their
historical transforms, current geometry and original RGB. Distinguish the
original URDF from an intermediate pre-handle-fit assembly; those references
answer different questions.

Perspective and a rotated camera can obscure alignment. Use an orthographic
front/top preset aligned to the relevant cabinet, or a fixed 2D projection with
equal metric axis scales. Apply the same coordinate change to model and scan.
A simple before/after PNG was easier to assess than the cooktop's dense 3D page;
keep both options instead of relying on one large interactive view.

### Overall heatmaps and historical versions

Offer scan-to-triangle and mesh-to-scan directions. The first highlights observed
surfaces unexplained by the map; the reverse can expose model regions without
scan support. Label hidden faces, missing coverage, clutter and partial model
scope. Nearest geometry is not necessarily semantic correspondence. Use the
heatmap to choose the next inspection area, rather than treating every red point
as a modelling error.

Use millimetre units, a shared adjustable colour range, uncapped hover values and
a separate layer for distant points. Preserve RGB context and toggleable mesh
geometry. Whole-room previews remain unsuitable for judging tiny features.
Subdividing existing triangles enables mesh colouring without changing the
surface, but reverse vertex statistics depend on tessellation and are not
automatically area-weighted.

For branch-start comparisons, verify the historical meshes' source revision/hash
and hold registration, scan indices, sampling and colour limits fixed. Keep old
and current output directories separate. A wrapping legend below the plot,
colour bar at the right and explanatory text in normal page flow avoided overlap.

The shared implementation is `scan_previews/alignment/scene_heatmap.py`.
Its distance computation and renderer are reusable, but the default collector,
body groups, scope controls and some explanatory text are case-specific.
Supply explicit current geometry to `SceneAlignmentHeatmap.run(geometry=...)`;
do not silently use its default kitchen world for another environment.

Each HTML is a snapshot, not a live view of the URDF. After accepted edits,
regenerate distances and pages for the affected representations. Record input
hashes, frame, scan selection and geometry policy. Reuse unchanged baselines
instead of recomputing them unnecessarily. Full-world inspection can use a
preview while handle/cooktop pages retain dense original crops; state both the
scored and displayed point counts.

### Exporting the result for colleagues

The world mesh exporter produces `scene.glb`, `manifest.json`, labelled
`mesh.npz` and individual meshes in `bodies/`. Record whether it uses visuals,
collisions or visuals with collision fallback. The apartment's shared export
uses the last policy, including its collision-only wall bodies.

Export from the current provider, reload the GLB, and verify geometry and
semantic membership. Include the bundled label viewer and a README explaining
how to serve it over local HTTP. This viewer needs HTTP for JavaScript modules;
self-contained Plotly pages can be opened directly. A mesh snapshot is not an
editable URDF package and does not require shipping the original laser scan.

Package the complete export directory in a `.tar.gz`, preserve the preceding
release, and verify archive contents against the current directory. Rebuild the
archive after corrections: refreshing a heatmap or source URDF alone does not
update what colleagues receive. Check modified body geometry in the export,
rather than relying only on a successful exporter exit.

Keep these deliverables together:

| Artifact | What it preserves |
| --- | --- |
| Input manifest and source revision | Original scan/model identity, units, poses, attributes and capture context |
| Original meshes and joint state | A baseline that survives source changes |
| Registration transforms and alignment review | Coordinate relationship and correspondence decisions |
| Fitting/evaluation points and masks | Exact evidence used by every reported comparison |
| Parameter values and constraints | Measured values, assumptions and fixed variables |
| Scripts and templates | Reproducible preparation, fitting, evaluation and export |
| Metrics, residual maps and viewers | Improvements and remaining limitations |
| Tests and check results | Semantic/geometric invariants and verification status |
| Short current handover | Completed work, next target, unresolved questions and commands |

Archive temporary scripts and templates in the workspace before handover. Keep
large scans and generated data in suitable artifact storage rather than silently
adding them to source control. Record required paths/hashes and which files must
travel with the analysis. Back up originals before rerunning any script that
could overwrite a baseline.

Transfer a review bundle (self-contained HTML, PNG, CSV and reports), a
reproduction bundle (scripts/resources, frames, baseline meshes, point arrays and
source/environment versions), and the large acquisition archive separately.
Opening self-contained Plotly HTML needs no Python or local server; the bundled
world label viewer requires HTTP as described above. Regeneration requires
adjacent resources, including the heatmap footer JavaScript. Local untracked artifacts
must be copied or archived explicitly; a viewer alone cannot recreate a fit.

Useful compaction checkpoints are after input preparation, confirmed alignment,
a frozen measurement baseline, and each applied/validated object group. Save the
current state and outstanding questions first; a conversational summary alone
is not a replacement for the artifacts.

## Adapting the laboratory scripts to another environment

Start by changing the environment factory and body/group mapping in the model
collector, input paths, coordinate metadata, registration configuration, target
features, candidate bounds and physical constraints. Replace kitchen-specific
names and assumptions throughout the fitting and viewer code. Test transform
direction and synthetic recovery before interpreting a real result.

Before running the scripts on another environment:

1. Create its own source/mesh snapshot and artifact directory.
2. Replace the constructor and body-prefix groups in `KitchenGeometry.from_world()`;
   its defaults remain kitchen-specific.
3. Inspect the new schema before adapting the preparation helpers' scan-0 XYZ/RGB
   assumptions, poses, invalid-record handling and export types.
4. Estimate a new registration; never reuse the lab's `scan_T_root.npy`.
5. Replace fitting bounds, normals, physical constraints and state assumptions.
6. Freeze the new observations, then retain candidate and actual applied meshes.
7. Rebuild source and score both stages on the same points.
8. Supply explicit historical geometry to `SceneAlignmentHeatmap.run(geometry=...)`
   with a separate `output_directory` when comparing older revisions.
9. Save a concise current handover with evidence, metrics, commands and decisions.

Inspect entry points before running them: registration, baseline and surface-fit
scripts overwrite original evidence. The default handle script snapshots the
current model into its original-model filename. The case reproduction guide
lists safe applied-viewer calls and the historical files each helper needs.

Useful starting resolutions from the lab work were 40–60 mm for registration,
10 mm for planar-feature analysis, and 20 mm for broader scoring; local normal
neighbourhoods were approximately 60 mm during fine face analysis. These are
case settings, not required defaults. Choose resolutions and evaluation
tolerances for scan noise, feature size, available memory and robot needs.

The transferable sequence is: preserve inputs, prepare manageable evidence,
align once, confirm correspondences, freeze comparisons, diagnose pose versus
shape, fit supported parameters with constraints, apply source changes, rebuild,
verify semantics/collisions, and report both improvements and unresolved errors.

## Lessons that changed the apartment result

These are examples of choosing the right correction, not constants to transfer
to another room. Detailed evidence is in the apartment stage reports.

| Observation | What helped | Lesson |
| --- | --- | --- |
| Independently fitted drawer pulls shifted sideways | Geometric drawer centres and a shared cabinet column | Constrain poorly observed directions rather than fitting noise |
| Handles in one row varied vertically | Shared scan-supported row heights | Use mounting relationships across several objects |
| Cooktop looked almost aligned but had the wrong footprint | Known 80 × 52 cm size, then dark-rim translation fit | Fix dimensions before estimating placement |
| Large unexplained returns at the island end | Extend the right waterfall from cabinet depth to countertop depth | Missing structure can matter more than pose refinement |
| Oven cabinet looked shorter | Compare equal-height meshes, original joint offsets, Git history and common scanned top | Distinguish misplaced assembly from wrong dimensions; do not invent historical intent |
| Cabinet top aligned but its handles turned red | Recheck all descendants and restore independent mounting constraints | Parent corrections invalidate child placements |
| A dense viewer looked empty or skewed | Visible batched points, original-position boxes and equal-scale 2D views | Viewer problems can imitate modelling problems |
| Export geometry looked good but semantics were incomplete | Inspect direct/context labels and explicit island composition | Geometric fidelity and semantic completeness are separate checks |

The current apartment handover is `scan_previews/apartment/HANDOVER.md`.
The laboratory handover remains `scan_previews/alignment/HANDOVER.md`; its
damaged-handle and partly-open-drawer exceptions do not apply automatically to
the apartment or the next scan.
