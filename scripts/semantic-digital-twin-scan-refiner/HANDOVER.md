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
