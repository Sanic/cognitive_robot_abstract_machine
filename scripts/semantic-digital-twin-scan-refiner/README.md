# SemDT scan refiner

Reusable local tools for inspecting an approximate SemDT against a measured
scan. Work is isolated to this directory on `semdt-world-refiner`, created from
`origin/main` at `d47537cf910577321df4757fa76253785bd63c59`.

## Current working slice

- Load a trusted Python world provider or URDF through SemDT.
- Freeze actual visual and collision geometry independently in root coordinates.
- Import a PLY/NPZ preview, retaining RGB bytes and preview-record identities.
- Search upright yaw/translation automatically and retain ranked alternatives.
- Review candidates in one self-contained browser page with geometry layers,
  orthographic top/front views and an optional scan-to-triangle heatmap.
- Accept a reviewed transform explicitly and save `scan_T_root.npy`.
- Index all E57 stations through an existing isolated reader, retaining raw
  record/station IDs, float64 coordinates, byte RGB and available intensity.
- Click a model body to select surrounding scan evidence and load 10k/50k/150k
  original regional records through the local server.

The viewer uses the installed Plotly bundle and needs no external scripts.
There are no added dependencies or package installation steps. The tools use
the project's configured interpreter. Providers execute trusted local Python.

This is an implementation increment, not the complete planned system.
Registration runs through the CLI; the browser currently reviews its results.
Regional detail loads on demand; continuous camera-driven level of detail and
web-triggered jobs remain future work. Initial PLY/NPZ import loads a bounded
preview into memory; dense E57 indexing streams bounded batches.

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
The current geometry snapshot does not yet serialize the semantic graph.

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
Numeric bounds are inclusive, in original scan coordinates and metres.
These are on-demand regional samples, not automatic camera-driven streaming.

The Python API also exposes `SpatialScanIndex.batches(region)` for every original
record in bounded batches. The CLI exports a bounded regional sample:

```bash
/home/pmania/venvs/cram-env/bin/python scripts/semantic-digital-twin-scan-refiner/refine.py crop \
  --project scripts/semantic-digital-twin-scan-refiner/projects/example \
  --lower 0 0 0 --upper 1 1 1 --maximum-count 50000 \
  --output scripts/semantic-digital-twin-scan-refiner/projects/example/region.npz
```

Both paths preserve original coordinates, IDs and available attributes. Loading
a different selection disables the old heatmap because its point identities no
longer match; regional heatmap jobs will be added separately.

## Small rendering improvements

The viewer shares one scan-position buffer between RGB and heatmap modes, uses
compact float32 typed arrays around a recorded local origin, omits duplicate
mesh vertices, and updates affected layers only. The stored scan/model remain
full precision. Point size, hover and mesh opacity are adjustable; hover starts
off and meshes start opaque. No rendering-library dependency was added.

Left-button dragging starts in free rotation. Top, Front and Perspective restore
rotation after other Plotly drag modes; Rotate restores it without changing the
current viewpoint. This avoids remaining in pan/zoom mode after a view change.

The apartment page decreased from 39,333,815 to 20,045,029 bytes (49%) with the
same displayed scan selection and heatmap. This is a payload measurement, not a
browser frame-rate benchmark. A graphical browser was unavailable for visual QA.

## Interpretation and persistence

The browser's support value is the fraction of sampled model points within the
configured distance of sampled scan points. It is a registration diagnostic,
not confidence, independent accuracy or a complete quality report.

The heatmap measures unsigned scan-to-triangle distances for one displayed
candidate and geometry role. Switching candidates disables that heatmap;
generate another with `view --candidate N --heatmap`. It uses reproducibly
selected preview records and does not inspect high-resolution raw evidence.
Coverage-aware model-to-scan comparison remains a follow-up.

Transforms follow SemDT conventions:

```text
p_scan = scan_T_root @ p_root
p_root = inverse(scan_T_root) @ p_scan
```

These are homogeneous column vectors. The browser transforms the model into
scan coordinates, then subtracts a recorded common display origin before using
float32 buffers. The footer reports that origin. Accepted transforms are stored
separately from proposals. Running `align` again clears previous acceptance and
replaces candidates; registration revision history is not yet implemented.

`world.npz` and `scan.npz` are frozen baselines. `project.json` records source
identity, scale and hashes; candidate JSON retains settings. Derived files are
written inside the project. The tool does not edit the imported URDF, meshes or
Python world. Undo/replay of geometry refinements and source writing are planned.

The local server provides read-only artifacts and regional queries on loopback. It exposes the selected
project directory, including snapshot files; it is not an authenticated sharing
service. Stop it with Ctrl+C. Generated projects are Git-ignored.

## Development

The focused tests include registration recovery, baseline persistence, actual
URDF geometry origins, point-to-triangle distances, regional boundary/provenance
checks, HTTP byte-layout decoding and bounded overview access. E57 integration
tests skip unless `SCAN_REFINER_E57_PYTHON` names an existing reader interpreter.

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

See [DESIGN.md](DESIGN.md) for the architecture and [ROADMAP.md](ROADMAP.md) for
accepted priorities and deferred work.

