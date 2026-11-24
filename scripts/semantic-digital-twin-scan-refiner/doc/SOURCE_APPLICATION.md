# Reviewed source application and export

Use **Refine → Review & export source copy** after accepting geometry proposals.
The workflow includes only accepted snapshot overrides. Unapplied proposals are
excluded. No scan registration is baked into the model: source geometry remains
in its original model-root coordinates.

1. **Review source changes** reloads the original source and checks that every
   labelled geometry role reproduces the frozen baseline. The review freezes the
   source content digest, frame/joint/semantic membership and accepted overrides.
2. Inspect the body list and accepted changes in the current model view.
3. **Write validated copy & export** applies the frozen edits to a fresh live world,
   writes a separate source copy, reloads it and checks the accepted geometry again.
4. **Download source & world bundle** provides the completed tar.gz. Failed
   validation does not publish a source bundle or overwrite an original asset.

A geometry change or Undo invalidates the review. Restoring the exact reviewed
geometry with Redo makes its immutable completed output applicable again. Source
file changes require a fresh review. Source-copy publication is a derived artifact;
it does not rebase the project or clear its geometry history. Undo the accepted
geometry edits and prepare a new copy when you want another source revision.

## Output contract

Outputs live under `projects/<project>/source-reviews/<review-id>/`. The completed
`bundle/` directory and `world.tar.gz` contain:

| File | Purpose |
|---|---|
| `refined.urdf` | Updated native URDF for a URDF source; original names, joint kinds/axes/limits and non-geometric XML metadata are retained |
| `assets/` | Independent OBJ meshes, with associated MTL/textures where supplied by the existing mesh exporter |
| `refined_world.py` | `RefinedWorld.create_world()` provider for the copied source or provider recipe |
| `refinement.json` | Frozen accepted transformations/dimensions and original source identity |
| `review.json` | Source content digest, changes and reviewed topology/semantic membership |
| `baseline.npz`, `world.npz` | Labelled before/after geometry snapshots with visual and collision roles kept separate |
| `scene.glb` | Labelled **visual** geometry in model-root coordinates |
| `collision.glb` | Labelled **collision** geometry in the same coordinates |
| `manifest.json` | Typed validation result and live-world entity counts |

GLBs are ordinary trimesh scene exports with body-named nodes. The manifest is the
refiner's source-copy contract, not the Warsaw mesh exporter's manifest schema.
The generated live-world provider can feed that exporter when it is available.

URDF primitive boxes stay parametric. Mesh-backed geometry uses the existing
SemDT/trimesh mesh implementation and OBJ export. Independent assets are written
for the copy; source mesh files remain unchanged. Reimported vertices must agree
bidirectionally within 2 micrometres; surface identities, triangle counts and
surface area are also checked. GLBs are display exports; NPZ/source validation
uses the more precise source geometry.

For a Python-defined SemDT, the output is a **provider recipe**. It reloads the
original provider, checks its geometry against the bundled baseline, and applies
accepted edits while retaining its semantic objects and articulation. It does
not rewrite arbitrary Python source. The original provider and its dependencies
must remain importable. A URDF source bundle instead includes its independent
refined URDF and mesh assets.

## Supported changes and validation limits

- Both geometry roles receive the same accepted mapping and retain their original
  size differences. No role substitutes for an absent role.
- Translations update incoming fixed/prismatic placements and compensate outgoing
  placements so unedited children keep their world poses. Root-body translations
  and translations moving an incoming rotating hinge are rejected.
- Dimensions support verified boxes, including mesh-backed/oriented boxes. Box
  primitives retain their orientation and parametrization; resized mesh assets
  retain their corresponding local origin and appearance where supported.
- Resizing a body with inertial properties is rejected until a mass/inertia policy
  has been reviewed. This tool does not infer physical mass from scan geometry.
- Live-world validation and reload checks preserve frame names, graph edges, joint
  types, axes, DOF names/limits and semantic classes/referenced frames. This checks
  consistency, not independent ontology correctness or collision-free operation.
- URDF loading preserves what the URDF parser supplies; it does not automatically
  invent semantic annotations with WorldReasoner. Include reasoning in a trusted
  Python provider when those inferred annotations are part of the desired source.
- Original sources must still be available. Older projects froze geometry rather
  than source code: non-geometric source history before the first source review
  cannot be reconstructed from their NPZ baseline.

## CLI and loading an exported world

Use the configured interpreter from the repository root:

```bash
/home/pmania/venvs/cram-env/bin/python \
  scripts/semantic-digital-twin-scan-refiner/source_task.py \
  --project scripts/semantic-digital-twin-scan-refiner/projects/apartment \
  --action review

/home/pmania/venvs/cram-env/bin/python \
  scripts/semantic-digital-twin-scan-refiner/source_task.py \
  --project scripts/semantic-digital-twin-scan-refiner/projects/apartment \
  --action publish --review <review-id>
```

Each operation prints its typed JSON result. HTTP uses the same process and
configured interpreter, with a 16 GiB address-space ceiling. The local endpoints
are POST `/api/source/review` and POST `/api/source/publish`; the latter takes the
review ID as plain request text. Publication only writes inside that project.

Add the extracted bundle directory and `scripts/semantic-digital-twin-scan-refiner`
to `PYTHONPATH`, then use `refined_world:RefinedWorld` as a world provider. It
resolves URDF-relative assets against the bundle rather than the process's working
directory. For direct parser use, supply
`SourceApplication.bundle_resolver(directory=<bundle-path>)` as `path_resolver`.
The generated provider is the easiest reload entry point.

## Current apartment verification

On 2026-11-18, a native source copy reloaded with 115 bodies and 114 joints and
matched both frozen geometry roles. The URDF parser supplies zero semantic
annotations; that is reported explicitly. No additional apartment geometry
changes were accepted during this infrastructure work. The download is a verified
copy of the current refined baseline, not a newly optimized apartment.


Workers now request a 16 GiB address-space ceiling. A lower inherited hard limit
from the launcher is respected, including memory-capped tests. The shared policy
lives in `scan_refiner/process_memory.py`; running workers retain their existing
limits, while new analysis/source processes use the updated default.
