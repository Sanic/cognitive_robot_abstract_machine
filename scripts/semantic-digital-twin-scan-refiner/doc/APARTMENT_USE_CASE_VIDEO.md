# Apartment: scan to a validated world

[Watch the complete use-case](http://127.0.0.1:8815/videos/apartment-usecase.mp4)
or open the [video gallery](http://127.0.0.1:8815/videos/).
Duration: **2 min 16 s**. H.264, 1440 × 1000, approximately 5.3 MB. Captions,
yellow control outlines, a visible pointer and activation pulses explain the actions.
Processing waits are shortened and labelled. There is no narration requirement.

This is one continuous apartment workflow, rather than a collection of unrelated
feature demonstrations. It uses real Apartment-Lab.e57 evidence and a disposable,
self-contained apartment URDF copy. Its initial placement deliberately supplies
height only. A **+30 mm world-Y error is deliberately introduced** in
`apartment/island_waterfall_right` before recording. The persistent demo badge and
opening captions disclose this. The demonstration does not edit the working
apartment or claim to discover a new defect in it.

| Approximate time | Action and purpose |
| --- | --- |
| 0:00 | Show the apartment model and real indexed laser scan. |
| 0:05 | Inspect the initial misalignment, confirm vertical Z, run automatic registration, review and accept a candidate. |
| 0:24 | Orient world axes, select the island panel, focus and crop away floor/countertop clutter. |
| 0:54 | Compare 10,000 original scan records with visual triangles and inspect the warm-coloured mismatch. |
| 1:07 | Keep X/Z locked, unlock Y, set a movement bound and review a translation proposal. |
| 1:32 | Apply the reviewed correction; demonstrate Undo and Redo. |
| 1:51 | Recompute the same crop and inspect the improved heatmap and measurements. |
| 1:57 | Review the accepted source change, write a separate copy and validate its reload. |
| 2:06 | Show the downloadable URDF/mesh/visual-and-collision bundle. |

The selected face's sampled median distance changes from **21.35 mm to 0.83 mm**;
P90 changes from **22.52 mm to 2.37 mm**. Returns within the 20 mm tolerance change
from **7.4% to 95.2%**. Before/after heatmaps use the identical original point records,
crop, registration, surface role and point budget, with a fixed 0–30 mm colour scale.
These are local, in-sample diagnostics, not an independent accuracy estimate or a
claim about unobserved areas of the apartment.

The export reload validation reports 115 bodies and 114 joints. The URDF parser
supplies zero additional semantic annotations, which is reported honestly in the UI.
The bundle contains a native URDF, mesh assets, visual/collision GLBs and the source
review/override records. Original source files are retained.

## Recording and verification artifacts

Generated, Git-ignored artifacts live under:

```text
scripts/semantic-digital-twin-scan-refiner/projects/apartment-usecase-20261119-v1/
```

The directory retains the recording/preparation/publication scripts, deliberately
rough source copy, disposable project, preflight jobs, chapter/check metadata,
verified export and decoded final-video frames. `setup.json` records the introduced
error and the intended crop; `verification.json` records final encoding and measured
checks. Earlier pilot takes are retained separately and are not published.

The final MP4 was fully decoded without diagnostics. Eight representative encoded
frames were inspected for scan visibility, registration, heatmaps, proposal review
and export. Dense-evidence capture verifies 50,000 displayed scan points and zero
visible model meshes. Publication verifies identical before/after point records
(including matching unavailable-intensity values), crop/settings, Y-only locks and
a validated native source export. The gallery's chapter playback is checked in a
real browser.

See [VIDEO_WORKFLOW.md](VIDEO_WORKFLOW.md) for the reusable capture method. New
recordings should recheck loading/inert/disabled state at activation time, after the
pre-action caption pause, and wait for the specific stage/tab result. Background
project refresh can start while the explanatory caption is being recorded.
