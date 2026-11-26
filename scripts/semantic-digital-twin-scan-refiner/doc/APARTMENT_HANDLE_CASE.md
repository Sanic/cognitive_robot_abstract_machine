# Apartment walkthrough — refine two Cabinet 3 handles

The [walkthrough (2:23)](http://127.0.0.1:8815/videos/apartment-handle-refinement-v6.mp4)
opens on the entire laser scan and SemDT apartment model in different
horizontal coordinates. The recording starts with a rough height match, runs
the refiner's upright placement search, and previews the whole-world fit in
plan. Both dropdown entries are scored across the entire apartment. The
recording selects the saved placement with slightly higher overall support and
accepts it once. It then switches the
inspection view from scan axes to world axes so the rotation is visible before
the cabinet is selected.

The generated and saved placements are evaluated against the same 8,000
visual-model surface samples and 20,000 scan preview samples. With the same
80 mm support distance, the two scores are **40.9%** and **41.2%**. The median
distances are 99.0 mm and 102.6 mm. Local cabinet measurements appear separately
in the handle investigation.

## Where the tree and group come from

The World tree uses parent–child links from the loaded SemDT world and the
world's entity names. This works the same way for imported URDFs and directly
constructed SemDT worlds. The tree shows each Cabinet 3 handle under its door.
The video creates **Cabinet 3 handles** explicitly from those two bodies with
**Save a named assembly**. It then appears under **Groups** as a saved selection.

## What the scan supports

Two recorded placements were checked against 1,600 probes on the visible
Cabinet 3 door faces. The whole-apartment candidate gave a 47.6 mm median
distance, while the saved kitchen placement gave 2.1 mm for the same surfaces.
The door seams also line up in the front projection under the kitchen
placement. The investigation freezes this placement for all comparisons.

Cabinet 3 has two horizontal door handles whose original lateral mounts are
already off-centre on their respective doors. The current mounts are within
0.6 mm of those original sideways positions. The exposed rail returns extend
farther along world +Y. Fits on alternating 4 mm lateral blocks favour a
+18 mm shift at the upper handle and +22 mm at the lower handle. The proposed
change set uses +20 mm world Y for both handles. It preserves their common
mounting line, the existing heights and depths, and the surrounding geometry.

The ending keeps the same shallow angle through acceptance and Undo so the
handles stand out from the doors. The recording verifies that one click restores
the original positions of both handles. The final front-view comparison shows
their 20 mm return against a fixed reference line.

| Held-out rail evidence | Occupied cells | Median, original → proposed | P90, original → proposed |
| --- | ---: | ---: | ---: |
| Upper handle | 795 | 3.05 → 2.43 mm | 7.01 → 5.59 mm |
| Lower handle | 1,144 | 3.82 → 3.12 mm | 7.04 → 6.34 mm |

The table uses disjoint blocks from one capture under a shared registration.
The live paired heatmap uses a separate 10,000-record crop over the exposed
rail area and scores against all visual triangles in both states. Its median
distance falls from 3.53 to 3.01 mm. In the two dashed right-end windows shown
in the close-up, the share of the same 2,141 points farther than 10 mm drops
from 29.3% to 0.2%. This is a local endpoint comparison. A dedicated rail
check also compares model probes back to the scan to test support at both ends.
Axis-aligned boxes in the diagnostic image are projections of the model bounds.
The heatmaps use the actual visual triangles.

Cabinet 6 retains its previously aligned column. The survey also found
sideways discrepancies on the island, including Cabinet 11, where the drawer
seams and handles both appear offset. Their shared shift needs a cabinet
placement check before proposing handle-only edits.

On an accepted trial copy, the existing native-source review reloaded the
original URDF, matched its frozen geometry, and identified exactly the two
Cabinet 3 handles as changed.

## Review and reproduction

Open [the pending review](http://127.0.0.1:8838/), go to **Refine**, and choose the
Cabinet 3 change set under **Agent proposal**. Select **Show paired heatmaps** and
toggle **Original / Proposed**. The recorded Accept and Undo use a separate copy.
The original apartment URDF and accepted source are preserved.

The ignored `projects/apartment-handles-walkthrough-v6/` directory retains the
whole-apartment scores and single placement acceptance take.
The `projects/apartment-handles-walkthrough-v4/`
directory retains the oblique Undo footage. The `projects/apartment-handles-walkthrough-v3/`
directory retains the whole-world registration job and fixed plan comparison.
The `projects/apartment-handles-walkthrough-v2/` directory retains the
placement sections, endpoint comparison, copied context, submitted request,
recording scripts and video chapters. The original measurement scripts and
hold-out results remain in `projects/apartment-handles-review-20261125/`. Both
runs reuse the frozen apartment evidence and scan index. See
[VIDEO_WORKFLOW.md](VIDEO_WORKFLOW.md) for browser capture and playback checks.
