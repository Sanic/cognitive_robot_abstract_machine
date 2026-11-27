# Guided feature videos

[From whole-apartment alignment to handle refinement (2:23)](http://127.0.0.1:8815/videos/apartment-handle-refinement-v7.mp4)
starts with the entire scan and SemDT model apart, runs the automatic placement
search, and previews the generated fit. It compares two placements using the
same whole-apartment sample and accepts one before selecting the cabinet. The rest follows
manual handle selection, the Codex handoff, paired 3D heatmaps, and both handles
returning to their original positions after one Undo from a shallow angle.
See [the measured handle case](APARTMENT_HANDLE_CASE.md) for the evidence and
pending interactive review.

[Select and inspect the kitchen island assembly (0:59)](http://127.0.0.1:8815/videos/assembly-selection.mp4)
shows the 39-body assembly focus, 50,000 dense points in Scan only mode, drilling
down to Cabinet9 and copying the complete context. The agent reopened the
recorded crop and read 1,000 original records. Capture scripts, chapter metadata
and verified handoff are in the ignored `projects/assembly-video-20261123/` directory.

[Live paired heatmaps in the 3D viewer (0:50.5)](http://127.0.0.1:8815/videos/paired-3d-heatmaps.mp4)
records **Show paired heatmaps**, an orbit, and the Original/Proposed toggle in
the regular viewport. At 0:28.5 it switches to Scan only so the change on the
back panel is easy to see. Both states use the same 50,000 points and 0–40 mm
scale. Source footage, checks and chapter metadata are in the ignored
`projects/paired-3d-video-20261122/` directory.

[Apartment proposal review with heatmaps (1:28.5)](http://127.0.0.1:8815/videos/agent-review-heatmap.mp4)
uses a scan-fitted 19.4 mm sideways shift and shows the original heatmap at 0:46
and the matched comparison at 1:02.5. Local median distance drops from 18.9 mm to
4.1 mm on 50,000 identical records. The video keeps the interactive RGB review
and inserts measured orthographic comparisons with a common 0–40 mm scale.

[Working apartment proposal review (1:11.5)](http://127.0.0.1:8815/videos/agent-review-live.mp4)
records the live viewer and an actual CLI submission. It follows Island Back
selection, context copy, three explicit panel edits, the linked scan, orbit,
Original/Proposed, Accept and a single Undo. The 30 mm movement is illustrative
and takes place on a disposable apartment copy. The prior design preview and the
eight-video archive remain separately available.

The standalone agent-assisted review design preview (1:07) opens with the
working `apartment/island_back` selection handoff. Labeled mockups then show
a three-panel change set, comparison on the same illustrated evidence, acceptance
and one Undo. [Watch the design preview](http://127.0.0.1:8815/videos/agent-review-preview.mp4).

Current collection: <http://127.0.0.1:8815/videos/>.
Download the eight-video collection (about 14 MB):
<http://127.0.0.1:8815/refiner-feature-videos-20261119.tar.gz>.
Extract it and open `refiner-feature-videos/index.html` for offline playback.
The recordings total approximately **8 minutes 6 seconds**, including the full apartment use-case. Each card has clickable chapters,
a direct MP4 download and chapter metadata. These are local, Git-ignored artifacts
under `projects/apartment/videos/`; restart the apartment server to serve them.

| Clip | Duration | Demonstrated functionality |
|---|---:|---|
| [Complete apartment use-case](http://127.0.0.1:8815/videos/apartment-usecase.mp4) | 135.75 s | Real apartment scan; automatic registration, local crop/heatmaps, constrained correction, Undo/Redo and validated URDF export. Disclosed demo error. |
| Representations and navigation | 27.5 s | Visual/collision/both/scan-only, perspective/orthographic views, actual mouse rotation after switching geometry |
| Automatic registration | 37.25 s | Upright confirmation, landmark-free search, cancellation, candidate review, explicit acceptance, Undo/Redo |
| Orientation, cropping and heatmaps | 58.5 s | Staged workspace, scan/world axes, tight body regions, graphical plan crop, regional worker, green-to-red heatmap, history |
| Dense scan inspection | 53.5 s | Prepare provenance, object search, visible 50k/150k original records, display clipping, metric sections, measurements, annotated views, Undo/Redo |
| Results and evidence support | 55.5 s | Two comparison directions, body metrics/filtering, sampled versus unknown evidence, report download, cached-result history |
| Constrained body translation | 54.25 s | Explicit axis locks/bounds, evidence diagnostics, ghost review, Apply, Undo/Redo |
| Box dimensions, including rotated boxes | 63.5 s | Verified geometric frame, extent locks, face anchors/support, full stretch preview, Apply, Undo/Redo, smaller collision preserved |

Yellow outlines identify controls *before* use, a visible pointer shows the action,
and red pulses mark activation. The new recordings follow the current UI;
workspace/crop, results and translation clips reuse the verified 15 November takes.
The workspace clip ends before its now-outdated roadmap closing caption.

Inspection/crop/results recordings use apartment evidence. Registration and geometry
editing demonstrations are explicitly synthetic. The rotated-box example fits
1000 to 1040 mm visual width and 980 to 1019.2 mm collision width while preserving
orientation and the other dimensions. No apartment geometry overrides were applied.

Source recordings, scripts, chapter logs and decoded verification frames remain in
`projects/*-tour-20261116/` and the earlier tour directories. See
[VIDEO_WORKFLOW.md](VIDEO_WORKFLOW.md) to reproduce the recording style. Browser
checks reported no unhandled errors. Encoded frames were decoded to check scan
visibility, measurements, registration overlap and dimensional proposals.

These videos cover interactive workflows. The full apartment use-case includes
source review and export. Shared alignment constraints, rotations, assembly
propagation, gravity/masks and new-object fitting remain follow-up work.


The seven feature clips predate the latest UI improvements. The full apartment
use-case records the current workflow, including source-copy export. See
[APARTMENT_USE_CASE_VIDEO.md](APARTMENT_USE_CASE_VIDEO.md) for its disclosure,
measurements and reproduction artifacts, and [SOURCE_APPLICATION.md](SOURCE_APPLICATION.md)
for the source-copy contract.
