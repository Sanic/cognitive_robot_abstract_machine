# Recording refiner walkthroughs

The user accepted the guided video style on 2026-11-15 and requested that it be
reused for subsequent increments. Follow this workflow when demonstrating new UI.

## Presentation contract

- Record the actual working application. Distinguish implemented behaviour from
  planned UI. Use a disposable project copy; retain production history/baselines.
- Highlight the relevant control in yellow **before** activation. Name the exact
  option in the caption, e.g. “Geometry → Scan only”. Pause briefly to read it.
- Draw a visible pointer and a red expanding click pulse. Use real browser mouse
  events for buttons and scan-point selections. Keep all enabling steps visible.
- Pause on the result. Avoid unexplained camera changes or hidden setup between
  successive steps. Caption a camera/framing change when needed.
- Show dense scan loading with model meshes hidden, using a local top/plan view.
  Opaque model geometry can obscure the scan and makes a density demonstration
  misleading. Use identical region bounds and camera orientation for both loads.
- Verify the displayed point counts and zero visible meshes for scan-only shots.
  Inspect decoded video frames: data-array counts alone do not prove visibility.
- For measurements, visibly identify and click both endpoints. Show the result.
- Use concise captions below the browser, readable labels and a small sequence of
  tasks. Aim for roughly 60–90 seconds. Audio is optional; captions must suffice.

## Existing tools, no installation

Project Python: `/home/pmania/venvs/cram-env/bin/python`.
Available recording dependencies: websocket-client, Pillow and imageio-ffmpeg.
The latter supplies its existing FFmpeg binary through
`imageio_ffmpeg.get_ffmpeg_exe()`. A system `ffmpeg` command is not required.
Ask before installing dependencies on another machine.

Use Chromium with software rendering when VirtualGL prevents ordinary capture:

```bash
chromium-browser --headless=new --no-sandbox --disable-dev-shm-usage \
  --use-angle=swiftshader --enable-unsafe-swiftshader \
  --remote-debugging-port=9229 \
  --remote-allow-origins=http://localhost:9229 \
  --user-data-dir=/tmp/semdt-refiner-video-browser about:blank
```

The GUI/browser sandbox may require the existing Chromium launch approval.
Choose free service/debug ports. Connect to the page WebSocket advertised by
`http://127.0.0.1:9229/json`, using the matching Origin header. Set the viewport
with `Emulation.setDeviceMetricsOverride` (1440×900 at scale 1 for the accepted
video). Wait for both renderer initialization and the action's completion.
Readiness expressions must return booleans: wrap DOM-node checks with `Boolean`
or `!!`. CDP serializes DOM objects as empty dictionaries with returnByValue,
which otherwise looks false to a Python polling loop.

Use `Page.captureScreenshot` to obtain bounded JPEG frames. Composite each frame
with Pillow into a 1440×1000 canvas: browser above, 100-pixel caption band below.
Use a dark caption background, 28-pixel title and 20-pixel explanatory text.
Calculate highlight/pointer positions from live `getBoundingClientRect()` values
and scroll the target into view before recording the instruction.

Stream RGB frames to FFmpeg rather than retaining a whole movie in memory.
Close its input and wait for encoder completion in a `finally` block; otherwise
an interrupted recording can lack its MP4 metadata. Never publish that partial file.

Encoding arguments:

```text
-f rawvideo -pixel_format rgb24 -video_size 1440x1000 -framerate 8 -i pipe:0
-an -c:v libx264 -threads 2 -preset fast -crf 20 -pix_fmt yuv420p
-movflags +faststart output.mp4
```

UI tutorials can use eight captured frames per second. Preserve their intended
pause durations explicitly; capture time is not a performance measurement.
Trim waiting periods only when this does not imply an operation is instantaneous.

## Checks and publication

1. Save a chapter list containing start times, captions and verification results.
2. Verify codec, dimensions and duration with imageio-ffmpeg's existing reader.
3. Decode and inspect representative frames: scan loads, sections, measurements
   and saved-view/history actions. Check for clipped captions and offscreen targets.
4. Record browser errors; fix unexplained failures before publishing.
5. Publish a versioned MP4 and atomically update the stable video URL only after
   verification. Keep the preceding version available. Test the HTTP link.
6. Stop the temporary browser and recording server. Leave the user's viewer up.

The accepted revision-2 reference is
`projects/functionality-video-v2-20261115/refiner-walkthrough-v2.mp4`:
71.5 seconds, H.264, 1440×1000, about 2.47 MB. Its `chapters.json` and decoded
frames document the result. The 50k/150k demonstrations start at 26.5/34.25 s.
These generated artifacts are Git-ignored; the workflow in this file is tracked.

The staged-workspace reference is
`projects/workspace-tour-20261115/workspace-walkthrough.mp4`: 60.5 seconds,
H.264, 1440×1000. The matching `record_walkthrough.py`,
`browser_connection.py`, chapters and decoded frames are archived beside it.
The published copy is `projects/apartment/workspace-walkthrough.mp4`.
World-axis orientation starts at 18 s, dense scan detail at 28.75 s, the
plan drag at 35.75 s and the regional heatmap at 48.5 s. The final take
was decoded and checked after fixing the interactive green-to-red palette.

The directional-results reference is
`projects/coverage-check-20261115/comparison-walkthrough.mp4`: 55.5 seconds,
H.264, 1440×1000. Its archived `record_comparison_walkthrough.py`,
`comparison_browser_connection.py`, `comparison-chapters.json` and decoded
frames reproduce the sequence. The published copy is
`projects/apartment/comparison-walkthrough.mp4`. Results begin at 15.75 s,
body filtering at 29.5 s and the unsampled-body distinction at 40 s.

When recording another job in a project with existing results, retain the old
job IDs before submission. Wait for the *new* ID to succeed and activate that
card's result button. Checking that any result button exists can select an older
job. Verify the displayed evaluation count and frozen request against the new job,
then inspect decoded frames. Tab changes should start at the inspector heading;
scroll targets into view only when their action is introduced.

The constrained-translation reference is
`projects/refinement-tour-20261115/refinement-walkthrough.mp4`: 54.25 seconds,
H.264, 1440×1000. It is explicitly a synthetic panel with a known 20 mm error,
not a claim about an apartment defect. Source scripts, chapter metadata and
representative decoded frames are beside the clip. Publication is
`projects/apartment/refinement-walkthrough.mp4`. Locks are chosen at 13.25 s,
proposal review starts at 23.5 s, current/proposed toggles at 32.5/35.25 s,
application at 39 s and Undo/Redo at 43/45.75 s.

The verified-box dimension reference is
`projects/dimension-tour-20261115/dimension-walkthrough.mp4`: 61.25 seconds,
H.264, 1440×1000. It explicitly uses a synthetic 40 mm width discrepancy and a
smaller collision. Archived recording scripts, `dimension-chapters.json` and
decoded frames are beside it; the published copy is
`projects/apartment/dimension-walkthrough.mp4`. Box mode starts at 9 s,
anchors at 16.75 s, proposal review at 26.25 s, Undo/Redo at 44.75/47.5 s,
and the retained smaller collision at 54.25 s. Do not present it as an apartment
correction: the current apartment meshes fail this first root-box eligibility rule.

## Full collection and rotated-box increment — 2026-11-16

[FEATURE_VIDEOS.md](FEATURE_VIDEOS.md) indexes seven guided videos, totaling 350 s.
The native-video gallery is `projects/apartment/videos/index.html`; its portable
archive contains MP4s, posters, chapters and the offline gallery. Four new takes
cover navigation (27.5 s), upright registration/cancellation (37.25 s), current
inspection controls (53.5 s) and oriented-box dimensions (63.5 s). Existing verified
workspace/crop, directional-results and translation recordings complete the set.
Archive scripts and decoded frames in their corresponding `*-tour-20261116`
directories before stopping temporary servers and the recording browser.

The oriented-box take is explicitly synthetic, with a rigidly rotated box and
separate scanner registration. Verify all three frame labels, required-face
counts, fixed center/orientation, 1000→1040 mm visual width, unchanged depth/height,
Undo/Redo and 980→1019.2 mm collision width. Browser reload after regenerating an
applied viewer must invert the complete embedded stretch matrix, including its
off-diagonal components; verify original dimensions return on Undo.

Do not wait for a specific history label after a no-op action: identical project
states create no new revision. Wait for the actual resulting state instead.
Likewise, an indexed whole-scan overview can read many tiles; use local regions
for dense-detail demonstrations and identify overview versus regional evidence.


## Complete apartment use-case — 2026-11-19

[APARTMENT_USE_CASE_VIDEO.md](APARTMENT_USE_CASE_VIDEO.md) documents the 135.75 s
full workflow. Real apartment registration and dense evidence are used, with a
deliberately incomplete starting placement and a disclosed +30 mm panel error in
a disposable source URDF. The production apartment is unchanged. Before/after
local median distance is 21.35→0.83 mm using identical original records. The story
finishes with Undo/Redo, source review and a validated native URDF/mesh bundle.

Recheck that the actual target is enabled and outside an inert subtree immediately
before clicking, after the pre-action caption pause. Wait for the requested stage
or tab to become active, not merely for a loading panel to disappear. Automatic
project refresh can start during the caption pause. After field changes, wait for
any redraw to finish before entering the next bound. Move the real pointer off the
scene for clean result shots so hover labels do not cover the evidence.

The recording retains the established visible pointer/highlight/pulse style and
explicitly labels shortened calculation waits. It verifies dense scan visibility,
identical before/after records, fixed crop/role/colour scale, axis locks and native
export reload before publication. Decode the final encoded movie, not only source
screenshots. The gallery puts this complete use-case first and retains the seven
focused feature clips and the older archive.

## Working grouped proposal review — 2026-11-21

The standalone [live review recording](http://127.0.0.1:8815/videos/agent-review-live.mp4)
uses the actual clipboard copy, CLI operations and browser controls. The CLI card
summarizes the real request and returned proposal ID. Recorders, source frames,
requests and chapter metadata are in the ignored
`projects/agent-review-live-20261121/` directory.

The first take exposed a camera reset after focusing and orbiting. For geometry
redraws, read the renderer's live camera when available. The final take checks
that Original and Proposed share the orbit camera, ranges, visible layers and
all 50,000 scan record identities. It checks three displacements and a single
Undo restoring the prior model revision. Full MP4 decoding and timestamped frame
inspection passed.

## Paired evidence cutaways — 2026-11-22

The [heatmap edition](http://127.0.0.1:8815/videos/agent-review-heatmap.mp4) records
a fresh proposal based on a fit of the apartment island's back panel along world X.
The three panels share its −19.43 mm translation and the countertop stays fixed.
A separate orthographic plot inserts the original heatmap just after loading the
crop and the paired heatmaps just after the Proposed preview. This lets the video
show measured residuals while the live review still uses RGB.

Reopen the exact copied crop and verify all source fields against the browser's
binary region response. Both heatmaps use the same 50,000 records, registration,
orthographic projection and 0–40 mm colour scale. Distances are to the visual
triangles of the three panels and fixed countertop. Floor points can remain red
because floor geometry is outside that comparison. Median 18.88→4.13 mm and
59.73→89.42% within 20 mm are local, in-sample results for this proposal.
The footage verifies camera/scan continuity and a single Apply/Undo. Captures,
scripts, point evidence and chapter metadata are in the ignored
`projects/agent-review-heatmap-20261122/` directory.
