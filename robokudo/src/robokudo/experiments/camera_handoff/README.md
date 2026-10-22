# Two-camera approach experiment

Run from the repository root with the workspace environment active:

```bash
source /opt/ros/jazzy/setup.bash
source /home/pmania/ros_ws/install/setup.bash
/home/pmania/venvs/cram-env/bin/python -m robokudo.experiments.camera_handoff
```

The command prints the recording directory. By default it creates a timestamped
directory under `/tmp/robokudo-camera-handoff`. To choose a destination:

```bash
/home/pmania/venvs/cram-env/bin/python -m robokudo.experiments.camera_handoff --output /tmp/my-approach
```

Use a fresh destination for each run. No robot, Giskard, ROS master, display,
downloaded model, or external service is needed. ROS message and behavior-tree
packages from the configured environment are still required.

## Textured models

The repository's Cheez-It box and cup can be selected explicitly:

```bash
python -m robokudo.experiments.camera_handoff --model obj_000001 --output /tmp/cheez-it-approach
python -m robokudo.experiments.camera_handoff --model obj_000004 --output /tmp/cup-approach
```

The PLY files and their PNG textures are read from
`segmind/resources/fame_episodes/alessandro_sliding_bueno/models`. The scale is
0.001 meters per source unit, matching the existing episode player. Model paths
and scale can be configured through `TexturedModel`; no assets are downloaded.

The textured camera samples the meshes' UV-mapped materials at SemDT scene ray
intersections. It needs no OpenGL display and does not silently replace textures
with flat colors. Lighting, shadows, and sensor noise are not simulated.

For textured runs the head view is oblique. The head pipeline follows RoboKudo's
demo: image preprocessing, point-cloud cropping, plane detection, and tabletop
clustering, followed by `ClusterPoseBBAnnotator`. Initialization selects the
cluster containing an image seed, which defaults to the head principal point.
This seed selects the desired instance; the prototype does not perform semantic
object recognition. The annotated pose and box dimensions initialize tracking
without a known table height or model dimensions. Box orientation is preserved
when projecting into the hand image.

`ClusterPoseBB` estimates a world-up oriented box around the visible cluster.
Its center and extents can differ from the complete object's geometry, and its
yaw is not a canonical model orientation. This is a geometric pose estimate,
not CAD-model registration.

The head supplies the initial object bounds, center, and a coarse dominant hue.
Only the estimated model bounds and uncertainty are projected into the hand image.
The hand acquires a colored silhouette inside that search region, initializes
corners from its own image, and follows them between successive hand frames.
Forward-backward optical flow and RANSAC reject inconsistent hand landmarks;
hand-image silhouette observations correct drift when available. The tracker
stores no head-image features and performs no head-to-hand descriptor matching.

The same projected search region, lateral controller, recorder, and evaluator
are reused. The green rectangle encloses the hand-image silhouette or its tracked
outline, and its cross marks the alignment point. The additional
`obj_000001.png` or `obj_000004.png` is the head-camera model preview.

This acquisition method assumes a distinctly colored foreground, which is useful
for these predominantly red assets against the gray table and blue distractor.
It does not establish general instance recognition among similarly colored
objects. Completion status is recorded even when an approach stops early.
These are controlled model experiments, not general six-dimensional object-pose
initialization.

## Recorded results

- `head.png`: fixed 512 × 512 head RGB image.
- `head_segmentation.png`: textured-run cluster rectangles and selected 3D box.
- `head_estimate.txt`: selected cluster, measured world pose, and box dimensions.
- `head_only/`: 240 × 240 hand frames during the biased approach without correction.
- `feedback/`: hand frames during the same approach with lateral feedback.
- `*_raw.png`: unchanged BGR sensor images saved as ordinary RGB PNGs.
- `*_overlay.png`: cyan search rectangle, green target rectangle and centroid,
  and white desired alignment cross.
- `measurements.csv`: camera positions, image errors, ground-truth lateral errors
  in meters, and target visibility for every saved step.
- `summary.txt`: completion status and final lateral errors of both approaches.

## Cube experiment

Two RoboKudo pipelines each have a CollectionReader and an independent CAS. Both
render the same persistent SemDT world through the existing RayTracer camera
interface. The scene contains an axis-aligned red cube and a blue distractor.
RGB uses flat surface colors; this does not test lighting or texture robustness.

The head looks straight down at the cube's top face. Its color mask and measured
depth locate the visible surface; the known edge length locates the cube center.
A deliberate position bias of 4 cm in world x and 2.5 cm in world z is then added.
The hand starts aligned with that biased estimate and looks horizontally at the
cube's front face. It moves from 55 cm to 22 cm from the estimated center in 1 cm
steps, without rotating.

The estimated cube bounds and configurable metric uncertainty are projected into
each hand image. A unique complete color component inside that region supplies
the image centroid. This is tracking by repeated detection with a geometric gate;
it does not use optical flow or cross-view feature matching. Missing, ambiguous,
or clipped observations stop movement. Feedback corrects the image centroid
toward the principal point, using head-derived front-face depth as a prior.

The hand CAS contains neither depth nor renderer object IDs, color-to-object
mappings, or scene ground truth. The evaluator reads the true cube center only
to measure error; its result does not drive the camera. The head-only comparison
uses the same initial estimate, images, and stopping distance.

This demonstrates projection, acquisition, and lateral alignment. It does not
simulate grasp execution or establish six-dimensional pose observability.
The face-center interpretation assumes this fixed, face-on cube geometry.
Camera resolutions, bias, uncertainty, step sizes, scene dimensions, color
selection, and feedback gain are dataclass parameters in `experiment.py`,
`simulation.py`, and `perception.py`.

## Tests

With sufficient free memory, cap the test process and run only the relevant file:

```bash
free -m
ulimit -v 12582912
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/pmania/venvs/cram-env/bin/python -m pytest test/robokudo_test/test_camera_handoff.py test/robokudo_test/test_textured_camera_handoff.py test/robokudo_test/test_tabletop_handoff.py --orm-build never --capture=fd
```
