# Fast-FoundationStereo RGB-D input

RoboKudo can consume the rectified left image and metric depth produced by the
separate Fast-FoundationStereo (FFS) ROS 2 pipeline. FFS runs in its own Python
environment; its model dependencies are not needed in RK's environment.

## Run FFS with the live Olive camera

Start `olive_stereo_rectified.py --publish` as usual. In the FFS environment,
launch inference with the Olive parameter file from this checkout:

```bash
workon fast-foundation-stereo
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
ros2 launch fast_foundation_stereo_ros stereo_pipeline.launch.py \
  repository_path:=$HOME/Documents/Experiment-Fast-FoundationStereo/Fast-FoundationStereo \
  model_path:=$HOME/Documents/Experiment-Fast-FoundationStereo/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  params_file:=$HOME/lib/cognitive_robot_abstract_machine/robokudo/config/foundation_stereo_olive.yaml
```

Adjust the checkout and model paths if your installation differs. The launch reads
the parameter file directly, so it does not need a rebuild. It subscribes to the
left and right rectified *color* images and their calibration, using reliable QoS.
RK needs `/ffs/depth`; the launch also publishes an optional cloud on `/ffs_points2`.
Neither `stereo_image_proc` nor `disparity_to_depth.py` is required for FFS.

In another terminal, activate the CRAM environment and source ROS and the overlay:

```bash
workon cram-env
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
ros2 run robokudo_ros main _ae foundation_stereo
```

Use the same ROS domain for the Olive output publisher, FFS and RK. The analysis
engine uses `olive_left_optical_frame`, reliable subscriptions and no TF lookup.
Its image preprocessor generates the cloud with a depth cutoff matching
`max_depth`; the existing crop and plane stages follow it. It does not establish
a world pose; add a real transform before world-coordinate processing.

## Input contract

| Input | Default topic | Contract |
|---|---|---|
| Left color | `/left/image_rect_color` | Raw `sensor_msgs/Image`, rectified left pixel grid |
| Depth | `/ffs/depth` | Raw `sensor_msgs/Image`, `32FC1`, meters, same dimensions and acquisition stamp as color |
| Calibration | `/left/camera_info` | Per-frame `sensor_msgs/CameraInfo`, full-size rectified left projection in `p` |

All three messages must use the configured `tf_from` optical frame. Color is
converted to BGR. Depth is copied into native contiguous float32 millimeters for
RK; NaN, infinity, nonpositive values and depths beyond `max_depth` become zero.
The ROS depth topic remains meter-valued. The CAS resolution ratio is identity.

Intrinsics come from the left rectified projection `p`, not the raw camera matrix
`k`. The CAS receives a calibration copy with matching pinhole `k`, identity `r`
and zero distortion. Input calibration is not modified. Nonstandard left
projections, cropping metadata and binning beyond identity are rejected.

## Physical VR.cam input

The VR.cam driver and calibrated fisheye rectification must run upstream of FFS.
RK does not capture the stereo pair or rectify raw fisheye images. The right
image and calibrated baseline are handled by FFS, not this reader.

Create a `foundation_stereo` collection-reader descriptor through
`CollectionReaderDescriptorFactory.create_descriptor`, overriding the fields of
`FoundationStereoCameraConfig` for the actual camera. The provided analysis engine
also accepts a `camera_config` when instantiated from another Python descriptor.
The defaults match the Olive publisher. Override `tf_from` and the topics when
using another driver, namespace or optical frame.

For a moving camera, enable `lookup_viewpoint` and set `tf_to` to the world frame.
TF is requested at the original image acquisition time. Publish the complete TF
chain, including any rotation between physical and rectified optical frames, and
retain enough TF history for inference latency. Frames with unavailable TF are
discarded and the collection reader keeps waiting.

For a fixed camera without TF, disable `lookup_viewpoint` and enable
`static_camera_transform_enabled`. Set `static_camera_frame` equal to `tf_from`,
and provide `static_world_frame` and `static_world_T_camera`. Static transforms
and TF lookup cannot both be enabled. For camera-frame-only processing, leave
both disabled.

## Synchronization, QoS and memory

RGB and depth are matched by identical acquisition timestamps using ROS
`TimeSynchronizer`. A separate synchronizer associates calibration within
`calibration_tolerance` seconds (default 0.03); setting it to zero requires exact
calibration timestamps too. Calibration tolerance never relaxes RGB/depth matching.
Camera information must be published per frame.

`synchronization_queue_size` bounds each message-filter input queue (default 30).
Choose it to cover camera input rate multiplied by expected depth latency, with
headroom for delivery jitter. At 30 Hz, 30 messages retain roughly one second of
input history. Account for resolution: buffering large color images can consume
substantial memory. Both the exact-image and calibration synchronization stages
have bounded queues. DDS histories are separately bounded by `history_depth`
(default 2); only one complete frame waits for RK consumption.

`max_frame_age` (default 1 second) is checked at receipt and consumption. It must
cover inference and transport latency; an expired frame is dropped. Out-of-order
or repeated acquisitions cannot replace newer frames. Input stamps and the RK
clock must share a time base, including `use_sim_time` during bag playback.

The default configuration requests reliable delivery on all streams, matching
the Olive publisher and FFS depth output. Configure `color_reliability`,
`calibration_reliability` and `depth_reliability` when publishers differ.

## Static upstream demo

The separate FFS project's demo launch remains unchanged. To consume that demo
from an RK descriptor, pass a `FoundationStereoCameraConfig` to the analysis engine
with `topic_color="/stereo/left/image_rect"`,
`topic_camera_info="/stereo/left/camera_info"` and
`tf_from="demo_left_optical"` and `topic_depth="/foundation_stereo/depth"`.
The default `foundation_stereo` analysis engine now
selects live Olive input.

The interface exposes `statistics` for matched frames, stale frames, replaced
pending frames, obsolete acquisitions and unavailable transforms. Messages evicted
inside ROS synchronization queues are not included in those counters. If RK
waits without matching frames, check topic names, QoS, exact image stamps,
per-frame calibration and queue retention first.

The `/ffs_points2` output is not required by RK. Its own preprocessor
creates the cloud from images, preserving the image/calibration data required by
other annotators. Raw fisheye geometry and physical depth accuracy still require
validation with real VR.cam data.
