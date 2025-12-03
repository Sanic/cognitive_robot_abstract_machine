#!/usr/bin/env bash
set -e

source /opt/ros/overlay_ws/install/setup.bash

source /opt/ros/cram-env/bin/activate

exec "$@"