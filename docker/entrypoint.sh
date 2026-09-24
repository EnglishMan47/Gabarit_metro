#!/usr/bin/env bash
# Entry point of the Gabarit container. See `gabarit help`.
set -e
source /opt/ros/humble/setup.bash
source /opt/gabarit/ros_ws/install/setup.bash
exec "$@"
