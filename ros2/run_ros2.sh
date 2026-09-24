#!/usr/bin/env bash
# ROS 2 node without Docker (Ubuntu 22.04 + ROS 2 Humble installed on the host).
#   ./ros2/run_ros2.sh                    build (first time) and start the node, live UI on :8080
#   ./ros2/run_ros2.sh input_topic:=/lidar_points
# In another terminal: source /opt/ros/humble/setup.bash && ros2 bag play <bag>
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/humble/setup.bash
python3 -m pip install --user -q -r "$ROOT/requirements.txt"
WS="$ROOT/ros2/ws"
if [ ! -f "$WS/install/setup.bash" ]; then
  mkdir -p "$WS/src" && ln -sfn "$ROOT/ros2/gabarit_ros" "$WS/src/gabarit_ros"
  (cd "$WS" && colcon build --packages-select gabarit_ros)
fi
source "$WS/install/setup.bash"
export PYTHONPATH="$ROOT:$PYTHONPATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# point clouds are 8-24 MB per message: CycloneDDS delivers them reliably
if [ -z "$RMW_IMPLEMENTATION" ] && [ -d /opt/ros/humble/share/rmw_cyclonedds_cpp ]; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi
exec ros2 launch gabarit_ros gabarit.launch.py config:="$ROOT/config/gabarit.yaml" "$@"
