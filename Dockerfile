# Gabarit — obstacle detection for a driverless metro train (ROS 2 Humble, Ubuntu 22.04)
#
#   docker build -t gabarit .
#   docker run --rm -it -p 8080:8080 -v /path/to/for_hackathon:/data gabarit demo doubleT_obstacle
#
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONPATH=/opt/gabarit \
    GABARIT_DATA=/data \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# CycloneDDS: point clouds of the dataset are 8-24 MB per message; the default
# FastDDS profile drops them on large fragmented transfers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions \
        ros-humble-rosbag2-storage-default-plugins \
        ros-humble-visualization-msgs ros-humble-sensor-msgs \
        ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# NumPy >= 1.25 is required for fast ufunc.at; the rest is for the web UI.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/gabarit
COPY gabarit ./gabarit
COPY web ./web
COPY scripts ./scripts
COPY config ./config
COPY tests ./tests
COPY docker ./docker
COPY ros2 ./ros_ws/src

RUN . /opt/ros/humble/setup.sh \
    && cd ros_ws && colcon build --packages-select gabarit_ros \
    && chmod +x /opt/gabarit/docker/entrypoint.sh /opt/gabarit/docker/gabarit \
    && ln -s /opt/gabarit/docker/gabarit /usr/local/bin/gabarit

EXPOSE 8080 8000
ENTRYPOINT ["/opt/gabarit/docker/entrypoint.sh"]
CMD ["gabarit", "detect"]
