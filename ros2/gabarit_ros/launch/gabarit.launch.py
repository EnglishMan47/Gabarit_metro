"""Launch the detector (and optionally RViz2).

    ros2 launch gabarit_ros gabarit.launch.py input_topic:=auto rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("gabarit_ros")
    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="auto"),
        DeclareLaunchArgument("input_qos", default_value="auto"),
        DeclareLaunchArgument("config", default_value="/opt/gabarit/config/gabarit.yaml"),
        DeclareLaunchArgument("web_port", default_value="8080"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="gabarit_ros",
            executable="gabarit_node",
            name="gabarit",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("input_topic"),
                "input_qos": LaunchConfiguration("input_qos"),
                "config": LaunchConfiguration("config"),
                "web_port": ParameterValue(LaunchConfiguration("web_port"), value_type=int),
                "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "gabarit.rviz")],
            parameters=[{"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
