# Copyright 2022 Tier IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.conditions import LaunchConfigurationNotEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
import yaml


def launch_setup(context, *args, **kwargs):
    # https://github.com/ros2/launch_ros/issues/156
    def load_composable_node_param(param_path):
        with open(LaunchConfiguration(param_path).perform(context), "r") as f:
            return yaml.safe_load(f)["/**"]["ros__parameters"]

    composable_nodes = [
        ComposableNode(
            package="gscam",
            plugin="gscam::GSCam",
            name=["gscam", LaunchConfiguration("camera_id")],
            namespace=LaunchConfiguration("gscam_namespace"),
            remappings=[
                (
                    "camera/image_raw",
                    [
                        "camera",
                        LaunchConfiguration("camera_id"),
                        "/",
                        LaunchConfiguration("image_topic"),
                    ],
                ),
                (
                    "camera/image_raw/compressed",
                    [
                        "camera",
                        LaunchConfiguration("camera_id"),
                        "/",
                        LaunchConfiguration("image_topic"),
                        "/compressed",
                    ],
                ),
                (
                    "camera/image_raw/compressedDepth",
                    [
                        "camera",
                        LaunchConfiguration("camera_id"),
                        "/",
                        LaunchConfiguration("image_topic"),
                        "/compressedDepth",
                    ],
                ),
                (
                    "camera/image_raw/theora",
                    [
                        "camera",
                        LaunchConfiguration("camera_id"),
                        "/",
                        LaunchConfiguration("image_topic"),
                        "/theora",
                    ],
                ),
                (
                    "camera/camera_info",
                    ["camera", LaunchConfiguration("camera_id"), "/camera_info"],
                ),
            ],
            parameters=[
                load_composable_node_param("gscam_param_path"),
                {
                    "camera_info_url": LaunchConfiguration("camera_info_url"),
                },
            ],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        ),
        ComposableNode(
            package="jetson_camera_trigger",
            plugin="jetson_camera_trigger::JetsonCameraTrigger",
            name=["camera", LaunchConfiguration("camera_id"), "_trigger"],
            namespace=LaunchConfiguration("gscam_namespace"),
            remappings=[
                ("trigger_time", ["camera", LaunchConfiguration("camera_id"), "/trigger_time"]),
            ],
            parameters=[
                load_composable_node_param("camera_trigger_param_path"),
            ],
            extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
        ),
    ]

    # If an existing container is not provided, start a container and load nodes into it
    gscam_container = ComposableNodeContainer(
        condition=LaunchConfigurationEquals("container", ""),
        name="gscam_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=composable_nodes,
        output="screen",
    )

    # If an existing container name is provided, load composable nodes into it
    # This will block until a container with the provided name is available and nodes are loaded
    load_composable_nodes = LoadComposableNodes(
        condition=LaunchConfigurationNotEquals("container", ""),
        composable_node_descriptions=composable_nodes,
        target_container=LaunchConfiguration("container"),
    )

    return [gscam_container, load_composable_nodes]


def generate_launch_description():

    launch_arguments = []

    def add_launch_arg(name: str, default_value=None):
        launch_arguments.append(DeclareLaunchArgument(name, default_value=default_value))

    add_launch_arg("container", "")
    add_launch_arg("image_topic")
    add_launch_arg("camera_id")
    add_launch_arg("gscam_namespace", "/sensing/camera")

    add_launch_arg("gscam_param_path")

    add_launch_arg("camera_info_url")

    add_launch_arg("camera_trigger_param_path")

    add_launch_arg("use_intra_process", "True")

    return LaunchDescription(
        [
            *launch_arguments,
            OpaqueFunction(function=launch_setup),
        ]
    )
