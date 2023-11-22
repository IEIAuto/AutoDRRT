# Copyright 2020 Tier IV, Inc. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
import yaml


def get_livox_tag_filter_component():
    # livox tag filter
    livox_tag_filter_component = ComposableNode(
        package="livox_tag_filter",
        plugin="livox_tag_filter::LivoxTagFilterNode",
        name="livox_tag_filter",
        remappings=[
            ("input", "livox/lidar"),
            ("output", "livox/tag_filtered/lidar"),
        ],
        parameters=[
            {
                "ignore_tags": [1, 2, 20, 21, 22, 23, 24],
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )
    return livox_tag_filter_component


def get_crop_box_min_range_component(context, livox_frame_id):
    use_tag_filter = IfCondition(LaunchConfiguration("use_tag_filter")).evaluate(context)
    crop_box_min_range_component = ComposableNode(
        package="pointcloud_preprocessor",
        plugin="pointcloud_preprocessor::CropBoxFilterComponent",
        name="crop_box_filter_min_range",
        remappings=[
            ("input", "livox/tag_filtered/lidar" if use_tag_filter else "livox/lidar"),
            ("output", "min_range_cropped/pointcloud"),
        ],
        parameters=[
            {
                "input_frame": livox_frame_id,
                "output_frame": LaunchConfiguration("base_frame"),
                "min_x": 0.0,
                "max_x": 1.5,
                "min_y": -2.0,
                "max_y": 2.0,
                "min_z": -2.0,
                "max_z": 2.0,
                "negative": True,
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )
    return crop_box_min_range_component


def launch_setup(context, *args, **kwargs):
    lidar_config_path = LaunchConfiguration("lidar_config_file").perform(context)
    with open(lidar_config_path, "r") as f:
        params = yaml.safe_load(f)["/**"]["ros__parameters"]

    livox_driver_node = Node(
        executable="lidar_ros_driver_node",
        package="lidar_ros_driver",
        name="livox_horizon",
        remappings=[
            ("livox/cloud", "livox/lidar"),
            ("livox/imu_packet", "livox/imu"),
        ],
        parameters=[params],
        condition=IfCondition(LaunchConfiguration("launch_driver")),
        output="screen",
    )

    container = ComposableNodeContainer(
        name="livox_horizon",
        namespace="livox_horizon",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            get_crop_box_min_range_component(context, params["frame_id"]),
        ],
        output="screen",
    )

    livox_tag_filter_loader = LoadComposableNodes(
        composable_node_descriptions=[get_livox_tag_filter_component()],
        target_container=container,
        condition=IfCondition(LaunchConfiguration("use_tag_filter")),
    )

    return [livox_driver_node, container, livox_tag_filter_loader]


def generate_launch_description():

    launch_arguments = []

    def add_launch_arg(name: str, default_value=None):
        launch_arguments.append(DeclareLaunchArgument(name, default_value=default_value))

    add_launch_arg("launch_driver", "true")
    add_launch_arg("base_frame", "base_link")
    add_launch_arg("use_tag_filter", "true")
    add_launch_arg("lidar_config_file")

    return launch.LaunchDescription(launch_arguments + [OpaqueFunction(function=launch_setup)])
