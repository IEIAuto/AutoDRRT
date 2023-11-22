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

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def get_vehicle_info(context):
    gp = context.launch_configurations.get("ros_params", {})
    p = {}
    p["vehicle_length"] = gp["front_overhang"] + gp["wheel_base"] + gp["rear_overhang"]
    p["vehicle_width"] = gp["wheel_tread"] + gp["left_overhang"] + gp["right_overhang"]
    p["min_longitudinal_offset"] = -gp["rear_overhang"]
    p["max_longitudinal_offset"] = gp["front_overhang"] + gp["wheel_base"]
    p["min_lateral_offset"] = -(gp["wheel_tread"] / 2.0 + gp["right_overhang"])
    p["max_lateral_offset"] = gp["wheel_tread"] / 2.0 + gp["left_overhang"]
    p["min_height_offset"] = 0.0
    p["max_height_offset"] = gp["vehicle_height"]
    return p


def get_livox_tag_filter_component(ns):
    # livox tag filter
    livox_tag_filter_component = ComposableNode(
        package="livox_tag_filter",
        plugin="livox_tag_filter::LivoxTagFilterNode",
        name=ns + "_livox_tag_filter",
        remappings=[
            ("input", ns + "/livox/lidar"),
            ("output", ns + "/livox/tag_filtered/lidar"),
        ],
        parameters=[
            {
                "ignore_tags": [1, 2, 20, 21, 22, 23, 24],
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )
    return livox_tag_filter_component


def get_crop_box_min_range_component(ns, context):
    use_tag_filter = IfCondition(LaunchConfiguration("use_tag_filter")).evaluate(context)
    crop_box_min_range_component = ComposableNode(
        package="pointcloud_preprocessor",
        plugin="pointcloud_preprocessor::CropBoxFilterComponent",
        name=ns + "_crop_box_filter_min_range",
        remappings=[
            ("input", ns + "/livox/tag_filtered/lidar" if use_tag_filter else ns + "/livox/lidar"),
            ("output", ns + "/min_range_cropped/pointcloud"),
        ],
        parameters=[
            {
                "input_frame": "livox_" + ns,
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
    livox_config_path = os.path.join(
        get_package_share_directory("individual_params"),
        "config",
        EnvironmentVariable(name="VEHICLE_ID", default_value="default").perform(context),
        "aip_x1",
        "livox_lidar_config.json",
    )

    # livox driver
    livox_driver_component = ComposableNode(
        package="livox_ros2_driver",
        plugin="livox_ros::LivoxDriver",
        name="livox_driver",
        parameters=[
            {
                "xfe_format": LaunchConfiguration("xfe_format"),
                "multi_topic": LaunchConfiguration("multi_topic"),
                "data_src": LaunchConfiguration("data_src"),
                "publish_freq": LaunchConfiguration("publish_freq"),
                "output_data_type": LaunchConfiguration("output_type"),
                "lvx_file_path": LaunchConfiguration("lvx_file_path"),
                "user_config_path": livox_config_path,
                "frame_id": LaunchConfiguration("sensor_frame"),
            },
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    front_left_crop_box_min_range_component = get_crop_box_min_range_component(
        "front_left", context
    )
    front_center_crop_box_min_range_component = get_crop_box_min_range_component(
        "front_center", context
    )
    front_right_crop_box_min_range_component = get_crop_box_min_range_component(
        "front_right", context
    )
    container = ComposableNodeContainer(
        name="pointcloud_preprocessor_container",
        namespace="livox_pointcloud_preprocessor",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            front_left_crop_box_min_range_component,
            front_center_crop_box_min_range_component,
            front_right_crop_box_min_range_component,
        ],
        output="screen",
    )

    livox_driver_loader = LoadComposableNodes(
        composable_node_descriptions=[livox_driver_component],
        target_container=container,
        condition=IfCondition(LaunchConfiguration("launch_driver")),
    )

    front_left_livox_tag_filter_component = get_livox_tag_filter_component("front_left")
    front_center_livox_tag_filter_component = get_livox_tag_filter_component("front_center")
    front_right_livox_tag_filter_component = get_livox_tag_filter_component("front_right")
    livox_tag_filter_loader = LoadComposableNodes(
        composable_node_descriptions=[
            front_left_livox_tag_filter_component,
            front_center_livox_tag_filter_component,
            front_right_livox_tag_filter_component,
        ],
        target_container=container,
        condition=IfCondition(LaunchConfiguration("use_tag_filter")),
    )

    return [container, livox_driver_loader, livox_tag_filter_loader]


def generate_launch_description():

    launch_arguments = []

    def add_launch_arg(name: str, default_value=None):
        launch_arguments.append(DeclareLaunchArgument(name, default_value=default_value))

    add_launch_arg("xfe_format", "0")
    add_launch_arg("multi_topic", "1")
    add_launch_arg("data_src", "0")
    add_launch_arg("publish_freq", "10.0")
    add_launch_arg("output_type", "0")
    add_launch_arg("lvx_file_path", "livox_test.lvx")
    add_launch_arg("launch_driver")
    add_launch_arg("base_frame", "base_link")
    add_launch_arg("sensor_frame", "livox_frame")
    add_launch_arg("use_tag_filter", "true")
    add_launch_arg("vehicle_mirror_param_file")

    return launch.LaunchDescription(launch_arguments + [OpaqueFunction(function=launch_setup)])
