# Copyright 2021 Tier IV, Inc.
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
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def _create_api_node(node_name, class_name, **kwargs):
    return ComposableNode(
        namespace="external",
        name=node_name,
        package="autoware_iv_external_api_adaptor",
        plugin="external_api::" + class_name,
        **kwargs
    )


def generate_launch_description():
    # RTCController is launched by tier4_autoware_api_launch because it is used by autoware.universe.
    components = [
        _create_api_node("calibration_status", "CalibrationStatus"),
        _create_api_node("cpu_usage", "CpuUsage"),
        _create_api_node("diagnostics", "Diagnostics"),
        _create_api_node("door", "Door"),
        _create_api_node("emergency", "Emergency"),
        _create_api_node("engage", "Engage"),
        _create_api_node("fail_safe_state", "FailSafeState"),
        _create_api_node("initial_pose", "InitialPose"),
        _create_api_node("localization_score", "LocalizationScore"),
        _create_api_node("map", "Map"),
        _create_api_node("operator", "Operator"),
        _create_api_node("metadata_packages", "MetadataPackages"),
        _create_api_node("route", "Route"),
        _create_api_node("service", "Service"),
        _create_api_node("start", "Start"),
        _create_api_node("vehicle_status", "VehicleStatus"),
        _create_api_node("velocity", "Velocity"),
        _create_api_node("version", "Version"),
    ]
    container = ComposableNodeContainer(
        namespace="external",
        name="autoware_iv_adaptor",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=components,
        output="screen",
    )
    return launch.LaunchDescription([container])
