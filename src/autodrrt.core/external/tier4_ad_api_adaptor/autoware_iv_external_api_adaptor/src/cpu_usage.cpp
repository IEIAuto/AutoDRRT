// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cpu_usage.hpp"

#include <memory>

namespace external_api
{

CpuUsage::CpuUsage(const rclcpp::NodeOptions & options) : Node("cpu_usage", options)
{
  pub_cpu_usage_ = create_publisher<tier4_external_api_msgs::msg::CpuUsage>(
    "/api/external/get/cpu_usage", rclcpp::QoS(1));
  sub_cpu_usage_ = create_subscription<tier4_external_api_msgs::msg::CpuUsage>(
    "/system/system_monitor/cpu_monitor/cpu_usage", rclcpp::QoS(1),
    [this](const tier4_external_api_msgs::msg::CpuUsage::SharedPtr msg) {
      pub_cpu_usage_->publish(*msg);
    });
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::CpuUsage)
