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

#include "fail_safe_state.hpp"

#include "converter/fail_safe_state.hpp"

namespace external_api
{

FailSafeState::FailSafeState(const rclcpp::NodeOptions & options)
: Node("external_api_fail_safe_state", options)
{
  pub_state_ = create_publisher<tier4_external_api_msgs::msg::FailSafeStateStamped>(
    "/api/external/get/fail_safe/state", rclcpp::QoS(1));
  sub_state_ = create_subscription<autoware_adapi_v1_msgs::msg::MrmState>(
    "/system/fail_safe/mrm_state", rclcpp::QoS(1),
    [this](const autoware_adapi_v1_msgs::msg::MrmState::ConstSharedPtr msg) {
      pub_state_->publish(converter::to_external(*msg));
    });
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::FailSafeState)
