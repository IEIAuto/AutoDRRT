// Copyright 2022 TIER IV, Inc.
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

#include "rosbag_logging_mode.hpp"

namespace external_api
{

RosbagLoggingMode::RosbagLoggingMode(const rclcpp::NodeOptions & options)
: Node("external_api_rosbag_logging_mode", options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_set_rosbag_logging_mode_ =
    proxy.create_service<tier4_external_api_msgs::srv::SetRosbagLoggingMode>(
      "/api/external/set/rosbag_logging_mode",
      std::bind(&RosbagLoggingMode::setRosbagLoggingMode, this, _1, _2),
      rmw_qos_profile_services_default, group_);
  cli_set_rosbag_logging_mode_ =
    proxy.create_client<tier4_external_api_msgs::srv::SetRosbagLoggingMode>(
      "/api/autoware/set/rosbag_logging_mode", rmw_qos_profile_services_default);
  pub_get_rosbag_logging_mode_ = create_publisher<tier4_external_api_msgs::msg::RosbagLoggingMode>(
    "/api/external/get/rosbag_logging_mode", rclcpp::QoS(1));
  sub_get_rosbag_logging_mode_ =
    create_subscription<tier4_external_api_msgs::msg::RosbagLoggingMode>(
      "/api/autoware/get/rosbag_logging_mode", rclcpp::QoS(1),
      std::bind(&RosbagLoggingMode::onRosbagLoggingMode, this, _1));
}

void RosbagLoggingMode::setRosbagLoggingMode(
  const tier4_external_api_msgs::srv::SetRosbagLoggingMode::Request::SharedPtr request,
  const tier4_external_api_msgs::srv::SetRosbagLoggingMode::Response::SharedPtr response)
{
  // systemd's default timeouts for starting and stopping are both 90 seconds.
  // See below for more details.
  // https://www.freedesktop.org/software/systemd/man/systemd-system.conf.html
  // So timeout for restarting is 180 seconds by default.
  // The value of timeout below is 10 seconds added with a margin.
  const auto [status, resp] =
    cli_set_rosbag_logging_mode_->call(request, std::chrono::seconds(190));
  if (!tier4_api_utils::is_success(status)) {
    response->status = status;
    return;
  }
  response->status = resp->status;
}

void RosbagLoggingMode::onRosbagLoggingMode(
  const tier4_external_api_msgs::msg::RosbagLoggingMode::ConstSharedPtr message)
{
  pub_get_rosbag_logging_mode_->publish(*message);
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::RosbagLoggingMode)
