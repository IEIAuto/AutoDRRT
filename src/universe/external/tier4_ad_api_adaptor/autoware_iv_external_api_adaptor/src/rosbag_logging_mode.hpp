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

#ifndef ROSBAG_LOGGING_MODE_HPP_
#define ROSBAG_LOGGING_MODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/msg/rosbag_logging_mode.hpp"
#include "tier4_external_api_msgs/srv/set_rosbag_logging_mode.hpp"

namespace external_api
{

class RosbagLoggingMode : public rclcpp::Node
{
public:
  explicit RosbagLoggingMode(const rclcpp::NodeOptions & options);

private:
  using SetRosbagLoggingMode = tier4_external_api_msgs::srv::SetRosbagLoggingMode;
  using GetRosbagLoggingMode = tier4_external_api_msgs::msg::RosbagLoggingMode;

  // ros interface
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<SetRosbagLoggingMode>::SharedPtr srv_set_rosbag_logging_mode_;
  tier4_api_utils::Client<SetRosbagLoggingMode>::SharedPtr cli_set_rosbag_logging_mode_;
  rclcpp::Publisher<GetRosbagLoggingMode>::SharedPtr pub_get_rosbag_logging_mode_;
  rclcpp::Subscription<GetRosbagLoggingMode>::SharedPtr sub_get_rosbag_logging_mode_;

  // ros callback
  void setRosbagLoggingMode(
    const tier4_external_api_msgs::srv::SetRosbagLoggingMode::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::SetRosbagLoggingMode::Response::SharedPtr response);
  void onRosbagLoggingMode(
    const tier4_external_api_msgs::msg::RosbagLoggingMode::ConstSharedPtr message);
};

}  // namespace external_api

#endif  // ROSBAG_LOGGING_MODE_HPP_
