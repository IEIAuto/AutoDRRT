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

#ifndef EMERGENCY_HPP_
#define EMERGENCY_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "tier4_external_api_msgs/msg/emergency.hpp"
#include "tier4_external_api_msgs/srv/set_emergency.hpp"

namespace external_api
{

class Emergency : public rclcpp::Node
{
public:
  explicit Emergency(const rclcpp::NodeOptions & options);

private:
  // ros interface
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<tier4_external_api_msgs::srv::SetEmergency>::SharedPtr srv_;
  tier4_api_utils::Client<tier4_external_api_msgs::srv::SetEmergency>::SharedPtr cli_;
  rclcpp::Publisher<tier4_external_api_msgs::msg::Emergency>::SharedPtr pub_emergency_;
  rclcpp::Subscription<tier4_external_api_msgs::msg::Emergency>::SharedPtr sub_emergency_;

  // ros callback
  void setEmergency(
    const tier4_external_api_msgs::srv::SetEmergency::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::SetEmergency::Response::SharedPtr response);
  void getEmergency(const tier4_external_api_msgs::msg::Emergency::SharedPtr message);
};

}  // namespace external_api

#endif  // EMERGENCY_HPP_
