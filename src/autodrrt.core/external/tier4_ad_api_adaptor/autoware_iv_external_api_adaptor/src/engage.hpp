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

#ifndef ENGAGE_HPP_
#define ENGAGE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "autoware_auto_system_msgs/msg/autoware_state.hpp"
#include "autoware_auto_vehicle_msgs/msg/engage.hpp"
#include "tier4_external_api_msgs/msg/engage_status.hpp"
#include "tier4_external_api_msgs/srv/engage.hpp"
#include "tier4_external_api_msgs/srv/set_operator.hpp"

namespace external_api
{

class Engage : public rclcpp::Node
{
public:
  explicit Engage(const rclcpp::NodeOptions & options);

private:
  using ExternalEngage = tier4_external_api_msgs::srv::Engage;
  using ExternalEngageStatus = tier4_external_api_msgs::msg::EngageStatus;
  using SetOperator = tier4_external_api_msgs::srv::SetOperator;
  using VehicleEngageStatus = autoware_auto_vehicle_msgs::msg::Engage;
  using AutowareState = autoware_auto_system_msgs::msg::AutowareState;

  // ros interface
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<ExternalEngage>::SharedPtr srv_engage_;
  tier4_api_utils::Client<ExternalEngage>::SharedPtr cli_engage_;
  tier4_api_utils::Client<SetOperator>::SharedPtr cli_set_operator_;
  rclcpp::Publisher<ExternalEngageStatus>::SharedPtr pub_engage_status_;
  rclcpp::Subscription<VehicleEngageStatus>::SharedPtr sub_engage_status_;
  rclcpp::Subscription<AutowareState>::SharedPtr sub_autoware_state_;

  // class state
  bool waiting_for_engage_;
  bool driving_;
  bool auto_operator_change_;

  // ros callback
  void setEngage(
    const tier4_external_api_msgs::srv::Engage::Request::SharedPtr request,
    const tier4_external_api_msgs::srv::Engage::Response::SharedPtr response);
  void onEngageStatus(const autoware_auto_vehicle_msgs::msg::Engage::SharedPtr message);
  void onAutowareState(const autoware_auto_system_msgs::msg::AutowareState::SharedPtr message);
};

}  // namespace external_api

#endif  // ENGAGE_HPP_
