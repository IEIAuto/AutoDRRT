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

#ifndef VEHICLE_STATUS_HPP_
#define VEHICLE_STATUS_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/gear_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/hazard_lights_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/steering_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/turn_indicators_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/velocity_report.hpp"
#include "tier4_external_api_msgs/msg/vehicle_command_stamped.hpp"
#include "tier4_external_api_msgs/msg/vehicle_status_stamped.hpp"

namespace external_api
{

class VehicleStatus : public rclcpp::Node
{
public:
  explicit VehicleStatus(const rclcpp::NodeOptions & options);

private:
  // ros interface for vehicle status
  rclcpp::Publisher<tier4_external_api_msgs::msg::VehicleStatusStamped>::SharedPtr pub_status_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr sub_velocity_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr sub_steering_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::TurnIndicatorsReport>::SharedPtr
    sub_turn_indicators_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::HazardLightsReport>::SharedPtr
    sub_hazard_lights_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::GearReport>::SharedPtr sub_gear_shift_;

  // ros interface for vehicle command
  rclcpp::Publisher<tier4_external_api_msgs::msg::VehicleCommandStamped>::SharedPtr pub_cmd_;
  rclcpp::Subscription<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr
    sub_cmd_;

  // ros callback
  void onTimer();

  // vehicle status
  autoware_auto_vehicle_msgs::msg::VelocityReport::ConstSharedPtr velocity_;
  autoware_auto_vehicle_msgs::msg::SteeringReport::ConstSharedPtr steering_;
  autoware_auto_vehicle_msgs::msg::TurnIndicatorsReport::ConstSharedPtr turn_indicators_;
  autoware_auto_vehicle_msgs::msg::HazardLightsReport::ConstSharedPtr hazard_lights_;
  autoware_auto_vehicle_msgs::msg::GearReport::ConstSharedPtr gear_shift_;
};

}  // namespace external_api

#endif  // VEHICLE_STATUS_HPP_
