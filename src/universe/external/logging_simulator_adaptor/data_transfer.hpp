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

#ifndef DATA_TRANSFER_HPP_
#define DATA_TRANSFER_HPP_

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
#include "autoware_auto_vehicle_msgs/msg/control_mode_report.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include "std_msgs/msg/string.hpp"
// #include "ublox_msgs/ublox_msgs.h"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
namespace logging_simulator_adator
{

class DataTransfer : public rclcpp::Node
{
public:
  explicit DataTransfer(const rclcpp::NodeOptions & options);

private:

  rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr pub_velocity_;
  rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr pub_steering_;
  rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::ControlModeReport>::SharedPtr pub_control_mode_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_raw_;
  // rclcpp::Publisher<ublox_msgs::msg::NavPVT>::SharedPtr pub_navpvt_;
  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_nav_sat_fix_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr pub_fix_velocity_;
  rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::GearReport>::SharedPtr pub_gear_shift_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_test_;


  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr sub_velocity_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr sub_steering_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::ControlModeReport>::SharedPtr sub_control_mode_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_raw_;
  // rclcpp::Subscription<ublox_msgs::msg::NavPVT>::SharedPtr sub_navpvt_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_nav_sat_fix_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_fix_velocity_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::GearReport>::SharedPtr sub_gear_shift_;

  

};

}  // namespace external_api

#endif  // VEHICLE_STATUS_HPP_
