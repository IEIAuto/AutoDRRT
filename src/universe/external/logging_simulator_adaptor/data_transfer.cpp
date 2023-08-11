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

#include "data_transfer.hpp"

#include "tier4_auto_msgs_converter/tier4_auto_msgs_converter.hpp"

#include "tier4_external_api_msgs/msg/gear_shift.hpp"
#include "tier4_external_api_msgs/msg/turn_signal.hpp"
#include "tier4_vehicle_msgs/msg/shift.hpp"
#include "tier4_vehicle_msgs/msg/turn_signal.hpp"
#include "std_msgs/msg/string.hpp"
#include <utility>
  // namespace

namespace logging_simulator_adator
{

DataTransfer::DataTransfer(const rclcpp::NodeOptions & options)
: Node("data_transfer", options)
{
  using namespace std::literals::chrono_literals;
  std::cout << "JUST FOR TEST FAULT TOLERANCE" << std::endl;
  this->sub_ownership_strenth = 0;
  // pub_status_ = create_publisher<tier4_external_api_msgs::msg::VehicleStatusStamped>(
  //   "/api/external/get/vehicle/status", rclcpp::QoS(1));
  // pub_cmd_ = create_publisher<tier4_external_api_msgs::msg::VehicleCommandStamped>(
  //   "/api/external/get/command/selected/vehicle", rclcpp::QoS(1));
  rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable();
        qos.deadline(rclcpp::Duration(1,0));
  pub_test_ = this->create_publisher<std_msgs::msg::String>("chatter", qos);


  
 
  pub_velocity_ = create_publisher<autoware_auto_vehicle_msgs::msg::VelocityReport>(
    "/vehicle/status/velocity_status", rclcpp::QoS(1));
  pub_steering_ = create_publisher<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "/vehicle/status/steering_status", rclcpp::QoS(1));

  pub_control_mode_ = create_publisher<autoware_auto_vehicle_msgs::msg::ControlModeReport>(
  "/vehicle/status/control_mode", rclcpp::QoS(1));

  pub_imu_raw_ = create_publisher<sensor_msgs::msg::Imu>(
  "/sensing/imu/tamagawa/imu_raw", rclcpp::QoS(1));


  // pub_navpvt_ = create_publisher<ublox_msgs::msg::NavPVT>(
  // "/sensing/gnss/ublox/navpvt", rclcpp::QoS(1));

  pub_nav_sat_fix_ = create_publisher<sensor_msgs::msg::NavSatFix>(
  "/sensing/gnss/ublox/nav_sat_fix", rclcpp::QoS(1));

  pub_fix_velocity_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
  "/sensing/gnss/ublox/fix_velocity", rclcpp::QoS(1));

  pub_gear_shift_ = create_publisher<autoware_auto_vehicle_msgs::msg::GearReport>(
    "/vehicle/status/gear_status", rclcpp::QoS(1));


  sub_velocity_ = create_subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>(
    "/vehicle/status/velocity_status/transfer", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::VelocityReport::ConstSharedPtr msg) {
      pub_velocity_ -> publish(*msg);
    });
  sub_steering_ = create_subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "/vehicle/status/steering_status/transfer", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::SteeringReport::ConstSharedPtr msg) {
       pub_steering_ -> publish(*msg);
    });

  sub_control_mode_ = create_subscription<autoware_auto_vehicle_msgs::msg::ControlModeReport>(
  "/vehicle/status/control_mode/transfer", rclcpp::QoS(1),
  [this](const autoware_auto_vehicle_msgs::msg::ControlModeReport::ConstSharedPtr msg) {
     pub_control_mode_ -> publish(*msg);
  });

  sub_imu_raw_ = create_subscription<sensor_msgs::msg::Imu>(
  "/sensing/imu/tamagawa/imu_raw/transfer", rclcpp::QoS(1),
  [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
     pub_imu_raw_ -> publish(*msg);
  });


  // sub_navpvt_ = create_subscription<ublox_msgs::msg::NavPVT>(
  // "/sensing/gnss/ublox/navpvt/transfer", rclcpp::QoS(1),
  // [this](const ublox_msgs::msg::NavPVT::ConstSharedPtr msg) {
  //    pub_navpvt_ -> publish(msg);
  // });

  sub_nav_sat_fix_ = create_subscription<sensor_msgs::msg::NavSatFix>(
  "/sensing/gnss/ublox/nav_sat_fix/transfer", rclcpp::QoS(1),
  [this](const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg) {
     pub_nav_sat_fix_ -> publish(*msg);
  });

   sub_fix_velocity_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
  "/sensing/gnss/ublox/fix_velocity/transfer", rclcpp::QoS(1),
  [this](const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr msg) {
     pub_fix_velocity_ -> publish(*msg);
  });

  sub_gear_shift_ = create_subscription<autoware_auto_vehicle_msgs::msg::GearReport>(
    "/vehicle/status/gear_status/transfer", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::GearReport::ConstSharedPtr msg) {
       pub_gear_shift_ -> publish(*msg);
    });

}


}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(logging_simulator_adator::DataTransfer)
