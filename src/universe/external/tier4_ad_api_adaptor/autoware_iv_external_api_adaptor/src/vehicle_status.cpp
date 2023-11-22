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

#include "vehicle_status.hpp"

#include "tier4_auto_msgs_converter/tier4_auto_msgs_converter.hpp"

#include "tier4_external_api_msgs/msg/gear_shift.hpp"
#include "tier4_external_api_msgs/msg/turn_signal.hpp"
#include "tier4_vehicle_msgs/msg/shift.hpp"
#include "tier4_vehicle_msgs/msg/turn_signal.hpp"

#include <utility>

namespace
{

tier4_external_api_msgs::msg::TurnSignal convert(const tier4_vehicle_msgs::msg::TurnSignal & msg)
{
  using External = tier4_external_api_msgs::msg::TurnSignal;
  using Internal = tier4_vehicle_msgs::msg::TurnSignal;

  switch (msg.data) {
    case Internal::NONE:
      return tier4_external_api_msgs::build<External>().data(External::NONE);
    case Internal::LEFT:
      return tier4_external_api_msgs::build<External>().data(External::LEFT);
    case Internal::RIGHT:
      return tier4_external_api_msgs::build<External>().data(External::RIGHT);
    case Internal::HAZARD:
      return tier4_external_api_msgs::build<External>().data(External::HAZARD);
  }
  throw std::out_of_range("turn_signal=" + std::to_string(msg.data));
}

tier4_external_api_msgs::msg::GearShift convert(const tier4_vehicle_msgs::msg::Shift & msg)
{
  using External = tier4_external_api_msgs::msg::GearShift;
  using Internal = tier4_vehicle_msgs::msg::Shift;

  switch (msg.data) {
    case Internal::NONE:
      return tier4_external_api_msgs::build<External>().data(External::NONE);
    case Internal::PARKING:
      return tier4_external_api_msgs::build<External>().data(External::PARKING);
    case Internal::REVERSE:
      return tier4_external_api_msgs::build<External>().data(External::REVERSE);
    case Internal::NEUTRAL:
      return tier4_external_api_msgs::build<External>().data(External::NEUTRAL);
    case Internal::DRIVE:
      return tier4_external_api_msgs::build<External>().data(External::DRIVE);
    case Internal::LOW:
      return tier4_external_api_msgs::build<External>().data(External::LOW);
  }
  throw std::out_of_range("gear_shift=" + std::to_string(msg.data));
}

}  // namespace

namespace external_api
{

VehicleStatus::VehicleStatus(const rclcpp::NodeOptions & options)
: Node("external_api_vehicle_status", options)
{
  using namespace std::literals::chrono_literals;

  pub_status_ = create_publisher<tier4_external_api_msgs::msg::VehicleStatusStamped>(
    "/api/external/get/vehicle/status", rclcpp::QoS(1));
  timer_ = rclcpp::create_timer(this, get_clock(), 200ms, std::bind(&VehicleStatus::onTimer, this));

  sub_velocity_ = create_subscription<autoware_auto_vehicle_msgs::msg::VelocityReport>(
    "/vehicle/status/velocity_status", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::VelocityReport::ConstSharedPtr msg) {
      velocity_ = msg;
    });
  sub_steering_ = create_subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "/vehicle/status/steering_status", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::SteeringReport::ConstSharedPtr msg) {
      steering_ = msg;
    });
  sub_turn_indicators_ = create_subscription<autoware_auto_vehicle_msgs::msg::TurnIndicatorsReport>(
    "/vehicle/status/turn_indicators_status", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::TurnIndicatorsReport::ConstSharedPtr msg) {
      turn_indicators_ = msg;
    });
  sub_hazard_lights_ = create_subscription<autoware_auto_vehicle_msgs::msg::HazardLightsReport>(
    "/vehicle/status/hazard_lights_status", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::HazardLightsReport::ConstSharedPtr msg) {
      hazard_lights_ = msg;
    });
  sub_gear_shift_ = create_subscription<autoware_auto_vehicle_msgs::msg::GearReport>(
    "/vehicle/status/gear_status", rclcpp::QoS(1),
    [this](const autoware_auto_vehicle_msgs::msg::GearReport::ConstSharedPtr msg) {
      gear_shift_ = msg;
    });

  pub_cmd_ = create_publisher<tier4_external_api_msgs::msg::VehicleCommandStamped>(
    "/api/external/get/command/selected/vehicle", rclcpp::QoS(1));
  sub_cmd_ = create_subscription<autoware_auto_control_msgs::msg::AckermannControlCommand>(
    "/control/command/control_cmd", rclcpp::QoS(1),
    [this](const autoware_auto_control_msgs::msg::AckermannControlCommand::ConstSharedPtr msg) {
      tier4_external_api_msgs::msg::VehicleCommandStamped cmd;
      cmd.stamp = msg->stamp;
      cmd.command.velocity = msg->longitudinal.speed;
      cmd.command.acceleration = msg->longitudinal.acceleration;
      pub_cmd_->publish(cmd);
    });
}

void VehicleStatus::onTimer()
{
  const auto subscriptions = {
    std::make_pair(std::static_pointer_cast<const void>(velocity_), "velocity"),
    std::make_pair(std::static_pointer_cast<const void>(steering_), "steering"),
    std::make_pair(std::static_pointer_cast<const void>(turn_indicators_), "turn_indicators"),
    std::make_pair(std::static_pointer_cast<const void>(hazard_lights_), "hazard_lights_"),
    std::make_pair(std::static_pointer_cast<const void>(gear_shift_), "gear_shift")};

  for (const auto & [pointer, topic] : subscriptions) {
    if (!pointer) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "The %s topic is not subscribed", topic);
      return;
    }
  }

  using namespace tier4_auto_msgs_converter;  // NOLINT (for overload)
  try {
    tier4_external_api_msgs::msg::VehicleStatusStamped msg;
    msg.stamp = now();
    msg.status.twist.linear.x = velocity_->longitudinal_velocity;
    msg.status.twist.linear.y = velocity_->lateral_velocity;
    msg.status.twist.angular.z = velocity_->heading_rate;
    msg.status.steering.data = convert(*steering_).data;
    msg.status.turn_signal = convert(convert(*turn_indicators_, *hazard_lights_));
    msg.status.gear_shift = convert(convert(*gear_shift_).shift);
    pub_status_->publish(msg);
  } catch (const std::out_of_range & exception) {
    RCLCPP_ERROR(get_logger(), exception.what());
  }
}

}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::VehicleStatus)
