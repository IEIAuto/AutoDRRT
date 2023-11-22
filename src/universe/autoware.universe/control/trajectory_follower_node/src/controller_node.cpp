// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#include "trajectory_follower_node/controller_node.hpp"

#include "mpc_lateral_controller/mpc_lateral_controller.hpp"
#include "pid_longitudinal_controller/pid_longitudinal_controller.hpp"
#include "pure_pursuit/pure_pursuit_lateral_controller.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::motion::control::trajectory_follower_node
{
Controller::Controller(const rclcpp::NodeOptions & node_options) : Node("controller", node_options)
{
  using std::placeholders::_1;

  const double ctrl_period = declare_parameter<double>("ctrl_period");
  timeout_thr_sec_ = declare_parameter<double>("timeout_thr_sec");

  const auto lateral_controller_mode =
    getLateralControllerMode(declare_parameter<std::string>("lateral_controller_mode", "mpc"));
  switch (lateral_controller_mode) {
    case LateralControllerMode::MPC: {
      lateral_controller_ = std::make_shared<mpc_lateral_controller::MpcLateralController>(*this);
      break;
    }
    case LateralControllerMode::PURE_PURSUIT: {
      lateral_controller_ = std::make_shared<pure_pursuit::PurePursuitLateralController>(*this);
      break;
    }
    default:
      throw std::domain_error("[LateralController] invalid algorithm");
  }

  const auto longitudinal_controller_mode = getLongitudinalControllerMode(
    declare_parameter<std::string>("longitudinal_controller_mode", "pid"));
  switch (longitudinal_controller_mode) {
    case LongitudinalControllerMode::PID: {
      longitudinal_controller_ =
        std::make_shared<pid_longitudinal_controller::PidLongitudinalController>(*this);
      break;
    }
    default:
      throw std::domain_error("[LongitudinalController] invalid algorithm");
  }

  sub_ref_path_ = create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/input/reference_trajectory", rclcpp::QoS{1}, std::bind(&Controller::onTrajectory, this, _1));
  sub_steering_ = create_subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "~/input/current_steering", rclcpp::QoS{1}, std::bind(&Controller::onSteering, this, _1));
  sub_odometry_ = create_subscription<nav_msgs::msg::Odometry>(
    "~/input/current_odometry", rclcpp::QoS{1}, std::bind(&Controller::onOdometry, this, _1));
  sub_accel_ = create_subscription<geometry_msgs::msg::AccelWithCovarianceStamped>(
    "~/input/current_accel", rclcpp::QoS{1}, std::bind(&Controller::onAccel, this, _1));
  sub_operation_mode_ = create_subscription<OperationModeState>(
    "~/input/current_operation_mode", rclcpp::QoS{1},
    [this](const OperationModeState::SharedPtr msg) { current_operation_mode_ptr_ = msg; });
  control_cmd_pub_ = create_publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>(
    "~/output/control_cmd", rclcpp::QoS{1}.transient_local());
  
  
  sub_scenario_ = create_subscription<tier4_planning_msgs::msg::Scenario>(
    "/planning/scenario_planning/scenario", 1, std::bind(&Controller::onScenario, this, _1));

  debug_marker_pub_ =
    create_publisher<visualization_msgs::msg::MarkerArray>("~/output/debug_marker", rclcpp::QoS{1});

  // Timer
  {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(ctrl_period));
    timer_control_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&Controller::callbackTimerControl, this));
    
    RCLCPP_INFO(get_logger(), "control period, %f", ctrl_period);
  }
}

void Controller::onScenario(const tier4_planning_msgs::msg::Scenario::ConstSharedPtr msg)
{
  current_scenario_ = msg->current_scenario;
}

Controller::LateralControllerMode Controller::getLateralControllerMode(
  const std::string & controller_mode) const
{
  if (controller_mode == "mpc") return LateralControllerMode::MPC;
  if (controller_mode == "pure_pursuit") return LateralControllerMode::PURE_PURSUIT;

  return LateralControllerMode::INVALID;
}

Controller::LongitudinalControllerMode Controller::getLongitudinalControllerMode(
  const std::string & controller_mode) const
{
  if (controller_mode == "pid") return LongitudinalControllerMode::PID;

  return LongitudinalControllerMode::INVALID;
}

void Controller::onTrajectory(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
  current_trajectory_ptr_ = msg;
}

void Controller::onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_odometry_ptr_ = msg;
}

void Controller::onSteering(const autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr msg)
{
  current_steering_ptr_ = msg;
}

void Controller::onAccel(const geometry_msgs::msg::AccelWithCovarianceStamped::SharedPtr msg)
{
  current_accel_ptr_ = msg;
}

bool Controller::isTimeOut(
  const trajectory_follower::LongitudinalOutput & lon_out,
  const trajectory_follower::LateralOutput & lat_out)
{
  const auto now = this->now();
  if ((now - lat_out.control_cmd.stamp).seconds() > timeout_thr_sec_) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000 /*ms*/,
      "Lateral control command too old, control_cmd will not be published.");
    return true;
  }
  if ((now - lon_out.control_cmd.stamp).seconds() > timeout_thr_sec_) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000 /*ms*/,
      "Longitudinal control command too old, control_cmd will not be published.");
    return true;
  }
  return false;
}

boost::optional<trajectory_follower::InputData> Controller::createInputData(
  rclcpp::Clock & clock) const
{
  if (!current_trajectory_ptr_) {
    RCLCPP_INFO_THROTTLE(get_logger(), clock, 5000, "Waiting for trajectory.");
    return {};
  }

  if (!current_odometry_ptr_) {
    RCLCPP_INFO_THROTTLE(get_logger(), clock, 5000, "Waiting for current odometry.");
    return {};
  }

  if (!current_steering_ptr_) {
    RCLCPP_INFO_THROTTLE(get_logger(), clock, 5000, "Waiting for current steering.");
    return {};
  }

  if (!current_accel_ptr_) {
    RCLCPP_INFO_THROTTLE(get_logger(), clock, 5000, "Waiting for current accel.");
    return {};
  }

  if (!current_operation_mode_ptr_) {
    RCLCPP_INFO_THROTTLE(get_logger(), clock, 5000, "Waiting for current operation mode.");
    return {};
  }

  trajectory_follower::InputData input_data;
  input_data.current_trajectory = *current_trajectory_ptr_;
  input_data.current_odometry = *current_odometry_ptr_;
  input_data.current_steering = *current_steering_ptr_;
  input_data.current_accel = *current_accel_ptr_;
  input_data.current_operation_mode = *current_operation_mode_ptr_;
  input_data.current_scenario_ = current_scenario_;

  return input_data;
}

void Controller::callbackTimerControl()
{
  
  rclcpp::Time start_time = this->get_clock()->now();

  // 1. create input data
  const auto input_data = createInputData(*get_clock());
  if (!input_data) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000, "Control is skipped since input data is not ready.");
    return;
  }

  rclcpp::Time end_time_1 = this->get_clock()->now();
  rclcpp::Duration duration = end_time_1 - start_time;  // 计算时间差
  double elapsed_time_ms_1 = duration.seconds() * 1000.0;  // 转换为毫秒
  RCLCPP_INFO(this->get_logger(), "----------------create data---------------------: %.2f ms", elapsed_time_ms_1);

  // 2. check if controllers are ready
  const bool is_lat_ready = lateral_controller_->isReady(*input_data);
  const bool is_lon_ready = longitudinal_controller_->isReady(*input_data);
  if (!is_lat_ready || !is_lon_ready) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Control is skipped since lateral and/or longitudinal controllers are not ready to run.");
    return;
  }

  rclcpp::Time end_time_2 = this->get_clock()->now();
  rclcpp::Duration duration_2 = end_time_2 - end_time_1;  // 计算时间差
  double elapsed_time_ms_2 = duration_2.seconds() * 1000.0;  // 转换为毫秒
  RCLCPP_INFO(this->get_logger(), "1----------------check input data---------------------: %.2f ms", elapsed_time_ms_2);

  // 3. run controllers
  const auto lat_out = lateral_controller_->run(*input_data);
  
  rclcpp::Time end_time_3 = this->get_clock()->now();
  rclcpp::Duration duration_3 = end_time_3 - end_time_2;  // 计算时间差
  double elapsed_time_ms_3 = duration_3.seconds() * 1000.0;  // 转换为毫秒
  RCLCPP_INFO(this->get_logger(), "1----------------run controllers_lateral---------------------: %.2f ms", elapsed_time_ms_3);

  const auto lon_out = longitudinal_controller_->run(*input_data);
  
  rclcpp::Time end_time_4 = this->get_clock()->now();
  rclcpp::Duration duration_4 = end_time_4 - end_time_3;  // 计算时间差
  double elapsed_time_ms_4 = duration_4.seconds() * 1000.0;  // 转换为毫秒
  RCLCPP_INFO(this->get_logger(), "1----------------run controllers_lon---------------------: %.2f ms", elapsed_time_ms_4);

  // 4. sync with each other controllers
  longitudinal_controller_->sync(lat_out.sync_data);
  lateral_controller_->sync(lon_out.sync_data);
  
  rclcpp::Time end_time_5 = this->get_clock()->now();
  rclcpp::Duration duration_5 = end_time_5 - end_time_4;  // 计算时间差
  double elapsed_time_ms_5 = duration_5.seconds() * 1000.0;  // 转换为毫秒
  RCLCPP_INFO(this->get_logger(), "1----------------run sys---------------------: %.2f ms", elapsed_time_ms_5);

  // TODO(Horibe): Think specification. This comes from the old implementation.
  if (isTimeOut(lon_out, lat_out)) return;

  // 5. publish control command
  autoware_auto_control_msgs::msg::AckermannControlCommand out;
  out.stamp = this->now();
  out.lateral = lat_out.control_cmd;
  out.longitudinal = lon_out.control_cmd;
  control_cmd_pub_->publish(out);

  // 6. publish debug marker
  publishDebugMarker(*input_data, lat_out);
}

void Controller::publishDebugMarker(
  const trajectory_follower::InputData & input_data,
  const trajectory_follower::LateralOutput & lat_out) const
{
  visualization_msgs::msg::MarkerArray debug_marker_array{};

  // steer converged marker
  {
    auto marker = tier4_autoware_utils::createDefaultMarker(
      "map", this->now(), "steer_converged", 0, visualization_msgs::msg::Marker::TEXT_VIEW_FACING,
      tier4_autoware_utils::createMarkerScale(0.0, 0.0, 1.0),
      tier4_autoware_utils::createMarkerColor(1.0, 1.0, 1.0, 0.99));
    marker.pose = input_data.current_odometry.pose.pose;

    std::stringstream ss;
    const double current = input_data.current_steering.steering_tire_angle;
    const double cmd = lat_out.control_cmd.steering_tire_angle;
    const double diff = current - cmd;
    ss << "current:" << current << " cmd:" << cmd << " diff:" << diff
       << (lat_out.sync_data.is_steer_converged ? " (converged)" : " (not converged)");
    marker.text = ss.str();

    debug_marker_array.markers.push_back(marker);
  }

  debug_marker_pub_->publish(debug_marker_array);
}

}  // namespace autoware::motion::control::trajectory_follower_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::motion::control::trajectory_follower_node::Controller)
