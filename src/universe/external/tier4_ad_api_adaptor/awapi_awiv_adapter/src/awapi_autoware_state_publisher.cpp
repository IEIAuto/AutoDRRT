// Copyright 2020 TIER IV, Inc.
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

#include "awapi_awiv_adapter/awapi_autoware_state_publisher.hpp"

#include "awapi_awiv_adapter/diagnostics_filter.hpp"
#include "tier4_auto_msgs_converter/tier4_auto_msgs_converter.hpp"

#include <regex>
#include <string>
#include <vector>

namespace autoware_api
{
AutowareIvAutowareStatePublisher::AutowareIvAutowareStatePublisher(rclcpp::Node & node)
: logger_(node.get_logger().get_child("awapi_awiv_autoware_state_publisher")),
  clock_(node.get_clock()),
  arrived_goal_(false)
{
  // publisher
  pub_state_ =
    node.create_publisher<tier4_api_msgs::msg::AwapiAutowareStatus>("output/autoware_status", 1);
}

void AutowareIvAutowareStatePublisher::statePublisher(const AutowareInfo & aw_info)
{
  tier4_api_msgs::msg::AwapiAutowareStatus status;

  // input header
  status.header.frame_id = "base_link";
  status.header.stamp = clock_->now();

  // get all info
  getAutowareStateInfo(aw_info.autoware_state_ptr, &status);
  getControlModeInfo(aw_info.control_mode_ptr, &status);
  getGateModeInfo(aw_info.gate_mode_ptr, &status);
  getEmergencyStateInfo(aw_info.mrm_state_ptr, &status);
  getCurrentMaxVelInfo(aw_info.current_max_velocity_ptr, &status);
  getHazardStatusInfo(aw_info, &status);
  getStopReasonInfo(aw_info.stop_reason_ptr, &status);
  getDiagInfo(aw_info, &status);
  getErrorDiagInfo(aw_info, &status);

  // publish info
  pub_state_->publish(status);
}

void AutowareIvAutowareStatePublisher::getAutowareStateInfo(
  const tier4_system_msgs::msg::AutowareState::ConstSharedPtr & autoware_state_ptr,
  tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!autoware_state_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(logger_, *clock_, 5000 /* ms */, "autoware_state is nullptr");
    return;
  }

  // get autoware_state
  using tier4_auto_msgs_converter::convert;
  status->autoware_state = autoware_state_ptr->state;
  status->arrived_goal = isGoal(autoware_state_ptr);
}

void AutowareIvAutowareStatePublisher::getControlModeInfo(
  const autoware_auto_vehicle_msgs::msg::ControlModeReport::ConstSharedPtr & control_mode_ptr,
  tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!control_mode_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(logger_, *clock_, 5000 /* ms */, "control mode is nullptr");
    return;
  }

  // get control mode
  status->control_mode = control_mode_ptr->mode;
}

void AutowareIvAutowareStatePublisher::getGateModeInfo(
  const tier4_control_msgs::msg::GateMode::ConstSharedPtr & gate_mode_ptr,
  tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!gate_mode_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(logger_, *clock_, 5000 /* ms */, "gate mode is nullptr");
    return;
  }

  // get control mode
  status->gate_mode = gate_mode_ptr->data;
}

void AutowareIvAutowareStatePublisher::getEmergencyStateInfo(
  const autoware_adapi_v1_msgs::msg::MrmState::ConstSharedPtr & mrm_state_ptr,
  tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!mrm_state_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(logger_, *clock_, 5000 /* ms */, "mrm_state is nullptr");
    return;
  }

  // get emergency
  using autoware_adapi_v1_msgs::msg::MrmState;
  status->emergency_stopped = (mrm_state_ptr->state == MrmState::MRM_OPERATING) ||
                              (mrm_state_ptr->state == MrmState::MRM_SUCCEEDED) ||
                              (mrm_state_ptr->state == MrmState::MRM_FAILED);
}

void AutowareIvAutowareStatePublisher::getCurrentMaxVelInfo(
  const tier4_planning_msgs::msg::VelocityLimit::ConstSharedPtr & current_max_velocity_ptr,
  tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!current_max_velocity_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] current_max_velocity is nullptr");
    return;
  }

  // get current max velocity
  status->current_max_velocity = current_max_velocity_ptr->max_velocity;
}

void AutowareIvAutowareStatePublisher::getHazardStatusInfo(
  const AutowareInfo & aw_info, tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!aw_info.autoware_state_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] autoware_state is nullptr");
    return;
  }

  if (!aw_info.control_mode_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] control_mode is nullptr");
    return;
  }

  if (!aw_info.hazard_status_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] hazard_status is nullptr");
    return;
  }

  // get emergency
  using tier4_auto_msgs_converter::convert;
  status->hazard_status = convert(*aw_info.hazard_status_ptr);

  // filter leaf diagnostics
  status->hazard_status.status.diagnostics_spf =
    diagnostics_filter::extractLeafDiagnostics(status->hazard_status.status.diagnostics_spf);
  status->hazard_status.status.diagnostics_lf =
    diagnostics_filter::extractLeafDiagnostics(status->hazard_status.status.diagnostics_lf);
  status->hazard_status.status.diagnostics_sf =
    diagnostics_filter::extractLeafDiagnostics(status->hazard_status.status.diagnostics_sf);
  status->hazard_status.status.diagnostics_nf =
    diagnostics_filter::extractLeafDiagnostics(status->hazard_status.status.diagnostics_nf);
}

void AutowareIvAutowareStatePublisher::getStopReasonInfo(
  const tier4_planning_msgs::msg::StopReasonArray::ConstSharedPtr & stop_reason_ptr,
  tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!stop_reason_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(logger_, *clock_, 5000 /* ms */, "stop reason is nullptr");
    return;
  }

  status->stop_reason = *stop_reason_ptr;
}

void AutowareIvAutowareStatePublisher::getDiagInfo(
  const AutowareInfo & aw_info, tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  if (!aw_info.diagnostic_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */, "[AutowareIvAutowareStatePublisher] diagnostics is nullptr");
    return;
  }

  // get diag
  status->diagnostics = diagnostics_filter::extractLeafDiagnostics(aw_info.diagnostic_ptr->status);
}

// This function is tentative and should be replaced with getHazardStatusInfo.
// TODO(Kenji Miyake): Make getErrorDiagInfo users to use getHazardStatusInfo.
void AutowareIvAutowareStatePublisher::getErrorDiagInfo(
  const AutowareInfo & aw_info, tier4_api_msgs::msg::AwapiAutowareStatus * status)
{
  using autoware_auto_vehicle_msgs::msg::ControlModeReport;
  using tier4_system_msgs::msg::AutowareState;

  if (!aw_info.autoware_state_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] autoware_state is nullptr");
    return;
  }

  if (!aw_info.control_mode_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] control mode is nullptr");
    return;
  }

  if (!aw_info.diagnostic_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */, "[AutowareIvAutowareStatePublisher] diagnostics is nullptr");
    return;
  }

  if (!aw_info.hazard_status_ptr) {
    RCLCPP_DEBUG_STREAM_THROTTLE(
      logger_, *clock_, 5000 /* ms */,
      "[AutowareIvAutowareStatePublisher] hazard_status is nullptr");
    return;
  }

  // get diag
  using diagnostic_msgs::msg::DiagnosticStatus;
  const auto & hazard_status = aw_info.hazard_status_ptr->status;
  std::vector<DiagnosticStatus> error_diagnostics;

  for (const auto & hazard_diag : hazard_status.diag_single_point_fault) {
    auto diag = hazard_diag;
    diag.message = "[Single Point Fault]" + hazard_diag.message;
    error_diagnostics.push_back(diag);
  }
  for (const auto & hazard_diag : hazard_status.diag_latent_fault) {
    auto diag = hazard_diag;
    diag.message = "[Latent Fault]" + hazard_diag.message;
    error_diagnostics.push_back(diag);
  }
  for (const auto & hazard_diag : hazard_status.diag_safe_fault) {
    auto diag = hazard_diag;
    diag.message = "[Safe Fault]" + hazard_diag.message;
    error_diagnostics.push_back(diag);
  }
  for (const auto & hazard_diag : hazard_status.diag_no_fault) {
    auto diag = hazard_diag;
    diag.message = "[No Fault]" + hazard_diag.message;
    diag.level = DiagnosticStatus::OK;
    error_diagnostics.push_back(diag);
  }

  // filter leaf diag
  status->error_diagnostics = diagnostics_filter::extractLeafDiagnostics(error_diagnostics);
}

bool AutowareIvAutowareStatePublisher::isGoal(
  const tier4_system_msgs::msg::AutowareState::ConstSharedPtr & autoware_state)
{
  // rename
  const auto & aw_state = autoware_state->state;

  if (aw_state == tier4_system_msgs::msg::AutowareState::ARRIVAL_GOAL) {
    arrived_goal_ = true;
  } else if (  // NOLINT
    prev_state_ == tier4_system_msgs::msg::AutowareState::DRIVING &&
    aw_state == tier4_system_msgs::msg::AutowareState::WAITING_FOR_ROUTE) {
    arrived_goal_ = true;
  }

  if (
    aw_state == tier4_system_msgs::msg::AutowareState::WAITING_FOR_ENGAGE ||
    aw_state == tier4_system_msgs::msg::AutowareState::DRIVING) {
    // cancel goal state
    arrived_goal_ = false;
  }

  prev_state_ = aw_state;

  return arrived_goal_;
}

}  // namespace autoware_api
