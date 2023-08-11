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

#include "rtc_controller.hpp"

#include <memory>

RTCModule::RTCModule(rclcpp::Node * node, const std::string & name)
{
  using namespace std::literals::chrono_literals;
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(node);

  module_sub_ = node->create_subscription<CooperateStatusArray>(
    cooperate_status_namespace_ + "/" + name, rclcpp::QoS(1),
    std::bind(&RTCModule::moduleCallback, this, _1));

  cli_set_module_ = proxy.create_client<CooperateCommands>(
    cooperate_commands_namespace_ + "/" + name, rmw_qos_profile_services_default);
}

void RTCModule::moduleCallback(const CooperateStatusArray::ConstSharedPtr message)
{
  module_statuses_ = message->statuses;
}

void RTCModule::insertMessage(std::vector<CooperateStatus> & cooperate_statuses)
{
  cooperate_statuses.insert(
    cooperate_statuses.end(), module_statuses_.begin(), module_statuses_.end());
}

void RTCModule::callService(
  CooperateCommands::Request::SharedPtr request,
  const CooperateCommands::Response::SharedPtr & responses)
{
  const auto [status, resp] = cli_set_module_->call(request);
  if (!tier4_api_utils::is_success(status)) {
    return;
  }
  responses->responses.insert(
    responses->responses.end(), resp->responses.begin(), resp->responses.end());
}

namespace external_api
{
RTCController::RTCController(const rclcpp::NodeOptions & options)
: Node("external_api_rtc_controller", options)
{
  using namespace std::literals::chrono_literals;
  using std::placeholders::_1;
  using std::placeholders::_2;
  tier4_api_utils::ServiceProxyNodeInterface proxy(this);

  blind_spot_ = std::make_unique<RTCModule>(this, "blind_spot");
  crosswalk_ = std::make_unique<RTCModule>(this, "crosswalk");
  detection_area_ = std::make_unique<RTCModule>(this, "detection_area");
  intersection_ = std::make_unique<RTCModule>(this, "intersection");
  no_stopping_area_ = std::make_unique<RTCModule>(this, "no_stopping_area");
  occlusion_spot_ = std::make_unique<RTCModule>(this, "occlusion_spot");
  traffic_light_ = std::make_unique<RTCModule>(this, "traffic_light");
  virtual_traffic_light_ = std::make_unique<RTCModule>(this, "virtual_traffic_light");
  lane_change_left_ = std::make_unique<RTCModule>(this, "lane_change_left");
  lane_change_right_ = std::make_unique<RTCModule>(this, "lane_change_right");
  avoidance_left_ = std::make_unique<RTCModule>(this, "avoidance_left");
  avoidance_right_ = std::make_unique<RTCModule>(this, "avoidance_right");
  pull_over_ = std::make_unique<RTCModule>(this, "pull_over");
  pull_out_ = std::make_unique<RTCModule>(this, "pull_out");

  rtc_status_pub_ =
    create_publisher<CooperateStatusArray>("/api/external/get/rtc_status", rclcpp::QoS(1));

  group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_set_rtc_ = proxy.create_service<CooperateCommands>(
    "/api/external/set/rtc_commands", std::bind(&RTCController::setRTC, this, _1, _2),
    rmw_qos_profile_services_default, group_);

  timer_ = rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&RTCController::onTimer, this));
}

void RTCController::insertionSortAndValidation(std::vector<CooperateStatus> & statuses_vector)
{
  if (statuses_vector.empty()) {
    return;
  }
  tier4_rtc_msgs::msg::CooperateStatus current_status;
  checkInfDistance(statuses_vector[0]);
  for (size_t i = 1; i < statuses_vector.size(); i++) {
    checkInfDistance(statuses_vector[i]);
    current_status = statuses_vector[i];
    int j = i - 1;

    while (j >= 0 && current_status.start_distance < statuses_vector[j].start_distance) {
      statuses_vector[j + 1] = statuses_vector[j];
      j = j - 1;
    }
    statuses_vector[j + 1] = current_status;
  }
}

void RTCController::checkInfDistance(CooperateStatus & status)  // Temporary fix for ROS2 humble
{
  if (!std::isfinite(status.start_distance)) {
    status.start_distance = -100000.0;
  }
  if (!std::isfinite(status.finish_distance)) {
    status.finish_distance = -100000.0;
  }
}

void RTCController::onTimer()
{
  std::vector<CooperateStatus> cooperate_statuses;
  blind_spot_->insertMessage(cooperate_statuses);
  crosswalk_->insertMessage(cooperate_statuses);
  detection_area_->insertMessage(cooperate_statuses);
  intersection_->insertMessage(cooperate_statuses);
  no_stopping_area_->insertMessage(cooperate_statuses);
  occlusion_spot_->insertMessage(cooperate_statuses);
  traffic_light_->insertMessage(cooperate_statuses);
  virtual_traffic_light_->insertMessage(cooperate_statuses);
  lane_change_left_->insertMessage(cooperate_statuses);
  lane_change_right_->insertMessage(cooperate_statuses);
  avoidance_left_->insertMessage(cooperate_statuses);
  avoidance_right_->insertMessage(cooperate_statuses);
  pull_over_->insertMessage(cooperate_statuses);
  pull_out_->insertMessage(cooperate_statuses);

  insertionSortAndValidation(cooperate_statuses);

  CooperateStatusArray msg;
  msg.stamp = now();
  msg.statuses = cooperate_statuses;
  rtc_status_pub_->publish(msg);
}

void RTCController::setRTC(
  const CooperateCommands::Request::SharedPtr requests,
  const CooperateCommands::Response::SharedPtr responses)
{
  for (tier4_rtc_msgs::msg::CooperateCommand & command : requests->commands) {
    auto request = std::make_shared<CooperateCommands::Request>();
    request->commands = {command};
    switch (command.module.type) {
      case Module::LANE_CHANGE_LEFT: {
        lane_change_left_->callService(request, responses);
        break;
      }
      case Module::LANE_CHANGE_RIGHT: {
        lane_change_right_->callService(request, responses);
        break;
      }
      case Module::AVOIDANCE_LEFT: {
        avoidance_left_->callService(request, responses);
        break;
      }
      case Module::AVOIDANCE_RIGHT: {
        avoidance_right_->callService(request, responses);
        break;
      }
      case Module::PULL_OVER: {
        pull_over_->callService(request, responses);
        break;
      }
      case Module::PULL_OUT: {
        pull_out_->callService(request, responses);
        break;
      }
      case Module::TRAFFIC_LIGHT: {
        traffic_light_->callService(request, responses);
        break;
      }
      case Module::INTERSECTION: {
        intersection_->callService(request, responses);
        break;
      }
      case Module::CROSSWALK: {
        crosswalk_->callService(request, responses);
        break;
      }
      case Module::BLIND_SPOT: {
        blind_spot_->callService(request, responses);
        break;
      }
      case Module::DETECTION_AREA: {
        detection_area_->callService(request, responses);
        break;
      }
      case Module::NO_STOPPING_AREA: {
        no_stopping_area_->callService(request, responses);
        break;
      }
      case Module::OCCLUSION_SPOT: {
        occlusion_spot_->callService(request, responses);
        break;
      }
        // virtual_traffic not found
    }
  }
}

}  // namespace external_api

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(external_api::RTCController)
