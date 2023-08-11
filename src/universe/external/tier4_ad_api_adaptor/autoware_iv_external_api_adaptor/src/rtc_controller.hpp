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

#ifndef RTC_CONTROLLER_HPP_
#define RTC_CONTROLLER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <tier4_api_utils/tier4_api_utils.hpp>

#include "tier4_rtc_msgs/msg/cooperate_command.hpp"
#include "tier4_rtc_msgs/msg/cooperate_status.hpp"
#include "tier4_rtc_msgs/msg/cooperate_status_array.hpp"
#include "tier4_rtc_msgs/msg/module.hpp"
#include "tier4_rtc_msgs/srv/cooperate_commands.hpp"

#include <memory>
#include <string>
#include <vector>

using CooperateCommands = tier4_rtc_msgs::srv::CooperateCommands;
using CooperateStatusArray = tier4_rtc_msgs::msg::CooperateStatusArray;
using CooperateStatus = tier4_rtc_msgs::msg::CooperateStatus;
using Module = tier4_rtc_msgs::msg::Module;

class RTCModule
{
public:
  std::string cooperate_status_namespace_ = "/planning/cooperate_status";
  std::string cooperate_commands_namespace_ = "/planning/cooperate_commands";
  std::vector<CooperateStatus> module_statuses_;
  rclcpp::Subscription<CooperateStatusArray>::SharedPtr module_sub_;
  tier4_api_utils::Client<CooperateCommands>::SharedPtr cli_set_module_;

  RTCModule(rclcpp::Node * node, const std::string & name);
  void moduleCallback(const CooperateStatusArray::ConstSharedPtr message);
  void insertMessage(std::vector<CooperateStatus> & cooperate_statuses);
  void callService(
    CooperateCommands::Request::SharedPtr request,
    const CooperateCommands::Response::SharedPtr & responses);
};

namespace external_api
{
class RTCController : public rclcpp::Node
{
public:
  explicit RTCController(const rclcpp::NodeOptions & options);

private:
  std::unique_ptr<RTCModule> blind_spot_;
  std::unique_ptr<RTCModule> crosswalk_;
  std::unique_ptr<RTCModule> detection_area_;
  std::unique_ptr<RTCModule> intersection_;
  std::unique_ptr<RTCModule> no_stopping_area_;
  std::unique_ptr<RTCModule> occlusion_spot_;
  std::unique_ptr<RTCModule> traffic_light_;
  std::unique_ptr<RTCModule> virtual_traffic_light_;
  std::unique_ptr<RTCModule> lane_change_left_;
  std::unique_ptr<RTCModule> lane_change_right_;
  std::unique_ptr<RTCModule> avoidance_left_;
  std::unique_ptr<RTCModule> avoidance_right_;
  std::unique_ptr<RTCModule> pull_over_;
  std::unique_ptr<RTCModule> pull_out_;

  /* publishers */
  rclcpp::Publisher<CooperateStatusArray>::SharedPtr rtc_status_pub_;
  /* service from external */
  rclcpp::CallbackGroup::SharedPtr group_;
  tier4_api_utils::Service<CooperateCommands>::SharedPtr srv_set_rtc_;

  /* Timer */
  rclcpp::TimerBase::SharedPtr timer_;

  void insertionSortAndValidation(std::vector<CooperateStatus> & statuses_vector);
  void checkInfDistance(CooperateStatus & status);

  void setRTC(
    const CooperateCommands::Request::SharedPtr requests,
    const CooperateCommands::Response::SharedPtr responses);

  // ros callback
  void onTimer();
};

}  // namespace external_api

#endif  // RTC_CONTROLLER_HPP_
