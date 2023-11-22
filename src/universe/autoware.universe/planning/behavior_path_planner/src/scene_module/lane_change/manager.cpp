// Copyright 2023 TIER IV, Inc.
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

#include "behavior_path_planner/scene_module/lane_change/manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace behavior_path_planner
{

LaneChangeModuleManager::LaneChangeModuleManager(
  rclcpp::Node * node, const std::string & name, const ModuleConfigParameters & config,
  std::shared_ptr<LaneChangeParameters> parameters)
: SceneModuleManagerInterface(node, name, config), parameters_{std::move(parameters)}
{
  rtc_interface_left_ = std::make_shared<RTCInterface>(node, name + "_left");
  rtc_interface_right_ = std::make_shared<RTCInterface>(node, name + "_right");
}

void LaneChangeModuleManager::updateModuleParams(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  [[maybe_unused]] auto p = parameters_;

  [[maybe_unused]] std::string ns = "lane_change.";
  // updateParam<bool>(parameters, ns + ..., ...);

  std::for_each(registered_modules_.begin(), registered_modules_.end(), [&p](const auto & m) {
    m->updateModuleParams(p);
  });
}

}  // namespace behavior_path_planner
