// Copyright 2021 Tier IV, Inc.
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

#ifndef MANAGER_HPP_
#define MANAGER_HPP_

#include "scene_no_stopping_area.hpp"

#include <behavior_velocity_planner_common/plugin_interface.hpp>
#include <behavior_velocity_planner_common/plugin_wrapper.hpp>
#include <behavior_velocity_planner_common/scene_module_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>

#include <functional>
#include <memory>

namespace behavior_velocity_planner
{
class NoStoppingAreaModuleManager : public SceneModuleManagerInterfaceWithRTC
{
public:
  explicit NoStoppingAreaModuleManager(rclcpp::Node & node);

  const char * getModuleName() override { return "no_stopping_area"; }

private:
  NoStoppingAreaModule::PlannerParam planner_param_;

  void launchNewModules(const autoware_auto_planning_msgs::msg::PathWithLaneId & path) override;

  std::function<bool(const std::shared_ptr<SceneModuleInterface> &)> getModuleExpiredFunction(
    const autoware_auto_planning_msgs::msg::PathWithLaneId & path) override;
};

class NoStoppingAreaModulePlugin : public PluginWrapper<NoStoppingAreaModuleManager>
{
};

}  // namespace behavior_velocity_planner

#endif  // MANAGER_HPP_
