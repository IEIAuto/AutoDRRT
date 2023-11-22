// Copyright 2020 Tier IV, Inc.
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

#include <scene_module/stop_line/manager.hpp>

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace behavior_velocity_planner
{
using lanelet::TrafficSign;

StopLineModuleManager::StopLineModuleManager(rclcpp::Node & node)
: SceneModuleManagerInterface(node, getModuleName())
{
  const std::string ns(getModuleName());
  auto & p = planner_param_;
  p.stop_margin = node.declare_parameter(ns + ".stop_margin", 0.0);
  p.hold_stop_margin_distance = node.declare_parameter(ns + ".hold_stop_margin_distance", 2.0);
  p.stop_duration_sec = node.declare_parameter(ns + ".stop_duration_sec", 1.0);
  p.use_initialization_stop_line_state =
    node.declare_parameter(ns + ".use_initialization_stop_line_state", false);
  // debug
  p.show_stopline_collision_check =
    node.declare_parameter(ns + ".debug.show_stopline_collision_check", false);
}

std::vector<StopLineWithLaneId> StopLineModuleManager::getStopLinesWithLaneIdOnPath(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const lanelet::LaneletMapPtr lanelet_map)
{
  std::vector<StopLineWithLaneId> stop_lines_with_lane_id;

  for (const auto & m : planning_utils::getRegElemMapOnPath<TrafficSign>(
         path, lanelet_map, planner_data_->current_odometry->pose)) {
    const auto & traffic_sign_reg_elem = m.first;
    const int64_t lane_id = m.second.id();
    // Is stop sign?
    if (traffic_sign_reg_elem->type() != "stop_sign") {
      continue;
    }

    for (const auto & stop_line : traffic_sign_reg_elem->refLines()) {
      stop_lines_with_lane_id.push_back(std::make_pair(stop_line, lane_id));
    }
  }

  return stop_lines_with_lane_id;
}

std::set<int64_t> StopLineModuleManager::getStopLineIdSetOnPath(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path,
  const lanelet::LaneletMapPtr lanelet_map)
{
  std::set<int64_t> stop_line_id_set;

  for (const auto & stop_line_with_lane_id : getStopLinesWithLaneIdOnPath(path, lanelet_map)) {
    stop_line_id_set.insert(stop_line_with_lane_id.first.id());
  }

  return stop_line_id_set;
}

void StopLineModuleManager::launchNewModules(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path)
{
  for (const auto & stop_line_with_lane_id :
       getStopLinesWithLaneIdOnPath(path, planner_data_->route_handler_->getLaneletMapPtr())) {
    const auto module_id = stop_line_with_lane_id.first.id();
    const auto lane_id = stop_line_with_lane_id.second;
    if (!isModuleRegistered(module_id)) {
      registerModule(std::make_shared<StopLineModule>(
        module_id, lane_id, stop_line_with_lane_id.first, planner_param_,
        logger_.get_child("stop_line_module"), clock_));
    }
  }
}

std::function<bool(const std::shared_ptr<SceneModuleInterface> &)>
StopLineModuleManager::getModuleExpiredFunction(
  const autoware_auto_planning_msgs::msg::PathWithLaneId & path)
{
  const auto stop_line_id_set =
    getStopLineIdSetOnPath(path, planner_data_->route_handler_->getLaneletMapPtr());

  return [stop_line_id_set](const std::shared_ptr<SceneModuleInterface> & scene_module) {
    return stop_line_id_set.count(scene_module->getModuleId()) == 0;
  };
}
}  // namespace behavior_velocity_planner
