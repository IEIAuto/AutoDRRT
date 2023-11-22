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

#ifndef BEHAVIOR_PATH_PLANNER__UTIL__PULL_OVER__UTIL_HPP_
#define BEHAVIOR_PATH_PLANNER__UTIL__PULL_OVER__UTIL_HPP_

#include "behavior_path_planner/util/pull_over/goal_searcher_base.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lane_departure_checker/lane_departure_checker.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_path.hpp>
#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <lanelet2_core/primitives/Primitive.h>

#include <memory>
#include <string>
#include <vector>

namespace behavior_path_planner
{
namespace pull_over_utils
{
using autoware_auto_perception_msgs::msg::PredictedObjects;
using autoware_auto_perception_msgs::msg::PredictedPath;
using autoware_auto_planning_msgs::msg::PathWithLaneId;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Twist;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

// TODO(sugahara) move to util
PathWithLaneId combineReferencePath(const PathWithLaneId & path1, const PathWithLaneId & path2);
lanelet::ConstLanelets getPullOverLanes(const RouteHandler & route_handler);
PredictedObjects filterObjectsByLateralDistance(
  const Pose & ego_pose, const double vehicle_width, const PredictedObjects & objects,
  const double distance_thresh, const bool filter_inside);

// debug
MarkerArray createPullOverAreaMarkerArray(
  const tier4_autoware_utils::MultiPolygon2d area_polygons, const std_msgs::msg::Header & header,
  const std_msgs::msg::ColorRGBA & color, const double z);
MarkerArray createPosesMarkerArray(
  const std::vector<Pose> & poses, std::string && ns, const std_msgs::msg::ColorRGBA & color);
MarkerArray createTextsMarkerArray(
  const std::vector<Pose> & poses, std::string && ns, const std_msgs::msg::ColorRGBA & color);
MarkerArray createGoalCandidatesMarkerArray(
  GoalCandidates & goal_candidates, const std_msgs::msg::ColorRGBA & color);
}  // namespace pull_over_utils
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__UTIL__PULL_OVER__UTIL_HPP_
