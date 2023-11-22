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

#ifndef BEHAVIOR_PATH_PLANNER__UTIL__LANE_CHANGE__UTIL_HPP_
#define BEHAVIOR_PATH_PLANNER__UTIL__LANE_CHANGE__UTIL_HPP_

#include "behavior_path_planner/marker_util/lane_change/debug.hpp"
#include "behavior_path_planner/parameters.hpp"
#include "behavior_path_planner/util/lane_change/lane_change_module_data.hpp"
#include "behavior_path_planner/util/lane_change/lane_change_path.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <route_handler/route_handler.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <lanelet2_core/primitives/Primitive.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace behavior_path_planner::lane_change_utils
{
using autoware_auto_perception_msgs::msg::PredictedObject;
using autoware_auto_perception_msgs::msg::PredictedObjects;
using autoware_auto_perception_msgs::msg::PredictedPath;
using autoware_auto_planning_msgs::msg::PathWithLaneId;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Twist;
using marker_utils::CollisionCheckDebug;
using tier4_autoware_utils::Polygon2d;

PathWithLaneId combineReferencePath(const PathWithLaneId & path1, const PathWithLaneId & path2);

bool isPathInLanelets(
  const PathWithLaneId & path, const lanelet::ConstLanelets & original_lanelets,
  const lanelet::ConstLanelets & target_lanelets);

double getExpectedVelocityWhenDecelerate(
  const double & current_velocity, const double & expected_acceleration,
  const double & lane_change_prepare_duration);

std::pair<double, double> calcLaneChangingSpeedAndDistanceWhenDecelerate(
  const double velocity, const double shift_length, const double deceleration,
  const double min_total_lc_len, const BehaviorPathPlannerParameters & com_param,
  const LaneChangeParameters & lc_param);

std::optional<LaneChangePath> constructCandidatePath(
  const PathWithLaneId & prepare_segment, const PathWithLaneId & lane_changing_segment,
  const PathWithLaneId & target_lane_reference_path, const ShiftLine & shift_line,
  const lanelet::ConstLanelets & original_lanelets, const lanelet::ConstLanelets & target_lanelets,
  const double acceleration, const double prepare_distance, const double prepare_duration,
  const double prepare_speed, const double lane_change_distance, const double lane_changing_speed,
  const BehaviorPathPlannerParameters & params, const LaneChangeParameters & lane_change_param);

std::pair<bool, bool> getLaneChangePaths(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & original_lanelets,
  const lanelet::ConstLanelets & target_lanelets, const Pose & pose, const Twist & twist,
  const PredictedObjects::ConstSharedPtr dynamic_objects,
  const BehaviorPathPlannerParameters & common_parameter, const LaneChangeParameters & parameter,
  const double check_distance, LaneChangePaths * candidate_paths,
  std::unordered_map<std::string, CollisionCheckDebug> * debug_data);

bool isLaneChangePathSafe(
  const LaneChangePath & lane_change_path, const PredictedObjects::ConstSharedPtr dynamic_objects,
  const LaneChangeTargetObjectIndices & dynamic_object_indices, const Pose & current_pose,
  const size_t current_seg_idx, const Twist & current_twist,
  const BehaviorPathPlannerParameters & common_parameters,
  const behavior_path_planner::LaneChangeParameters & lane_change_parameters,
  const double front_decel, const double rear_decel, Pose & ego_pose_before_collision,
  std::unordered_map<std::string, CollisionCheckDebug> & debug_data,
  const double acceleration = 0.0);

bool hasEnoughDistance(
  const LaneChangePath & path, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes, const Pose & current_pose, const Pose & goal_pose,
  const RouteHandler & route_handler, const double minimum_lane_change_length);

ShiftLine getLaneChangeShiftLine(
  const PathWithLaneId & path1, const PathWithLaneId & path2,
  const lanelet::ConstLanelets & target_lanes, const PathWithLaneId & reference_path);

PathWithLaneId getReferencePathFromTargetLane(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & target_lanes,
  const Pose & lane_changing_start_pose, const double target_lane_length,
  const LaneChangePhaseInfo dist_prepare_to_lc_end, const double min_total_lane_changing_distance,
  const double forward_path_length, const double resample_interval, const bool is_goal_in_route);

PathWithLaneId getLaneChangePathPrepareSegment(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & original_lanelets,
  const double arc_length_from_current, const double backward_path_length,
  const double prepare_distance, const double prepare_speed);

PathWithLaneId getLaneChangePathLaneChangingSegment(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & target_lanelets,
  const double forward_path_length, const double arc_length_from_target,
  const double target_lane_length, const LaneChangePhaseInfo dist_prepare_to_lc_end,
  const double lane_changing_speed, const double total_required_min_dist);

bool isEgoWithinOriginalLane(
  const lanelet::ConstLanelets & current_lanes, const Pose & current_pose,
  const BehaviorPathPlannerParameters & common_param);

void get_turn_signal_info(
  const LaneChangePath & lane_change_path, TurnSignalInfo * turn_signal_info);

std::vector<DrivableLanes> generateDrivableLanes(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & lane_change_lanes);

std::optional<LaneChangePath> getAbortPaths(
  const std::shared_ptr<const PlannerData> & planner_data, const LaneChangePath & selected_path,
  const Pose & ego_lerp_pose_before_collision, const BehaviorPathPlannerParameters & common_param,
  const LaneChangeParameters & lane_change_param);

double getLateralShift(const LaneChangePath & path);

bool hasEnoughDistanceToLaneChangeAfterAbort(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & current_lanes,
  const Pose & curent_pose, const double abort_return_dist,
  const BehaviorPathPlannerParameters & common_param);

lanelet::ConstLanelets getExtendedTargetLanesForCollisionCheck(
  const RouteHandler & route_handler, const lanelet::ConstLanelet & target_lanes,
  const Pose & current_pose, const double backward_length);

LaneChangeTargetObjectIndices filterObjectIndices(
  const LaneChangePaths & lane_change_paths, const PredictedObjects & objects,
  const lanelet::ConstLanelets & target_backward_lanes, const Pose & current_pose,
  const double forward_path_length, const double filter_width,
  const bool ignore_unknown_obj = false);

double calcLateralBufferForFiltering(const double vehicle_width, const double lateral_buffer = 0.0);

}  // namespace behavior_path_planner::lane_change_utils
#endif  // BEHAVIOR_PATH_PLANNER__UTIL__LANE_CHANGE__UTIL_HPP_
