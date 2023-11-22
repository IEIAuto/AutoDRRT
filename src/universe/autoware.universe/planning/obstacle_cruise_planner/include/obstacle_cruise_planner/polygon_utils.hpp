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

#ifndef OBSTACLE_CRUISE_PLANNER__POLYGON_UTILS_HPP_
#define OBSTACLE_CRUISE_PLANNER__POLYGON_UTILS_HPP_

#include "obstacle_cruise_planner/type_alias.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"
#include "vehicle_info_util/vehicle_info_util.hpp"

#include <boost/geometry.hpp>
#include <boost/optional.hpp>

#include <limits>
#include <vector>

namespace polygon_utils
{
namespace bg = boost::geometry;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Polygon2d;

boost::optional<size_t> getCollisionIndex(
  const Trajectory & traj, const std::vector<Polygon2d> & traj_polygons,
  const geometry_msgs::msg::PoseStamped & obj_pose, const Shape & shape,
  std::vector<geometry_msgs::msg::PointStamped> & collision_points,
  const double max_dist = std::numeric_limits<double>::max());

std::vector<geometry_msgs::msg::PointStamped> getCollisionPoints(
  const Trajectory & traj, const std::vector<Polygon2d> & traj_polygons,
  const std_msgs::msg::Header & obj_header, const PredictedPath & predicted_path,
  const Shape & shape, const rclcpp::Time & current_time,
  const double vehicle_max_longitudinal_offset, const bool is_driving_forward,
  std::vector<size_t> & collision_index, const double max_dist = std::numeric_limits<double>::max(),
  const double max_prediction_time_for_collision_check = std::numeric_limits<double>::max());

std::vector<geometry_msgs::msg::PointStamped> willCollideWithSurroundObstacle(
  const Trajectory & traj, const std::vector<Polygon2d> & traj_polygons,
  const std_msgs::msg::Header & obj_header, const PredictedPath & predicted_path,
  const Shape & shape, const rclcpp::Time & current_time, const double max_dist,
  const double ego_obstacle_overlap_time_threshold,
  const double max_prediction_time_for_collision_check, std::vector<size_t> & collision_index,
  const double vehicle_max_longitudinal_offset, const bool is_driving_forward);

std::vector<Polygon2d> createOneStepPolygons(
  const Trajectory & traj, const vehicle_info_util::VehicleInfo & vehicle_info,
  const double expand_width);

geometry_msgs::msg::PointStamped calcNearestCollisionPoint(
  const size_t & first_within_idx,
  const std::vector<geometry_msgs::msg::PointStamped> & collision_points,
  const Trajectory & decimated_traj, const double vehicle_max_longitudinal_offset,
  const bool is_driving_forward);
}  // namespace polygon_utils

#endif  // OBSTACLE_CRUISE_PLANNER__POLYGON_UTILS_HPP_
