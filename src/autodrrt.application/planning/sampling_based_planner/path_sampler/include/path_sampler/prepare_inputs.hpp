// Copyright 2023 Tier IV, Inc.
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

#ifndef PATH_SAMPLER__PREPARE_INPUTS_HPP_
#define PATH_SAMPLER__PREPARE_INPUTS_HPP_

#include "frenet_planner/structures.hpp"
#include "path_sampler/parameters.hpp"
#include "path_sampler/type_alias.hpp"
#include "sampler_common/transform/spline_transform.hpp"

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/path.hpp>
#include <autoware_auto_planning_msgs/msg/path_point.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/detail/occupancy_grid__struct.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <string>
#include <vector>

namespace path_sampler
{
/// @brief prepare constraints
void prepareConstraints(
  sampler_common::Constraints & constraints, const PredictedObjects & predicted_objects,
  const std::vector<geometry_msgs::msg::Point> & left_bound,
  const std::vector<geometry_msgs::msg::Point> & right_bound);
/// @brief prepare sampling parameters to generate Frenet paths
frenet_planner::SamplingParameters prepareSamplingParameters(
  const sampler_common::State & initial_state, const double base_length,
  const sampler_common::transform::Spline2D & path_spline, const Parameters & params);
/// @brief prepare the 2D spline representation of the given Path message
sampler_common::transform::Spline2D preparePathSpline(
  const std::vector<TrajectoryPoint> & path_msg, const bool smooth_path);
}  // namespace path_sampler

#endif  // PATH_SAMPLER__PREPARE_INPUTS_HPP_
