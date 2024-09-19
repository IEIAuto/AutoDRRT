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

#include "path_sampler/utils/trajectory_utils.hpp"

#include "motion_utils/resample/resample.hpp"
#include "motion_utils/trajectory/tmp_conversion.hpp"
#include "path_sampler/utils/geometry_utils.hpp"
#include "tf2/utils.h"

#include "autoware_auto_planning_msgs/msg/path_point.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory_point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stack>
#include <vector>

namespace path_sampler
{
namespace trajectory_utils
{
void compensateLastPose(
  const PathPoint & last_path_point, std::vector<TrajectoryPoint> & traj_points,
  const double delta_dist_threshold, const double delta_yaw_threshold)
{
  if (traj_points.empty()) {
    traj_points.push_back(convertToTrajectoryPoint(last_path_point));
    return;
  }

  const geometry_msgs::msg::Pose last_traj_pose = traj_points.back().pose;

  const double dist =
    tier4_autoware_utils::calcDistance2d(last_path_point.pose.position, last_traj_pose.position);
  const double norm_diff_yaw = [&]() {
    const double diff_yaw =
      tf2::getYaw(last_path_point.pose.orientation) - tf2::getYaw(last_traj_pose.orientation);
    return tier4_autoware_utils::normalizeRadian(diff_yaw);
  }();
  if (dist > delta_dist_threshold || std::fabs(norm_diff_yaw) > delta_yaw_threshold) {
    traj_points.push_back(convertToTrajectoryPoint(last_path_point));
  }
}

Trajectory createTrajectory(
  const std_msgs::msg::Header & header, const std::vector<TrajectoryPoint> & traj_points)
{
  auto traj = motion_utils::convertToTrajectory(traj_points);
  traj.header = header;

  return traj;
}

std::vector<TrajectoryPoint> resampleTrajectoryPoints(
  const std::vector<TrajectoryPoint> traj_points, const double interval)
{
  constexpr bool enable_resampling_stop_point = true;

  const auto traj = motion_utils::convertToTrajectory(traj_points);
  const auto resampled_traj = motion_utils::resampleTrajectory(
    traj, interval, false, true, true, enable_resampling_stop_point);
  return motion_utils::convertToTrajectoryPointArray(resampled_traj);
}

// NOTE: stop point will not be resampled
std::vector<TrajectoryPoint> resampleTrajectoryPointsWithoutStopPoint(
  const std::vector<TrajectoryPoint> traj_points, const double interval)
{
  constexpr bool enable_resampling_stop_point = false;

  const auto traj = motion_utils::convertToTrajectory(traj_points);
  const auto resampled_traj = motion_utils::resampleTrajectory(
    traj, interval, false, true, true, enable_resampling_stop_point);
  return motion_utils::convertToTrajectoryPointArray(resampled_traj);
}

void insertStopPoint(
  std::vector<TrajectoryPoint> & traj_points, const geometry_msgs::msg::Pose & input_stop_pose,
  const size_t stop_seg_idx)
{
  const double offset_to_segment = motion_utils::calcLongitudinalOffsetToSegment(
    traj_points, stop_seg_idx, input_stop_pose.position);

  const auto traj_spline = SplineInterpolationPoints2d(traj_points);
  const auto stop_pose = traj_spline.getSplineInterpolatedPose(stop_seg_idx, offset_to_segment);

  if (geometry_utils::isSamePoint(traj_points.at(stop_seg_idx), stop_pose)) {
    traj_points.at(stop_seg_idx).longitudinal_velocity_mps = 0.0;
    return;
  }
  if (geometry_utils::isSamePoint(traj_points.at(stop_seg_idx + 1), stop_pose)) {
    traj_points.at(stop_seg_idx + 1).longitudinal_velocity_mps = 0.0;
    return;
  }

  TrajectoryPoint additional_traj_point;
  additional_traj_point.pose = stop_pose;
  additional_traj_point.longitudinal_velocity_mps = 0.0;

  traj_points.insert(traj_points.begin() + stop_seg_idx + 1, additional_traj_point);
}
}  // namespace trajectory_utils
}  // namespace path_sampler
