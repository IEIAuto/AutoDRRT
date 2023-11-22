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

#include "obstacle_avoidance_planner/replan_checker.hpp"

#include "motion_utils/motion_utils.hpp"
#include "obstacle_avoidance_planner/utils/trajectory_utils.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include <vector>

namespace obstacle_avoidance_planner
{
ReplanChecker::ReplanChecker(rclcpp::Node * node, const EgoNearestParam & ego_nearest_param)
: ego_nearest_param_(ego_nearest_param), logger_(node->get_logger().get_child("replan_checker"))
{
  max_path_shape_around_ego_lat_dist_ =
    node->declare_parameter<double>("replan.max_path_shape_around_ego_lat_dist");
  max_ego_moving_dist_ = node->declare_parameter<double>("replan.max_ego_moving_dist");
  max_goal_moving_dist_ = node->declare_parameter<double>("replan.max_goal_moving_dist");
  max_delta_time_sec_ = node->declare_parameter<double>("replan.max_delta_time_sec");
}

void ReplanChecker::onParam(const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  updateParam<double>(
    parameters, "replan.max_path_shape_around_ego_lat_dist", max_path_shape_around_ego_lat_dist_);
  updateParam<double>(parameters, "replan.max_ego_moving_dist", max_ego_moving_dist_);
  updateParam<double>(parameters, "replan.max_goal_moving_dist", max_goal_moving_dist_);
  updateParam<double>(parameters, "replan.max_delta_time_sec", max_delta_time_sec_);
}

bool ReplanChecker::isResetRequired(const PlannerData & planner_data)
{
  const auto & p = planner_data;

  const bool reset_required = [&]() {
    // guard for invalid variables
    if (!prev_traj_points_ptr_ || !prev_ego_pose_ptr_) {
      return true;
    }
    const auto & prev_traj_points = *prev_traj_points_ptr_;

    // path shape changes
    if (isPathAroundEgoChanged(planner_data, prev_traj_points)) {
      RCLCPP_INFO(logger_, "Replan with resetting optimization since path shape changed.");
      return true;
    }

    // path goal changes
    if (isPathGoalChanged(planner_data, prev_traj_points)) {
      RCLCPP_INFO(logger_, "Replan with resetting optimization since path goal changed.");
      return true;
    }

    // ego pose is lost or new ego pose is designated in simulation
    const double delta_dist =
      tier4_autoware_utils::calcDistance2d(p.ego_pose, prev_ego_pose_ptr_->position);
    if (max_ego_moving_dist_ < delta_dist) {
      RCLCPP_INFO(
        logger_,
        "Replan with resetting optimization since current ego pose is far from previous ego pose.");
      return true;
    }

    return false;
  }();

  // update previous information required in this function
  prev_traj_points_ptr_ = std::make_shared<std::vector<TrajectoryPoint>>(p.traj_points);
  prev_ego_pose_ptr_ = std::make_shared<geometry_msgs::msg::Pose>(p.ego_pose);

  return reset_required;
}

bool ReplanChecker::isReplanRequired(const rclcpp::Time & current_time)
{
  const bool replan_required = [&]() {
    // guard for invalid variables
    if (!prev_replanned_time_ptr_ || !prev_traj_points_ptr_) {
      return true;
    }

    // time elapses
    const double delta_time_sec = (current_time - *prev_replanned_time_ptr_).seconds();
    if (max_delta_time_sec_ < delta_time_sec) {
      return true;
    }

    return false;
  }();

  // update previous information required in this function
  if (replan_required) {
    prev_replanned_time_ptr_ = std::make_shared<rclcpp::Time>(current_time);
  }

  return replan_required;
}

bool ReplanChecker::isPathAroundEgoChanged(
  const PlannerData & planner_data, const std::vector<TrajectoryPoint> & prev_traj_points) const
{
  const auto & p = planner_data;

  // calculate ego's lateral offset to previous trajectory points
  const auto prev_ego_seg_idx =
    trajectory_utils::findEgoSegmentIndex(prev_traj_points, p.ego_pose, ego_nearest_param_);
  const double prev_ego_lat_offset =
    motion_utils::calcLateralOffset(prev_traj_points, p.ego_pose.position, prev_ego_seg_idx);

  // calculate ego's lateral offset to current trajectory points
  const auto ego_seg_idx =
    trajectory_utils::findEgoSegmentIndex(p.traj_points, p.ego_pose, ego_nearest_param_);
  const double ego_lat_offset =
    motion_utils::calcLateralOffset(p.traj_points, p.ego_pose.position, ego_seg_idx);

  const double diff_ego_lat_offset = prev_ego_lat_offset - ego_lat_offset;
  if (std::abs(diff_ego_lat_offset) < max_path_shape_around_ego_lat_dist_) {
    return false;
  }

  return true;
}

bool ReplanChecker::isPathGoalChanged(
  const PlannerData & planner_data, const std::vector<TrajectoryPoint> & prev_traj_points) const
{
  const auto & p = planner_data;

  // check if the vehicle is stopping
  constexpr double min_vel = 1e-3;
  if (min_vel < std::abs(p.ego_vel)) {
    return false;
  }

  const double goal_moving_dist =
    tier4_autoware_utils::calcDistance2d(p.traj_points.back(), prev_traj_points.back());
  if (goal_moving_dist < max_goal_moving_dist_) {
    return false;
  }

  return true;
}
}  // namespace obstacle_avoidance_planner
