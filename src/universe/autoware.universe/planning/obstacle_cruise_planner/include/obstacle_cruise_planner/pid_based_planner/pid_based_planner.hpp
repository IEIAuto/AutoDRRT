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

#ifndef OBSTACLE_CRUISE_PLANNER__PID_BASED_PLANNER__PID_BASED_PLANNER_HPP_
#define OBSTACLE_CRUISE_PLANNER__PID_BASED_PLANNER__PID_BASED_PLANNER_HPP_

#include "obstacle_cruise_planner/pid_based_planner/cruise_planning_debug_info.hpp"
#include "obstacle_cruise_planner/pid_based_planner/pid_controller.hpp"
#include "obstacle_cruise_planner/planner_interface.hpp"
#include "signal_processing/lowpass_filter_1d.hpp"
#include "tier4_autoware_utils/system/stop_watch.hpp"

#include "visualization_msgs/msg/marker_array.hpp"

#include <boost/optional.hpp>

#include <memory>
#include <vector>

class PIDBasedPlanner : public PlannerInterface
{
public:
  struct CruiseObstacleInfo
  {
    CruiseObstacleInfo(
      const TargetObstacle & obstacle_arg, const double error_cruise_dist_arg,
      const double dist_to_obstacle_arg, const double target_dist_to_obstacle_arg)
    : obstacle(obstacle_arg),
      error_cruise_dist(error_cruise_dist_arg),
      dist_to_obstacle(dist_to_obstacle_arg),
      target_dist_to_obstacle(target_dist_to_obstacle_arg)
    {
    }

    TargetObstacle obstacle;
    double error_cruise_dist;
    double dist_to_obstacle;
    double target_dist_to_obstacle;
  };

  PIDBasedPlanner(
    rclcpp::Node & node, const LongitudinalInfo & longitudinal_info,
    const vehicle_info_util::VehicleInfo & vehicle_info, const EgoNearestParam & ego_nearest_param);

  Trajectory generateCruiseTrajectory(
    const ObstacleCruisePlannerData & planner_data, boost::optional<VelocityLimit> & vel_limit,
    DebugData & debug_data) override;

  void updateParam(const std::vector<rclcpp::Parameter> & parameters) override;

private:
  void calcObstaclesToCruise(
    const ObstacleCruisePlannerData & planner_data,
    boost::optional<CruiseObstacleInfo> & cruise_obstacle_info);

  Float32MultiArrayStamped getCruisePlanningDebugMessage(
    const rclcpp::Time & current_time) const override
  {
    return cruise_planning_debug_info_.convertToMessage(current_time);
  }

private:
  boost::optional<CruiseObstacleInfo> calcObstacleToCruise(
    const ObstacleCruisePlannerData & planner_data);
  Trajectory planCruise(
    const ObstacleCruisePlannerData & planner_data, boost::optional<VelocityLimit> & vel_limit,
    const boost::optional<CruiseObstacleInfo> & cruise_obstacle_info, DebugData & debug_data);

  // velocity limit based planner
  VelocityLimit doCruiseWithVelocityLimit(
    const ObstacleCruisePlannerData & planner_data,
    const CruiseObstacleInfo & cruise_obstacle_info);

  // velocity insertion based planner
  Trajectory doCruiseWithTrajectory(
    const ObstacleCruisePlannerData & planner_data,
    const CruiseObstacleInfo & cruise_obstacle_info);
  Trajectory getAccelerationLimitedTrajectory(
    const Trajectory traj, const geometry_msgs::msg::Pose & start_pose, const double v0,
    const double a0, const double target_acc, const double target_jerk_ratio) const;

  // ROS parameters
  double min_accel_during_cruise_;
  double min_cruise_target_vel_;

  CruisePlanningDebugInfo cruise_planning_debug_info_;

  struct VelocityLimitBasedPlannerParam
  {
    std::unique_ptr<PIDController> pid_vel_controller;
    double output_ratio_during_accel;
    double vel_to_acc_weight;
    bool enable_jerk_limit_to_output_acc{false};
    bool disable_target_acceleration{false};
  };
  VelocityLimitBasedPlannerParam velocity_limit_based_planner_param_;

  struct VelocityInsertionBasedPlannerParam
  {
    std::unique_ptr<PIDController> pid_acc_controller;
    std::unique_ptr<PIDController> pid_jerk_controller;
    double output_acc_ratio_during_accel;
    double output_jerk_ratio_during_accel;
    bool enable_jerk_limit_to_output_acc{false};
  };
  VelocityInsertionBasedPlannerParam velocity_insertion_based_planner_param_;

  // stop watch
  tier4_autoware_utils::StopWatch<
    std::chrono::milliseconds, std::chrono::microseconds, std::chrono::steady_clock>
    stop_watch_;

  boost::optional<double> prev_target_acc_;

  // lpf for nodes's input
  std::shared_ptr<LowpassFilter1d> lpf_dist_to_obstacle_ptr_;
  std::shared_ptr<LowpassFilter1d> lpf_error_cruise_dist_ptr_;
  std::shared_ptr<LowpassFilter1d> lpf_obstacle_vel_ptr_;

  // lpf for planner's input
  std::shared_ptr<LowpassFilter1d> lpf_normalized_error_cruise_dist_ptr_;

  // lpf for output
  std::shared_ptr<LowpassFilter1d> lpf_output_vel_ptr_;
  std::shared_ptr<LowpassFilter1d> lpf_output_acc_ptr_;
  std::shared_ptr<LowpassFilter1d> lpf_output_jerk_ptr_;

  Trajectory prev_traj_;

  bool use_velocity_limit_based_planner_{true};

  double time_to_evaluate_rss_;

  std::function<double(double)> error_func_;
};

#endif  // OBSTACLE_CRUISE_PLANNER__PID_BASED_PLANNER__PID_BASED_PLANNER_HPP_
