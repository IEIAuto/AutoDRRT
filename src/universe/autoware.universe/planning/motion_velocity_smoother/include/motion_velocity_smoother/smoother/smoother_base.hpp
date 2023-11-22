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

#ifndef MOTION_VELOCITY_SMOOTHER__SMOOTHER__SMOOTHER_BASE_HPP_
#define MOTION_VELOCITY_SMOOTHER__SMOOTHER__SMOOTHER_BASE_HPP_

#include "motion_utils/trajectory/trajectory.hpp"
#include "motion_velocity_smoother/resample.hpp"
#include "motion_velocity_smoother/trajectory_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tier4_autoware_utils/geometry/geometry.hpp"
#include "vehicle_info_util/vehicle_info_util.hpp"

#include "autoware_auto_planning_msgs/msg/trajectory_point.hpp"

#include "boost/optional.hpp"

#include <limits>
#include <vector>

namespace motion_velocity_smoother
{
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using TrajectoryPoints = std::vector<TrajectoryPoint>;
using vehicle_info_util::VehicleInfoUtil;

class SmootherBase
{
public:
  struct BaseParam
  {
    double max_accel;   // max acceleration in planning [m/s2] > 0
    double min_decel;   // min deceleration in planning [m/s2] < 0
    double stop_decel;  // deceleration at a stop point [m/s2] <= 0
    double max_jerk;
    double min_jerk;
    double max_lateral_accel;                     // max lateral acceleration [m/ss] > 0
    double min_decel_for_lateral_acc_lim_filter;  // deceleration limit applied in the lateral
                                                  // acceleration filter to avoid sudden braking.
    double min_curve_velocity;                    // min velocity at curve [m/s]
    double decel_distance_before_curve;  // distance before slow down for lateral acc at a curve
    double decel_distance_after_curve;   // distance after slow down for lateral acc at a curve
    double max_steering_angle_rate;      // max steering angle rate [degree/s]
    double wheel_base;                   // wheel base [m]
    double sample_ds;                    // distance between trajectory points [m]
    double curvature_threshold;  // look-up distance of Trajectory point for calculation of steering
                                 // angle limit [m]
    double curvature_calculation_distance;  // threshold steering degree limit to trigger
                                            // steeringRateLimit [degree]
    resampling::ResampleParam resample_param;
  };

  explicit SmootherBase(rclcpp::Node & node);
  virtual ~SmootherBase() = default;
  virtual bool apply(
    const double initial_vel, const double initial_acc, const TrajectoryPoints & input,
    TrajectoryPoints & output, std::vector<TrajectoryPoints> & debug_trajectories) = 0;

  virtual TrajectoryPoints resampleTrajectory(
    const TrajectoryPoints & input, const double v0, const geometry_msgs::msg::Pose & current_pose,
    const double nearest_dist_threshold, const double nearest_yaw_threshold) const = 0;

  virtual TrajectoryPoints applyLateralAccelerationFilter(
    const TrajectoryPoints & input, [[maybe_unused]] const double v0 = 0.0,
    [[maybe_unused]] const double a0 = 0.0, [[maybe_unused]] const bool enable_smooth_limit = false,
    const bool use_resampling = true, const double input_points_interval = 1.0) const;

  TrajectoryPoints applySteeringRateLimit(
    const TrajectoryPoints & input, const bool use_resampling = true,
    const double input_points_interval = 1.0) const;

  double getMaxAccel() const;
  double getMinDecel() const;
  double getMaxJerk() const;
  double getMinJerk() const;

  void setParam(const BaseParam & param);
  BaseParam getBaseParam() const;

protected:
  BaseParam base_param_;
};
}  // namespace motion_velocity_smoother

#endif  // MOTION_VELOCITY_SMOOTHER__SMOOTHER__SMOOTHER_BASE_HPP_
