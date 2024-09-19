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

#ifndef PATH_SMOOTHER__ELASTIC_BAND_SMOOTHER_HPP_
#define PATH_SMOOTHER__ELASTIC_BAND_SMOOTHER_HPP_

#include "motion_utils/trajectory/trajectory.hpp"
#include "path_smoother/common_structs.hpp"
#include "path_smoother/elastic_band.hpp"
#include "path_smoother/replan_checker.hpp"
#include "path_smoother/type_alias.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tier4_autoware_utils/ros/logger_level_configure.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace path_smoother
{
class ElasticBandSmoother : public rclcpp::Node
{
public:
  explicit ElasticBandSmoother(const rclcpp::NodeOptions & node_options);

protected:
  class DrivingDirectionChecker
  {
  public:
    bool isDrivingForward(const std::vector<PathPoint> & path_points)
    {
      const auto is_driving_forward = motion_utils::isDrivingForward(path_points);
      is_driving_forward_ = is_driving_forward ? is_driving_forward.value() : is_driving_forward_;
      return is_driving_forward_;
    }

  private:
    bool is_driving_forward_{true};
  };
  DrivingDirectionChecker driving_direction_checker_{};

  // argument variables
  mutable std::shared_ptr<TimeKeeper> time_keeper_ptr_{nullptr};

  // flags for some functions
  bool enable_debug_info_;

  // algorithms
  std::shared_ptr<EBPathSmoother> eb_path_smoother_ptr_{nullptr};
  std::shared_ptr<ReplanChecker> replan_checker_ptr_{nullptr};

  // parameters
  CommonParam common_param_{};
  EgoNearestParam ego_nearest_param_{};

  // variables for subscribers
  Odometry::ConstSharedPtr ego_state_ptr_;

  // variables for previous information
  std::shared_ptr<std::vector<TrajectoryPoint>> prev_optimized_traj_points_ptr_;

  // interface publisher
  rclcpp::Publisher<Trajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<Path>::SharedPtr path_pub_;

  // interface subscriber
  rclcpp::Subscription<Path>::SharedPtr path_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;

  // debug publisher
  rclcpp::Publisher<Trajectory>::SharedPtr debug_extended_traj_pub_;
  rclcpp::Publisher<StringStamped>::SharedPtr debug_calculation_time_str_pub_;
  rclcpp::Publisher<Float64Stamped>::SharedPtr debug_calculation_time_float_pub_;

  // parameter callback
  rcl_interfaces::msg::SetParametersResult onParam(
    const std::vector<rclcpp::Parameter> & parameters);
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  // subscriber callback function
  void onPath(const Path::ConstSharedPtr path_ptr);

  // reset functions
  void initializePlanning();
  void resetPreviousData();

  // main functions
  bool isDataReady(const Path & path, rclcpp::Clock clock) const;
  void applyInputVelocity(
    std::vector<TrajectoryPoint> & output_traj_points,
    const std::vector<TrajectoryPoint> & input_traj_points,
    const geometry_msgs::msg::Pose & ego_pose) const;
  std::vector<TrajectoryPoint> extendTrajectory(
    const std::vector<TrajectoryPoint> & traj_points,
    const std::vector<TrajectoryPoint> & optimized_points) const;

  std::unique_ptr<tier4_autoware_utils::LoggerLevelConfigure> logger_configure_;
};
}  // namespace path_smoother

#endif  // PATH_SMOOTHER__ELASTIC_BAND_SMOOTHER_HPP_
