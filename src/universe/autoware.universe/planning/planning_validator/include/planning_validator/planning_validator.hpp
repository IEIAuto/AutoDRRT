// Copyright 2022 Tier IV, Inc.
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

#ifndef PLANNING_VALIDATOR__PLANNING_VALIDATOR_HPP_
#define PLANNING_VALIDATOR__PLANNING_VALIDATOR_HPP_

#include "planning_validator/debug_marker.hpp"
#include "planning_validator/msg/planning_validator_status.hpp"
#include "vehicle_info_util/vehicle_info_util.hpp"

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <memory>
#include <string>

namespace planning_validator
{
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using diagnostic_updater::DiagnosticStatusWrapper;
using diagnostic_updater::Updater;
using nav_msgs::msg::Odometry;
using planning_validator::msg::PlanningValidatorStatus;

struct ValidationParams
{
  double interval_threshold;
  double relative_angle_threshold;
  double curvature_threshold;
  double lateral_acc_threshold;
  double longitudinal_max_acc_threshold;
  double longitudinal_min_acc_threshold;
  double steering_threshold;
  double steering_rate_threshold;
  double velocity_deviation_threshold;
  double distance_deviation_threshold;
};

class PlanningValidator : public rclcpp::Node
{
public:
  explicit PlanningValidator(const rclcpp::NodeOptions & options);

  void onTrajectory(const Trajectory::ConstSharedPtr msg);

  bool checkValidFiniteValue(const Trajectory & trajectory);
  bool checkValidInterval(const Trajectory & trajectory);
  bool checkValidRelativeAngle(const Trajectory & trajectory);
  bool checkValidCurvature(const Trajectory & trajectory);
  bool checkValidLateralAcceleration(const Trajectory & trajectory);
  bool checkValidMaxLongitudinalAcceleration(const Trajectory & trajectory);
  bool checkValidMinLongitudinalAcceleration(const Trajectory & trajectory);
  bool checkValidSteering(const Trajectory & trajectory);
  bool checkValidSteeringRate(const Trajectory & trajectory);
  bool checkValidVelocityDeviation(const Trajectory & trajectory);
  bool checkValidDistanceDeviation(const Trajectory & trajectory);

private:
  void setupDiag();

  void setupParameters();

  bool isDataReady();

  void validate(const Trajectory & trajectory);

  void publishTrajectory();
  void publishDebugInfo();
  void displayStatus();

  void setStatus(DiagnosticStatusWrapper & stat, const bool & is_ok, const std::string & msg);

  rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_traj_;
  rclcpp::Publisher<Trajectory>::SharedPtr pub_traj_;
  rclcpp::Publisher<PlanningValidatorStatus>::SharedPtr pub_status_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;

  // system parameters
  enum class InvalidTrajectoryHandlingType {
    PUBLISH_AS_IT_IS,
    STOP_PUBLISHING,
    USE_PREVIOUS_RESULT,
  } invalid_trajectory_handling_type_;
  bool publish_diag_ = true;
  int diag_error_count_threshold_ = 0;
  bool display_on_terminal_ = true;

  Updater diag_updater_{this};

  PlanningValidatorStatus validation_status_;
  ValidationParams validation_params_;  // for thresholds

  vehicle_info_util::VehicleInfo vehicle_info_;

  bool isAllValid(const PlanningValidatorStatus & status);

  Trajectory::ConstSharedPtr current_trajectory_;
  Trajectory::ConstSharedPtr previous_published_trajectory_;

  Odometry::ConstSharedPtr current_kinematics_;

  std::shared_ptr<PlanningValidatorDebugMarkerPublisher> debug_pose_publisher_;
};
}  // namespace planning_validator

#endif  // PLANNING_VALIDATOR__PLANNING_VALIDATOR_HPP_
