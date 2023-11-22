// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#ifndef TRAJECTORY_FOLLOWER_NODE__CONTROLLER_NODE_HPP_
#define TRAJECTORY_FOLLOWER_NODE__CONTROLLER_NODE_HPP_

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Geometry"
#include "rclcpp/rclcpp.hpp"
#include "tf2/utils.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "trajectory_follower_base/lateral_controller_base.hpp"
#include "trajectory_follower_base/longitudinal_controller_base.hpp"
#include "trajectory_follower_node/visibility_control.hpp"
#include "vehicle_info_util/vehicle_info_util.hpp"

#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_auto_control_msgs/msg/longitudinal_command.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_vehicle_msgs/msg/vehicle_odometry.hpp"
#include "geometry_msgs/msg/accel_stamped.hpp"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <tier4_planning_msgs/msg/scenario.hpp>
#include "tf2_msgs/msg/tf_message.hpp"


#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::motion::control
{
using trajectory_follower::LateralOutput;
using trajectory_follower::LongitudinalOutput;
namespace trajectory_follower_node
{

using autoware_adapi_v1_msgs::msg::OperationModeState;

namespace trajectory_follower = ::autoware::motion::control::trajectory_follower;

/// \classController
/// \brief The node class used for generating longitudinal control commands (velocity/acceleration)
class TRAJECTORY_FOLLOWER_PUBLIC Controller : public rclcpp::Node
{
public:
  explicit Controller(const rclcpp::NodeOptions & node_options);
  virtual ~Controller() {}

private:
  rclcpp::TimerBase::SharedPtr timer_control_;
  double timeout_thr_sec_;
  boost::optional<LongitudinalOutput> longitudinal_output_{boost::none};

  std::shared_ptr<trajectory_follower::LongitudinalControllerBase> longitudinal_controller_;
  std::shared_ptr<trajectory_follower::LateralControllerBase> lateral_controller_;

  std::string current_scenario_;
  void onScenario(const tier4_planning_msgs::msg::Scenario::ConstSharedPtr msg);
  rclcpp::Subscription<tier4_planning_msgs::msg::Scenario>::SharedPtr sub_scenario_;

  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr sub_ref_path_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr sub_steering_;
  rclcpp::Subscription<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr sub_accel_;
  rclcpp::Subscription<OperationModeState>::SharedPtr sub_operation_mode_;
  rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr
    control_cmd_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_marker_pub_;

  autoware_auto_planning_msgs::msg::Trajectory::SharedPtr current_trajectory_ptr_;
  nav_msgs::msg::Odometry::SharedPtr current_odometry_ptr_;
  autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr current_steering_ptr_;
  geometry_msgs::msg::AccelWithCovarianceStamped::SharedPtr current_accel_ptr_;
  OperationModeState::SharedPtr current_operation_mode_ptr_;

  enum class LateralControllerMode {
    INVALID = 0,
    MPC = 1,
    PURE_PURSUIT = 2,
  };
  enum class LongitudinalControllerMode {
    INVALID = 0,
    PID = 1,
  };

  /**
   * @brief compute control command, and publish periodically
   */
  boost::optional<trajectory_follower::InputData> createInputData(rclcpp::Clock & clock) const;
  void callbackTimerControl();
  void onTrajectory(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr);
  void onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg);
  void onSteering(const autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr msg);
  void onAccel(const geometry_msgs::msg::AccelWithCovarianceStamped::SharedPtr msg);
  bool isTimeOut(const LongitudinalOutput & lon_out, const LateralOutput & lat_out);
  LateralControllerMode getLateralControllerMode(const std::string & algorithm_name) const;
  LongitudinalControllerMode getLongitudinalControllerMode(
    const std::string & algorithm_name) const;
  void publishDebugMarker(
    const trajectory_follower::InputData & input_data,
    const trajectory_follower::LateralOutput & lat_out) const;
};
}  // namespace trajectory_follower_node
}  // namespace autoware::motion::control

#endif  // TRAJECTORY_FOLLOWER_NODE__CONTROLLER_NODE_HPP_
