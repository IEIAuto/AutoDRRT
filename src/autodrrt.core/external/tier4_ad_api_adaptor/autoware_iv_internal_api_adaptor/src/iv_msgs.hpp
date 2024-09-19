// Copyright 2021 TIER IV, Inc.
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

#ifndef IV_MSGS_HPP_
#define IV_MSGS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <autoware_adapi_v1_msgs/msg/mrm_state.hpp>
#include <autoware_auto_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_system_msgs/msg/autoware_state.hpp>
#include <autoware_auto_vehicle_msgs/msg/control_mode_report.hpp>
#include <tier4_perception_msgs/msg/dynamic_object_array.hpp>
#include <tier4_planning_msgs/msg/trajectory.hpp>
#include <tier4_system_msgs/msg/autoware_state.hpp>

namespace internal_api
{
class IVMsgs : public rclcpp::Node
{
public:
  explicit IVMsgs(const rclcpp::NodeOptions & options);

private:
  using EmergencyStateAuto = autoware_adapi_v1_msgs::msg::MrmState;
  using AutowareStateAuto = autoware_auto_system_msgs::msg::AutowareState;
  using AutowareStateIV = tier4_system_msgs::msg::AutowareState;
  rclcpp::Subscription<EmergencyStateAuto>::SharedPtr sub_emergency_;
  rclcpp::Subscription<AutowareStateAuto>::SharedPtr sub_state_;
  rclcpp::Publisher<AutowareStateIV>::SharedPtr pub_state_;

  using ControlModeAuto = autoware_auto_vehicle_msgs::msg::ControlModeReport;
  rclcpp::Subscription<ControlModeAuto>::SharedPtr sub_control_mode_;
  rclcpp::Publisher<ControlModeAuto>::SharedPtr pub_control_mode_;

  using TrajectoryAuto = autoware_auto_planning_msgs::msg::Trajectory;
  using TrajectoryIV = tier4_planning_msgs::msg::Trajectory;
  rclcpp::Subscription<TrajectoryAuto>::SharedPtr sub_trajectory_;
  rclcpp::Publisher<TrajectoryIV>::SharedPtr pub_trajectory_;

  using TrackedObjectsAuto = autoware_auto_perception_msgs::msg::TrackedObjects;
  using DynamicObjectsIV = tier4_perception_msgs::msg::DynamicObjectArray;
  rclcpp::Subscription<TrackedObjectsAuto>::SharedPtr sub_tracked_objects_;
  rclcpp::Publisher<DynamicObjectsIV>::SharedPtr pub_dynamic_objects_;

  void onState(const AutowareStateAuto::ConstSharedPtr message);
  void onEmergency(const EmergencyStateAuto::ConstSharedPtr message);
  void onControlMode(const ControlModeAuto::ConstSharedPtr message);
  void onTrajectory(const TrajectoryAuto::ConstSharedPtr message);
  void onTrackedObjects(const TrackedObjectsAuto::ConstSharedPtr message);

  bool is_emergency_;
};

}  // namespace internal_api

#endif  // IV_MSGS_HPP_
