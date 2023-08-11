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

#ifndef DATA_TRANSFER_HPP_
#define DATA_TRANSFER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "tier4_api_utils/tier4_api_utils.hpp"

#include "autoware_auto_control_msgs/msg/ackermann_control_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/gear_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/hazard_lights_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/steering_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/turn_indicators_report.hpp"
#include "autoware_auto_vehicle_msgs/msg/velocity_report.hpp"

#include "autoware_auto_vehicle_msgs/msg/gear_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/hazard_lights_command.hpp"
#include "autoware_auto_vehicle_msgs/msg/turn_indicators_command.hpp"





#include "tier4_external_api_msgs/msg/vehicle_command_stamped.hpp"
#include "tier4_external_api_msgs/msg/vehicle_status_stamped.hpp"
#include "autoware_auto_vehicle_msgs/msg/control_mode_report.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include "std_msgs/msg/string.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tier4_vehicle_msgs/msg/vehicle_emergency_stamped.hpp>

// #include "ublox_msgs/ublox_msgs.h"

namespace logging_simulator_adator
{
using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_vehicle_msgs::msg::GearCommand;
using autoware_auto_vehicle_msgs::msg::HazardLightsCommand;
using autoware_auto_vehicle_msgs::msg::SteeringReport;
using autoware_auto_vehicle_msgs::msg::TurnIndicatorsCommand;
using autoware_auto_vehicle_msgs::msg::ControlModeReport;
using autoware_auto_vehicle_msgs::msg::GearReport;
using autoware_auto_vehicle_msgs::msg::HazardLightsReport;
using autoware_auto_vehicle_msgs::msg::TurnIndicatorsReport;
using autoware_auto_vehicle_msgs::msg::VelocityReport;
using tier4_vehicle_msgs::msg::VehicleEmergencyStamped;
class DataTransferAwsim : public rclcpp::Node
{
public:
  explicit DataTransferAwsim(const rclcpp::NodeOptions & options);

private:
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr control_cmd_pub_;
  rclcpp::Publisher<VehicleEmergencyStamped>::SharedPtr vehicle_cmd_emergency_pub_;
  rclcpp::Publisher<GearCommand>::SharedPtr gear_cmd_pub_;
  rclcpp::Publisher<TurnIndicatorsCommand>::SharedPtr turn_indicator_cmd_pub_;
  rclcpp::Publisher<HazardLightsCommand>::SharedPtr hazard_light_cmd_pub_;

  rclcpp::Subscription<AckermannControlCommand>::SharedPtr control_cmd_sub_;
  rclcpp::Subscription<VehicleEmergencyStamped>::SharedPtr vehicle_cmd_emergency_sub_;
  rclcpp::Subscription<GearCommand>::SharedPtr gear_cmd_sub_;
  rclcpp::Subscription<TurnIndicatorsCommand>::SharedPtr turn_indicator_cmd_sub_;
  rclcpp::Subscription<HazardLightsCommand>::SharedPtr hazard_light_cmd_sub_;

  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_image_pub_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_image_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr gnss_pose_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gnss_pose_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr gnss_pose_with_cov_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr gnss_pose_with_cov_sub_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_raw_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_raw_;

  rclcpp::Subscription<VelocityReport>::SharedPtr sub_velocity_;
  rclcpp::Subscription<SteeringReport>::SharedPtr sub_steer_;
  rclcpp::Subscription<ControlModeReport>::SharedPtr sub_control_mode_report_;
  rclcpp::Subscription<GearReport>::SharedPtr sub_gear_report_;
  rclcpp::Subscription<TurnIndicatorsReport>::SharedPtr sub_turn_indicators_report_;
  rclcpp::Subscription<HazardLightsReport>::SharedPtr sub_hazard_lights_report_;

  rclcpp::Publisher<VelocityReport>::SharedPtr pub_velocity_;
  rclcpp::Publisher<SteeringReport>::SharedPtr pub_steer_;
  rclcpp::Publisher<ControlModeReport>::SharedPtr pub_control_mode_report_;
  rclcpp::Publisher<GearReport>::SharedPtr pub_gear_report_;
  rclcpp::Publisher<TurnIndicatorsReport>::SharedPtr pub_turn_indicators_report_;
  rclcpp::Publisher<HazardLightsReport>::SharedPtr pub_hazard_lights_report_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pointcloud_raw_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pointcloud_raw_ex_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_raw_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_raw_ex_;


};

}  // namespace external_api

#endif  // VEHICLE_STATUS_HPP_
