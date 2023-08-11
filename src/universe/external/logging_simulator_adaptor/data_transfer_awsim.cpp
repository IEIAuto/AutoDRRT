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

#include "data_transfer_awsim.hpp"

#include "tier4_auto_msgs_converter/tier4_auto_msgs_converter.hpp"

#include "tier4_external_api_msgs/msg/gear_shift.hpp"
#include "tier4_external_api_msgs/msg/turn_signal.hpp"
#include "tier4_vehicle_msgs/msg/shift.hpp"
#include "tier4_vehicle_msgs/msg/turn_signal.hpp"
#include "std_msgs/msg/string.hpp"
#include <utility>
  // namespace

namespace logging_simulator_adator
{

DataTransferAwsim::DataTransferAwsim(const rclcpp::NodeOptions & options)
: Node("data_transfer_awsim", options)
{
  using namespace std::literals::chrono_literals;
  std::cout << "JUST FOR TEST FAULT TOLERANCE" << std::endl;
  //message come from autoware to awsim
  // this->pub_ownership_strenth = 0;
  // this->sub_ownership_strenth = 15;
  control_cmd_pub_ =
    this->create_publisher<AckermannControlCommand>("/control/command/control_cmd", rclcpp::QoS(1));
  vehicle_cmd_emergency_pub_ =
    this->create_publisher<VehicleEmergencyStamped>("/control/command/emergency_cmd", rclcpp::QoS(1));
  gear_cmd_pub_ = this->create_publisher<GearCommand>("/control/command/gear_cmd", rclcpp::QoS(1));
  turn_indicator_cmd_pub_ =
    this->create_publisher<TurnIndicatorsCommand>("/control/command/turn_indicators_cmd", rclcpp::QoS(1));
  hazard_light_cmd_pub_ =
    this->create_publisher<HazardLightsCommand>("/control/command/hazard_lights_cmd", rclcpp::QoS(1));

  control_cmd_sub_ = create_subscription<AckermannControlCommand>(
    "/control/command/control_cmd", rclcpp::QoS(1),
    [this](const AckermannControlCommand::ConstSharedPtr msg) {
       control_cmd_pub_ -> publish(*msg);
    });
  vehicle_cmd_emergency_sub_ = create_subscription<VehicleEmergencyStamped>(
  "/control/command/emergency_cmd", rclcpp::QoS(1),
  [this](const VehicleEmergencyStamped::ConstSharedPtr msg) {
      vehicle_cmd_emergency_pub_ -> publish(*msg);
  });
  gear_cmd_sub_ = create_subscription<GearCommand>(
  "/control/command/gear_cmd", rclcpp::QoS(1),
  [this](const GearCommand::ConstSharedPtr msg) {
      gear_cmd_pub_ -> publish(*msg);
  });
  turn_indicator_cmd_sub_ = create_subscription<TurnIndicatorsCommand>(
  "/control/command/turn_indicators_cmd", rclcpp::QoS(1),
  [this](const TurnIndicatorsCommand::ConstSharedPtr msg) {
      turn_indicator_cmd_pub_ -> publish(*msg);
  });
  hazard_light_cmd_sub_ = create_subscription<HazardLightsCommand>(
  "/control/command/hazard_lights_cmd", rclcpp::QoS(1),
  [this](const HazardLightsCommand::ConstSharedPtr msg) {
      hazard_light_cmd_pub_ -> publish(*msg);
  });


  //message come from awsim to autoware
  // this->pub_ownership_strenth = 15;
  // this->sub_ownership_strenth = 0;
  camera_info_pub_ =
  this->create_publisher<sensor_msgs::msg::CameraInfo>("/sensing/camera/traffic_light/camera_info",rclcpp::SensorDataQoS());
  rclcpp::QoS custom_qos(5);
  custom_qos.best_effort();
  camera_image_pub_ =
  this->create_publisher<sensor_msgs::msg::Image>("/sensing/camera/traffic_light/image_raw", custom_qos);
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
  "/sensing/camera/traffic_light/camera_info", rclcpp::SensorDataQoS(),
  [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
      camera_info_pub_ -> publish(*msg);
  });
  camera_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
  "/sensing/camera/traffic_light/image_raw", custom_qos,
  [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
      camera_image_pub_ -> publish(*msg);
  });

  gnss_pose_pub_ =
  this->create_publisher<geometry_msgs::msg::PoseStamped>("/sensing/gnss/pose", rclcpp::QoS(1));
  gnss_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
  "/sensing/gnss/pose", rclcpp::QoS(1),
  [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
      // std::cout << "/sensing/gnss/pose" << std::endl;
      gnss_pose_pub_ -> publish(*msg);
  });


  gnss_pose_with_cov_pub_ =
  this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/sensing/gnss/pose_with_covariance", rclcpp::QoS(1));
  gnss_pose_with_cov_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
  "/sensing/gnss/pose_with_covariance", rclcpp::QoS(1),
  [this](const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
      // std::cout << "/sensing/gnss/pose_with_covariance" << std::endl;
      gnss_pose_with_cov_pub_ -> publish(*msg);
  });

  pub_imu_raw_ =
  this->create_publisher<sensor_msgs::msg::Imu>("/sensing/imu/tamagawa/imu_raw", rclcpp::QoS(1));
  sub_imu_raw_ = create_subscription<sensor_msgs::msg::Imu>(
  "/sensing/imu/tamagawa/imu_raw", rclcpp::QoS(1),
  [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    // std::cout << "/sensing/gnss/pose_with_covariance" << std::endl;
      pub_imu_raw_ -> publish(*msg);
  });


  pub_velocity_ =
    this->create_publisher<VelocityReport>("/vehicle/status/velocity_status", rclcpp::QoS(1));
  pub_steer_ =
    this->create_publisher<SteeringReport>("/vehicle/status/steering_status", rclcpp::QoS(1));
  pub_control_mode_report_ = this->create_publisher<ControlModeReport>("/vehicle/status/control_mode", rclcpp::QoS(1));
  pub_gear_report_ =
    this->create_publisher<GearReport>("/vehicle/status/gear_status", rclcpp::QoS(1));
  pub_turn_indicators_report_ =
    this->create_publisher<TurnIndicatorsReport>("/vehicle/status/turn_indicators_status", rclcpp::QoS(1));
  pub_hazard_lights_report_ =
    this->create_publisher<HazardLightsReport>("/vehicle/status/hazard_lights_status", rclcpp::QoS(1));

   
  sub_velocity_ = create_subscription<VelocityReport>(
    "/vehicle/status/velocity_status", rclcpp::QoS(1),
    [this](const VelocityReport::ConstSharedPtr msg) {
       pub_velocity_ -> publish(*msg);
    });
  sub_steer_ = create_subscription<SteeringReport>(
  "/vehicle/status/steering_status", rclcpp::QoS(1),
  [this](const SteeringReport::ConstSharedPtr msg) {
      pub_steer_ -> publish(*msg);
  });
  sub_control_mode_report_ = create_subscription<ControlModeReport>(
  "/vehicle/status/control_mode", rclcpp::QoS(1),
  [this](const ControlModeReport::ConstSharedPtr msg) {
      pub_control_mode_report_ -> publish(*msg);
  });
  sub_gear_report_ = create_subscription<GearReport>(
  "/vehicle/status/gear_status", rclcpp::QoS(1),
  [this](const GearReport::ConstSharedPtr msg) {
      pub_gear_report_ -> publish(*msg);
  });
  sub_turn_indicators_report_ = create_subscription<TurnIndicatorsReport>(
  "/vehicle/status/turn_indicators_status", rclcpp::QoS(1),
  [this](const TurnIndicatorsReport::ConstSharedPtr msg) {
      pub_turn_indicators_report_ -> publish(*msg);
  });
   sub_hazard_lights_report_ = create_subscription<HazardLightsReport>(
  "/vehicle/status/hazard_lights_status", rclcpp::QoS(1),
  [this](const HazardLightsReport::ConstSharedPtr msg) {
      pub_hazard_lights_report_ -> publish(*msg);
  });





  pub_pointcloud_raw_ =
  this->create_publisher<sensor_msgs::msg::PointCloud2>("/sensing/lidar/top/pointcloud_raw", rclcpp::SensorDataQoS().keep_last(5));
  sub_pointcloud_raw_ = create_subscription<sensor_msgs::msg::PointCloud2>(
  "/sensing/lidar/top/pointcloud_raw", rclcpp::SensorDataQoS().keep_last(5),
  [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    // std::cout << "/sensing/lidar/top/pointcloud_raw" << std::endl;
      pub_pointcloud_raw_ -> publish(*msg);
  });

  pub_pointcloud_raw_ex_ =
  this->create_publisher<sensor_msgs::msg::PointCloud2>("/sensing/lidar/top/pointcloud_raw_ex", rclcpp::SensorDataQoS().keep_last(5));
  sub_pointcloud_raw_ex_ = create_subscription<sensor_msgs::msg::PointCloud2>(
  "/sensing/lidar/top/pointcloud_raw_ex", rclcpp::SensorDataQoS().keep_last(5),
  [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    // std::cout << "/sensing/lidar/top/pointcloud_raw_ex" << std::endl;
      pub_pointcloud_raw_ex_ -> publish(*msg);
  });

  // this->sub_ownership_strenth = 15;
  auto sub_pointcloud_raw_test = create_subscription<sensor_msgs::msg::PointCloud2>(
  "/sensing/lidar/top/self_cropped/pointcloud_ex", rclcpp::SensorDataQoS().keep_last(5),
  [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {

    std::cout << "/sensing/lidar/top/self_cropped/pointcloud_ex" << std::endl;

  });

  auto sub_pointcloud_raw_test_mirror = create_subscription<sensor_msgs::msg::PointCloud2>(
  "/sensing/lidar/top/mirror_cropped/pointcloud_ex", rclcpp::SensorDataQoS().keep_last(5),
  [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    std::cout << "/sensing/lidar/top/mirror_cropped/pointcloud_ex" << std::endl;
  });
  


}


}  // namespace external_api

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(logging_simulator_adator::DataTransferAwsim)
