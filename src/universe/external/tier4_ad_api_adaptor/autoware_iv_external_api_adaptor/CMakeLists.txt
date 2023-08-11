cmake_minimum_required(VERSION 3.14)
project(autoware_iv_external_api_adaptor)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/calibration_status.cpp
  src/cpu_usage.cpp
  src/diagnostics.cpp
  src/door.cpp
  src/emergency.cpp
  src/engage.cpp
  src/fail_safe_state.cpp
  src/initial_pose.cpp
  src/localization_score.cpp
  src/map.cpp
  src/metadata_packages.cpp
  src/operator.cpp
  src/rosbag_logging_mode.cpp
  src/route.cpp
  src/rtc_controller.cpp
  src/service.cpp
  src/start.cpp
  src/vehicle_status.cpp
  src/velocity.cpp
  src/version.cpp
)

rclcpp_components_register_nodes(${PROJECT_NAME}
  "external_api::CalibrationStatus"
  "external_api::CpuUsage"
  "external_api::Diagnostics"
  "external_api::Door"
  "external_api::Emergency"
  "external_api::Engage"
  "external_api::FailSafeState"
  "external_api::InitialPose"
  "external_api::LocalizationScore"
  "external_api::Map"
  "external_api::MetadataPackages"
  "external_api::Operator"
  "external_api::RosbagLoggingMode"
  "external_api::Route"
  "external_api::RTCController"
  "external_api::Service"
  "external_api::Start"
  "external_api::VehicleStatus"
  "external_api::Velocity"
  "external_api::Version"
)

ament_auto_package()
