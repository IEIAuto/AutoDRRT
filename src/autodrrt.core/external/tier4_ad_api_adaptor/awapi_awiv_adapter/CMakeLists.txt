cmake_minimum_required(VERSION 3.14)
project(awapi_awiv_adapter)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_executable(awapi_awiv_adapter
  src/awapi_awiv_adapter_node.cpp
  src/awapi_awiv_adapter_core.cpp
  src/awapi_vehicle_state_publisher.cpp
  src/awapi_autoware_state_publisher.cpp
  src/awapi_stop_reason_aggregator.cpp
  src/awapi_v2x_aggregator.cpp
  src/awapi_lane_change_state_publisher.cpp
  src/awapi_obstacle_avoidance_state_publisher.cpp
  src/awapi_max_velocity_publisher.cpp
  src/awapi_autoware_util.cpp
)

ament_auto_package(INSTALL_TO_SHARE launch)
