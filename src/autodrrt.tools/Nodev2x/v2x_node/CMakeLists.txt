cmake_minimum_required(VERSION 3.14)
project(node_v2x)

find_package(autoware_cmake REQUIRED)

autoware_package()
include_directories(
  include
  SYSTEM
)

ament_auto_add_library(node_v2x SHARED
  src/send.cpp
  src/recv.cpp
)

rclcpp_components_register_node(node_v2x
  PLUGIN "node_v2x::NodeV2X1"
  EXECUTABLE v2x_node1
)

rclcpp_components_register_node(node_v2x
  PLUGIN "node_v2x::NodeV2X2"
  EXECUTABLE v2x_node2
)

ament_auto_package(INSTALL_TO_SHARE)
