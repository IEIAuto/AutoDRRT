cmake_minimum_required(VERSION 3.14)
project(dma_transfer)

find_package(autoware_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

autoware_package()
include_directories(
  include
  SYSTEM
)

ament_auto_add_library(dma_transfer SHARED
  src/send.cpp
  src/recv.cpp
)

# target_link_libraries(dma_transfer dma_base)

rclcpp_components_register_node(dma_transfer
  PLUGIN "dma_transfer::DemoNode1"
  EXECUTABLE demo_node1
)

rclcpp_components_register_node(dma_transfer
  PLUGIN "dma_transfer::DemoNode2"
  EXECUTABLE demo_node2
)




ament_auto_package(INSTALL_TO_SHARE)
