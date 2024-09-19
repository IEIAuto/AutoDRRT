cmake_minimum_required(VERSION 3.5)
project(pointcloud_to_laserscan)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(PCL REQUIRED QUIET COMPONENTS common)

include_directories(
  ${PCL_COMMON_INCLUDE_DIRS}
)

ament_auto_add_library(laserscan_to_pointcloud SHARED
  src/laserscan_to_pointcloud_node.cpp)

rclcpp_components_register_node(laserscan_to_pointcloud
  PLUGIN "pointcloud_to_laserscan::LaserScanToPointCloudNode"
  EXECUTABLE laserscan_to_pointcloud_node)

ament_auto_add_library(pointcloud_to_laserscan SHARED
  src/pointcloud_to_laserscan_node.cpp)

target_link_libraries(pointcloud_to_laserscan ${PCL_LIBRARIES})

if(${tf2_sensor_msgs_VERSION} VERSION_LESS 0.18.0)
  target_compile_definitions(laserscan_to_pointcloud PUBLIC
    USE_TF2_SENSOR_MSGS_DEPRECATED_HEADER
  )
  target_compile_definitions(pointcloud_to_laserscan PUBLIC
    USE_TF2_SENSOR_MSGS_DEPRECATED_HEADER
  )
endif()

rclcpp_components_register_node(pointcloud_to_laserscan
  PLUGIN "pointcloud_to_laserscan::PointCloudToLaserScanNode"
  EXECUTABLE pointcloud_to_laserscan_node)

ament_auto_add_executable(dummy_pointcloud_publisher
  src/dummy_pointcloud_publisher.cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    launch
)
