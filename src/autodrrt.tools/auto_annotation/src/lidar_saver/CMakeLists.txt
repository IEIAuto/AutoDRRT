cmake_minimum_required(VERSION 3.8)
project(lidar_saver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
ament_auto_add_executable(lidar_saver_node 
    src/lidar_saver_node.cpp
  )
ament_target_dependencies(lidar_saver_node)

ament_auto_package(
  INSTALL_TO_SHARE
    )

